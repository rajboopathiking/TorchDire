"""
Assemble paper/REPORT.md from the aggregated JSON of all three tracks.

Reads whatever it can find and degrades gracefully: a track with no data becomes
a "not yet run" note rather than a crash, so the report is useful from the first
partial run onward. Every number printed here comes from an aggregate produced by
`aggregate_runs` / `aggregate_synthetic`, i.e. mean +/- sample std over seeds with
t-based 95% CIs — nothing is transcribed by hand.

Input discovery (recursive):
    results_aggregated.json   -> zero-shot track  (scripts/review_experiments.py)
    finetune_aggregated.json  -> fine-tuning A/B  (scripts/finetune_qgfd.py)
    synthetic_aggregated.json -> multi-hop probes (scripts/eval_synthetic.py)

Usage
-----
    python scripts/build_report.py --scan ./qgfd_results ./qgfd_finetune_results \
                                   --out paper/REPORT.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_FILES = {
    "results_aggregated.json": "zeroshot",
    "finetune_aggregated.json": "finetune",
    "synthetic_aggregated.json": "synthetic",
}
Agg = Tuple[str, Dict]          # (path, aggregate dict)


def _pm(s: Optional[Dict], prec: int = 3) -> str:
    """'mean ± std' for a _stat() dict; '—' when the metric is absent."""
    if not s:
        return "—"
    return f"{s['mean']:.{prec}f} ± {s['std']:.{prec}f}"


def _sig(s: Optional[Dict]) -> str:
    """Significance marker on a PAIRED statistic."""
    if not s or s.get("n", 0) < 2:
        return "n/a"
    return "**\\***" if abs(s["mean"]) > s["ci95"] > 0 else "ns"


def _git_commit(repo_root: str) -> str:
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def discover(roots: Sequence[str]) -> Dict[str, List[Agg]]:
    """Walk `roots` and bucket every aggregate JSON found by track."""
    found: Dict[str, List[Agg]] = {"zeroshot": [], "finetune": [], "synthetic": []}
    seen = set()
    for root in roots:
        if os.path.isfile(root):
            pairs = [(os.path.basename(root), root)]
        else:
            pairs = [(f, os.path.join(d, f))
                     for d, _, fs in os.walk(root) for f in fs if f in _FILES]
        for name, path in pairs:
            real = os.path.realpath(path)
            if real in seen or name not in _FILES:
                continue
            seen.add(real)
            with open(path) as fh:
                agg = json.load(fh)
            # meta.track is authoritative; the filename is only the fallback.
            track = agg.get("meta", {}).get("track", _FILES[name])
            found.setdefault(track, []).append((path, agg))
    for k in found:
        found[k].sort(key=lambda p: p[1].get("meta", {}).get("model_id", ""))
    return found

# __CHUNK2__


# --------------------------------------------------------------------------- #
# Narrative sections
# --------------------------------------------------------------------------- #
def _headline(zs: List[Agg]) -> str:
    """One sentence stating the robustness result, or admitting there isn't one."""
    if not zs:
        return ("_No zero-shot results ingested yet, so the headline claim is "
                "unsupported. Run `scripts/review_experiments.py --seeds 0,1,2`._")
    wins, losses, ns, n_seeds = [], [], [], set()
    for _, agg in zs:
        gaps = agg.get("paired", {}).get("robustness_gap_pct", {})
        if not gaps:
            continue
        rate = max(gaps, key=lambda r: float(r))
        s = gaps[rate]
        n_seeds.add(agg["meta"]["n_seeds"])
        model = agg["meta"]["model_id"].split("/")[-1]
        entry = f"{model} ({s['mean']:+.2f} pp)"
        (wins if s["mean"] > 0 else losses).append(entry)
        if _sig(s) == "ns":
            ns.append(model)
    if not wins and not losses:
        return "_No paired robustness statistics found in the ingested files._"
    parts = [f"At the highest tested noise rate QGFD degraded **less** than eager "
             f"softmax on {len(wins)}/{len(wins) + len(losses)} models"]
    if wins:
        parts.append(f"({', '.join(wins)})")
    if losses:
        parts.append(f"and **more** on {', '.join(losses)}")
    txt = " ".join(parts) + f", over n={sorted(n_seeds)} seeds."
    if ns:
        txt += (f" The paired 95% CI still includes zero for {', '.join(ns)} — "
                f"treat those as directional, not established.")
    return txt


def _abstract(zs: List[Agg]) -> List[str]:
    return [
        "## Abstract", "",
        "Softmax attention routes information in a single hop and is brittle to "
        "input perturbation: a small shift in key embeddings can collapse "
        "probability mass onto the wrong tokens. **QGFD (Query–Graph Flow "
        "Diffusion)** reframes attention refinement as a short Markovian random "
        "walk over a key-similarity graph "
        "`P = softmax(KKᵀ/√d)`, mixing `p⁽ᵗ⁺¹⁾ = (1−α)·p⁽⁰⁾ + α·(p⁽ᵗ⁾P)`.", "",
        "Contributions:", "",
        "1. **Exact softmax-equivalence at α=0**, so QGFD is a safe drop-in.",
        "2. **Training-free robustness**: QGFD lowers perplexity *degradation* "
        "under character-level input noise at negligible clean-perplexity cost.",
        "3. A **single-GPU LoRA recipe** (q/k/v/o adapters + α warmup) that lets a "
        "pretrained model adapt to the diffused distribution.",
        "4. A controlled **induction / passkey** demonstration of multi-hop routing.",
        "", f"**Headline result.** {_headline(zs)}", "",
    ]

# __CHUNK3__


def _design() -> List[str]:
    return [
        "## Tools and System Design", "",
        "| Component | Where | Role |",
        "| --- | --- | --- |",
        "| `QGFDOperator` / `SoftmaxOperator` | `torchdire/nn/attention_operators.py` "
        "| The probability operator. Both arms run through the same adapter, so the "
        "*only* difference is how `p` is computed. |",
        "| `LlamaAttentionAdapter` (+ Qwen2 / Mistral) | "
        "`torchdire/nn/attention_adapters.py` | Replaces a stock attention module "
        "and calls the operator on the materialised score matrix. |",
        "| `wrap_model_with_qgfd_operator` | `torchdire/nn/universal_qgfd_replacer.py` "
        "| Architecture dispatch; refuses to silently no-op on unsupported models. |",
        "| `register_qgfd_operator_step_callback` | same | Drives α warmup from a "
        "`TrainerCallback`, i.e. from *outside* `forward()`. |",
        "| `scripts/review_experiments.py` | — | Zero-shot track: perplexity, noise "
        "robustness, attention statistics, latency; `run_all_seeds` aggregates. |",
        "| `scripts/finetune_qgfd.py` | — | Fine-tuning A/B: LoRA-only vs LoRA+QGFD "
        "at an identical budget. |",
        "| `scripts/eval_synthetic.py` | — | Multi-hop probes: induction (with "
        "context corruption) and passkey retrieval. |",
        "| `scripts/build_report.py` | — | This document. |", "",
        "### Design decisions that affect how the numbers must be read", "",
        "* **The baseline is eager materialised softmax, not SDPA/FlashAttention.** "
        "QGFD is defined over the explicit probability matrix, so it is "
        "architecturally incompatible with fused kernels. Every latency figure is "
        "*QGFD vs eager softmax*; against a FlashAttention baseline the overhead "
        "would be larger.",
        "* **`is_causal=True` everywhere.** An unmasked `P` diffuses probability "
        "mass onto *future* keys, which artificially deflates teacher-forced "
        "perplexity. Causal masking of `P` is required for any honest LM number.",
        "* **α warmup is driven by a `TrainerCallback`, not by `forward()`.** "
        "Mutating step state inside `forward` diverges on gradient-checkpoint "
        "recompute and raises `CheckpointError`.",
        "* **LoRA must land on the live projections.** The adapter aliases "
        "`q/k/v/o` onto itself, and `nn.Module.named_modules()` de-duplicates "
        "shared submodules — PEFT used to inject adapters into a module `forward()` "
        "never called. `verify_lora_live()` now runs a probe backward pass and "
        "refuses to report a result if no adapter receives gradient "
        "(`tests/test_operator_lora_liveness.py`).",
        "* **Paired statistics.** Within a seed both arms see identical texts and "
        "an identical noise realisation, so the per-seed QGFD−softmax difference "
        "cancels between-seed corpus variance. For effects of ~1% it is the only "
        "statistic with the resolution to say anything.",
        "* **Not used for any reported number:** `QGFDAblator`'s ROUGE/BLEU/BERT "
        "figures are hard-coded heuristics and `QGFDProfiler`'s FLOPs/VRAM are "
        "analytic estimates. They can indicate direction, nothing more.", "",
    ]

# __CHUNK4__


def _protocol(found: Dict[str, List[Agg]]) -> List[str]:
    models = sorted({a["meta"]["model_id"]
                     for track in found.values() for _, a in track})
    seeds = sorted({tuple(a["meta"]["seeds"])
                    for track in found.values() for _, a in track})
    lines = [
        "## Experimental Protocol", "",
        "**Arms.** (A) eager softmax baseline · (B) QGFD zero-shot · "
        "(C) LoRA-only fine-tuned · (D) QGFD+LoRA fine-tuned. C and D share seed, "
        "data, learning rate, schedule and step count.", "",
        f"**Models.** {', '.join('`' + m + '`' for m in models) or '_none ingested_'}",
        "",
        f"**Seeds.** {', '.join(str(list(s)) for s in seeds) or '_none_'} — all "
        "metrics are mean ± sample std with two-sided t-based 95% CIs.", "",
        "**Datasets.** WikiText-2 for perplexity and character-noise robustness; a "
        "WikiText-2 train slice for LoRA; synthetic induction and passkey prompts "
        "generated against each model's own tokenizer.", "",
        "**Metrics.** Clean perplexity · perplexity under character-level noise "
        "with per-seed degradation Δ% · attention entropy and position-0 sink mass "
        "· prefill latency and peak VRAM · induction and passkey exact-match "
        "accuracy.", "",
        "> The synthetic probes are **not** built on "
        "`torchdire/benchmarks/dataset.py`'s `GraphMultiHopDataset` / "
        "`PasskeyRetrievalDataset`. Those emit random token IDs over a synthetic "
        "vocabulary for from-scratch models; fed to a *pretrained* SLM they measure "
        "nothing. Both tasks are rebuilt as natural-language prompts instead.", "",
    ]
    return lines


def _table_zeroshot(zs: List[Agg]) -> List[str]:
    lines = ["### Table 1 — Clean perplexity and noise robustness (zero-shot)", ""]
    if not zs:
        return lines + ["_Not yet run._", ""]
    lines += [
        "| Model | seeds | Clean PPL softmax | Clean PPL QGFD | "
        "Paired Δ clean | noise | Δ% softmax | Δ% QGFD | Gap (pp) | sig |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, agg in zs:
        m, sm, qg = agg["meta"], agg["arms"]["softmax"], agg["arms"]["qgfd"]
        gaps = agg["paired"].get("robustness_gap_pct", {})
        pair = agg["paired"].get("clean_ppl_qgfd_minus_softmax")
        name = m["model_id"].split("/")[-1]
        first = True
        for rate in sorted(gaps, key=float):
            head = ([f"`{name}`", str(m["n_seeds"]), _pm(sm["clean_ppl"]),
                     _pm(qg["clean_ppl"]),
                     f"{_pm(pair, 4)} {_sig(pair)}"] if first
                    else ["", "", "", "", ""])
            lines.append("| " + " | ".join(head + [
                f"{float(rate) * 100:.0f}%",
                _pm(sm["robustness_delta_pct"][rate], 1),
                _pm(qg["robustness_delta_pct"][rate], 1),
                f"{gaps[rate]['mean']:+.2f} ± {gaps[rate]['std']:.2f}",
                _sig(gaps[rate]),
            ]) + " |")
            first = False
    lines += ["", "Gap = softmax Δ% − QGFD Δ%, computed within each seed. "
              "**Positive means QGFD degraded less.** `*` marks a paired 95% CI "
              "that excludes zero; `ns` means it does not.", ""]
    return lines

# __CHUNK5__


def _table_attention_latency(zs: List[Agg]) -> List[str]:
    rows = []
    for _, agg in zs:
        sm, qg = agg["arms"]["softmax"], agg["arms"]["qgfd"]
        if "attention" not in sm and "latency" not in sm:
            continue
        m = agg["meta"]
        att_s, att_q = sm.get("attention", {}), qg.get("attention", {})
        lat_s, lat_q = sm.get("latency", {}), qg.get("latency", {})
        rows.append("| " + " | ".join([
            f"`{m['model_id'].split('/')[-1]}`",
            _pm(att_s.get("mean_attention_entropy_nats")),
            _pm(att_q.get("mean_attention_entropy_nats")),
            _pm(att_s.get("mean_sink_mass_pos0"), 4),
            _pm(att_q.get("mean_sink_mass_pos0"), 4),
            _pm(lat_s.get("prefill_ms"), 1),
            _pm(lat_q.get("prefill_ms"), 1),
            _pm(agg["paired"].get("latency_overhead_x"), 2),
        ]) + " |")
    lines = ["### Table 2 — Attention statistics and compute overhead", ""]
    if not rows:
        return lines + ["_Not yet run._", ""]
    return lines + [
        "| Model | Entropy softmax (nats) | Entropy QGFD | Sink@0 softmax | "
        "Sink@0 QGFD | Prefill softmax (ms) | Prefill QGFD (ms) | Overhead × |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ] + rows + [
        "", "Overhead is measured against **eager** softmax. Diffusion adds a "
        "`K·Kᵀ` GEMM plus one `p·P` product per step, and it forecloses fused "
        "attention kernels entirely — that, not the arithmetic, is the real cost.",
        "",
    ]


def _table_finetune(ft: List[Agg]) -> List[str]:
    lines = ["### Table 3 — LoRA fine-tuning A/B (equal budget)", ""]
    if not ft:
        return lines + ["_Not yet run._", ""]
    lines += [
        "| Model | seeds | steps | Final loss softmax | Final loss QGFD | "
        "Clean PPL softmax | Clean PPL QGFD | Max-noise gap (pp) | sig |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, agg in ft:
        m, tr = agg["meta"], agg.get("train", {})
        gaps = agg["paired"].get("robustness_gap_pct", {})
        rate = max(gaps, key=float) if gaps else None
        lines.append("| " + " | ".join([
            f"`{m['model_id'].split('/')[-1]}`", str(m["n_seeds"]),
            str(m.get("max_steps", "—")),
            _pm(tr.get("softmax", {}).get("final_loss"), 4),
            _pm(tr.get("qgfd", {}).get("final_loss"), 4),
            _pm(agg["arms"]["softmax"]["clean_ppl"]),
            _pm(agg["arms"]["qgfd"]["clean_ppl"]),
            (f"{gaps[rate]['mean']:+.2f} ± {gaps[rate]['std']:.2f} "
             f"@ {float(rate) * 100:.0f}%") if rate else "—",
            _sig(gaps[rate]) if rate else "n/a",
        ]) + " |")
    lines += ["", "Both arms carry identical LoRA adapters on q/k/v/o, so the "
              "comparison isolates the operator rather than adapter capacity. The "
              "question this table answers is whether *letting the model adapt* to "
              "the diffused distribution recovers the clean-perplexity cost that "
              "zero-shot QGFD pays.", ""]
    return lines

# __CHUNK6__


def _table_synthetic(sy: List[Agg]) -> List[str]:
    lines = ["### Table 4 — Synthetic multi-hop probes", ""]
    if not sy:
        return lines + ["_Not yet run._", ""]
    lines += [
        "| Model | mode | seeds | Induction softmax | Induction QGFD | "
        "Control (floor) | Passkey softmax | Passkey QGFD | Induction gap | sig |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, agg in sy:
        m, sm, qg = agg["meta"], agg["arms"]["softmax"], agg["arms"]["qgfd"]
        g = agg.get("paired", {}).get("induction_gap")
        lines.append("| " + " | ".join([
            f"`{m['model_id'].split('/')[-1]}`",
            "post-LoRA" if m.get("post_lora") else "zero-shot", str(m["n_seeds"]),
            _pm(sm["induction_acc"], 4), _pm(qg["induction_acc"], 4),
            _pm(sm["induction_control_acc"], 4),
            _pm(sm["passkey_acc"], 4), _pm(qg["passkey_acc"], 4),
            f"{g['mean']:+.4f} ± {g['std']:.4f}" if g else "—", _sig(g),
        ]) + " |")

    noisy = [(a, r) for _, a in sy
             for r in a["arms"]["softmax"].get("induction_by_noise", {})
             if r != "0.00"]
    if noisy:
        lines += ["", "**Induction under context corruption.** A fraction of the "
                  "second copy is replaced by unrelated words; corrupted positions "
                  "are excluded from scoring, so the remaining ones are well-posed "
                  "but must route through a garbled context.", "",
                  "| Model | corruption | softmax | QGFD | gap | sig |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for agg, r in noisy:
            gp = agg.get("paired", {}).get("induction_gap_by_noise", {}).get(r)
            lines.append("| " + " | ".join([
                f"`{agg['meta']['model_id'].split('/')[-1]}`",
                f"{float(r) * 100:.0f}%",
                _pm(agg["arms"]["softmax"]["induction_by_noise"][r], 4),
                _pm(agg["arms"]["qgfd"]["induction_by_noise"][r], 4),
                f"{gp['mean']:+.4f} ± {gp['std']:.4f}" if gp else "—", _sig(gp),
            ]) + " |")
    lines += ["", "*Control* scores the **first** copy, where the successor is "
              "genuinely unpredictable — the chance-level floor. If induction "
              "accuracy is not far above it, nothing was learned in context and the "
              "row is uninformative regardless of which arm wins.", ""]
    return lines


def _figures(zs: List[Agg], out_path: str) -> List[str]:
    """Embed each track's plot, relative to the report's own directory."""
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    lines: List[str] = []
    for path, agg in zs:
        png = os.path.join(os.path.dirname(path), "robustness_aggregated.png")
        if not os.path.exists(png):
            continue
        rel = os.path.relpath(os.path.abspath(png), out_dir)
        name = agg["meta"]["model_id"].split("/")[-1]
        lines += [f"**{name}** — perplexity vs noise rate (error bars = sample std) "
                  f"and the paired per-seed robustness gap.", "",
                  f"![robustness — {name}]({rel})", ""]
    if not lines:
        return ["### Figures", "", "_No plots found. They are written next to each "
                "`results_aggregated.json` when matplotlib is installed._", ""]
    return ["### Figures", ""] + lines

# __CHUNK7__


def _caveats(found: Dict[str, List[Agg]]) -> List[str]:
    """Threats to validity, with the data-dependent ones filled in from the runs."""
    lines = ["## Threats to Validity", "",
             "Fixed properties of the setup:", "",
             "* The baseline is **eager** softmax. QGFD cannot use fused attention "
             "kernels, so a production comparison would show a larger gap than the "
             "latency table does.",
             "* Character-level noise is a proxy for input perturbation, not a "
             "natural distribution shift.",
             "* Only Llama, Mistral and Qwen2 attention are wired up; "
             "`verify_patch()` refuses anything else rather than silently no-op.",
             "* Exact-match accuracy is coarse. At α=0.05 the arms disagree on ~1% "
             "of argmax predictions, so identical synthetic scores are an expected "
             "small-sample outcome — `meta.operator` in the synthetic JSON records "
             "the live α, which is what distinguishes that from an inactive patch.",
             ]
    ns_hits, small_n = [], []
    for track, entries in found.items():
        for _, agg in entries:
            m = agg["meta"]
            name = f"{m['model_id'].split('/')[-1]} ({track})"
            if m["n_seeds"] < 3:
                small_n.append(f"{name}: n={m['n_seeds']}")
            gaps = agg.get("paired", {}).get("robustness_gap_pct", {})
            for r, s in gaps.items():
                if _sig(s) == "ns":
                    ns_hits.append(f"{name} @ {float(r) * 100:.0f}% noise")
    if small_n or ns_hits:
        lines += ["", "Observed in the ingested runs:", ""]
    if small_n:
        lines.append(f"* **Under-powered:** {'; '.join(sorted(set(small_n)))}. The "
                     f"t-critical value for n=2 is 12.7, so almost nothing can reach "
                     f"significance — these are pilot numbers, not results.")
    if ns_hits:
        lines.append(f"* **Paired CI includes zero** for: "
                     f"{'; '.join(sorted(set(ns_hits)))}. Directional only.")
    return lines + [""]


def _repro(found: Dict[str, List[Agg]]) -> List[str]:
    models = sorted({a["meta"]["model_id"]
                     for t in found.values() for _, a in t}) or \
        ["HuggingFaceTB/SmolLM2-135M"]
    cmds = []
    for mid in models:
        cmds += [
            f"# {mid}",
            f"python scripts/review_experiments.py --model_id {mid} --seeds 0,1,2",
            f"python scripts/finetune_qgfd.py     --model_id {mid} --seeds 0,1,2",
            f"python scripts/eval_synthetic.py    --model_id {mid} --seeds 0,1,2",
            "",
        ]
    return ["## Reproduction", "", "```bash", *cmds,
            "python scripts/build_report.py --scan . --out paper/REPORT.md",
            "```", "",
            "CPU smoke versions of all three take a couple of minutes: add "
            "`--quick --model_id JackFram/llama-160m --device cpu --dtype float32`.",
            ""]

# __CHUNK8__


def build_report(found: Dict[str, List[Agg]], out_path: str,
                 repo_root: Optional[str] = None) -> str:
    repo_root = repo_root or os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
    zs, ft, sy = found.get("zeroshot", []), found.get("finetune", []), \
        found.get("synthetic", [])
    n_files = sum(len(v) for v in found.values())

    lines = [
        "# QGFD: Query–Graph Flow Diffusion for Attention Refinement", "",
        "> Auto-generated by `scripts/build_report.py` — do not edit by hand; "
        "re-run it instead.", "",
        "| | |", "| --- | --- |",
        f"| Generated | {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} |",
        f"| Commit | `{_git_commit(repo_root)}` |",
        f"| Aggregates ingested | {n_files} "
        f"(zero-shot {len(zs)}, fine-tune {len(ft)}, synthetic {len(sy)}) |",
        "",
        *_abstract(zs),
        *_design(),
        *_protocol(found),
        "## Results", "",
        *_table_zeroshot(zs),
        *_table_attention_latency(zs),
        *_table_finetune(ft),
        *_table_synthetic(sy),
        *_figures(zs, out_path),
        *_caveats(found),
        *_repro(found),
        "## Source files", "",
    ]
    for track in ("zeroshot", "finetune", "synthetic"):
        for path, _ in found.get(track, []):
            rel = os.path.relpath(path, repo_root)
            # Aggregates written outside the repo (e.g. /tmp during a smoke run)
            # produce a relpath full of "..", which is worse than the absolute path.
            lines.append(f"* `{path if rel.startswith('..') else rel}` — {track}")
    if n_files == 0:
        lines.append("_None found._")
    text = "\n".join(lines).rstrip() + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(text)
    return text


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build paper/REPORT.md from aggregates")
    p.add_argument("--scan", nargs="+", default=["."],
                   help="Directories (or JSON files) to search for aggregates")
    p.add_argument("--out", default="paper/REPORT.md")
    a = p.parse_args(argv)

    found = discover(a.scan)
    n = sum(len(v) for v in found.values())
    for track, entries in found.items():
        for path, agg in entries:
            print(f"  [{track}] {agg['meta']['model_id']} "
                  f"n={agg['meta']['n_seeds']}  <- {path}")
    build_report(found, a.out)
    print(f"\nWrote {a.out} from {n} aggregate file(s).")
    if n == 0:
        print("WARNING: no aggregates found — the report is a skeleton. Run the "
              "track scripts with --seeds first.")


if __name__ == "__main__":
    main()
