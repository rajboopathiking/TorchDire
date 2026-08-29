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
    mechanism_results.json    -> E2-E8 falsifiers (scripts/mechanism_experiments.py)

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


MECH_FILE = "mechanism_results.json"


def discover_mechanism(roots: Sequence[str]) -> List[Agg]:
    """
    Collect `mechanism_results.json` files (Track 5, E2-E8).

    Kept out of `discover()` deliberately: those files carry a `config` block
    rather than the `meta` block every other aggregate has, and a mechanism run is
    single-shot rather than seed-aggregated. Bucketing it with the rest would make
    `agg["meta"]["n_seeds"]` blow up in three unrelated call sites.
    """
    out: List[Agg] = []
    seen = set()
    for root in roots:
        paths = ([root] if os.path.isfile(root) and os.path.basename(root) == MECH_FILE
                 else [os.path.join(d, f) for d, _, fs in os.walk(root)
                       for f in fs if f == MECH_FILE])
        for path in paths:
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            try:
                with open(path) as fh:
                    out.append((path, json.load(fh)))
            except (OSError, json.JSONDecodeError):
                continue
    out.sort(key=lambda p: p[1].get("config", {}).get("model_id", ""))
    return out


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
    verdict = _absolute_verdict(zs)
    if verdict == "backwards":
        txt += (" **That is a relative statistic and it does not survive E9:** with "
                "the clean-perplexity denominator held fixed, QGFD's absolute "
                "perplexity under noise is significantly *higher* (Table 1b).")
    elif verdict in ("not_survived", "no_positive"):
        txt += (" **E9 does not support reading this as robustness:** no model keeps "
                "a positive residual once the clean-perplexity denominator is "
                "controlled for (Table 1b).")
    elif verdict == "partly":
        txt += (" Part of that gap is a denominator artefact — quote the residual in "
                "Table 1b.")
    return txt


def _gap_at_worst_noise(agg: Dict) -> Optional[Dict]:
    """The paired robustness gap at the highest noise rate this run measured."""
    gaps = agg.get("paired", {}).get("robustness_gap_pct", {})
    if not gaps:
        return None
    return gaps[max(gaps, key=lambda r: float(r))]


def _contributions(zs: List[Agg], sy: List[Agg]) -> List[str]:
    """
    Contributions 2 and 4 are claims about measured behaviour, so they are written
    from the ingested numbers rather than asserted.

    A hard-coded "QGFD lowers degradation under noise" survived a run in which the
    gap reversed sign on the largest model, which is exactly the failure mode an
    auto-generated report is supposed to make impossible. Claim 1 stays fixed: it
    is an algebraic identity, verified bit-exact by E1, not an empirical trend.

    Contribution 2 is additionally gated on E9 (`_absolute_verdict`): a *relative*
    degradation gap can be produced by QGFD's larger clean-perplexity denominator
    alone, so the abstract may not claim a robustness effect that Table 1b reports
    as arithmetic.
    """
    means = [g["mean"] for _, a in zs if (g := _gap_at_worst_noise(a))]
    sig = [g["mean"] for _, a in zs
           if (g := _gap_at_worst_noise(a)) and _sig(g) != "ns"]
    if not means:
        c2 = ("**Training-free robustness** — _claim unevaluated: no paired "
              "robustness statistics ingested._")
    elif all(m > 0 for m in means) and len(sig) == len(means):
        c2 = ("**Training-free robustness**: QGFD lowers perplexity *degradation* "
              "under character-level input noise on every model tested, at a small "
              "but measurable clean-perplexity cost.")
    elif any(m > 0 for m in means):
        c2 = (f"**A scale-dependent robustness effect**, reported as such: the gap "
              f"is positive on {sum(m > 0 for m in means)}/{len(means)} models and "
              f"negative on {sum(m <= 0 for m in means)}, and it *shrinks* as "
              f"parameter count grows (see Table 1a). This is not a general "
              f"robustness claim and must not be written as one.")
    else:
        c2 = ("**No robustness benefit was found.** The gap is non-positive on "
              "every model tested; the clean-perplexity cost is real. Contribution "
              "(2) as originally stated is falsified.")

    verdict = _absolute_verdict(zs)
    if means and verdict == "backwards":
        c2 = ("**No robustness benefit survives the denominator control, and the "
              "absolute effect runs the other way.** The reported degradation gap "
              "is a difference of *relative* degradations, and QGFD's higher clean "
              "perplexity inflates it mechanically; once that is removed, QGFD's "
              "**absolute** perplexity under noise is significantly *higher* than "
              "the softmax arm's (Table 1b). Contribution (2) is falsified — the "
              "statistic was measuring the clean-perplexity cost.")
    elif means and verdict in ("not_survived", "no_positive"):
        c2 = ("**No robustness benefit survives the denominator control.** The "
              "reported degradation gap is a difference of *relative* degradations "
              "and is inflated by QGFD's higher clean perplexity; no model shows a "
              "positive residual whose 95% CI excludes zero (Table 1b). "
              "Contribution (2) is not supported by these runs.")
    elif means and verdict == "partly":
        c2 += (" Part of that gap is a denominator artefact of QGFD's higher clean "
               "perplexity; the surviving residual is in Table 1b and is the number "
               "to quote.")

    ind = [(a["meta"]["model_id"].split("/")[-1],
            a.get("paired", {}).get("induction_gap")) for _, a in sy]
    ind = [(m, s) for m, s in ind if s]
    if not ind:
        c4 = ("A controlled **induction / passkey** probe of multi-hop routing "
              "— _not yet run_.")
    elif all(s["mean"] < 0 for _, s in ind):
        worse = [m for m, s in ind if _sig(s) != "ns"]
        c4 = ("A controlled **induction / passkey** probe which QGFD does **not** "
              "win: the induction gap is negative on every model"
              + (f", significantly so on {', '.join(worse)}" if worse else "")
              + ". Passkey is at ceiling for both arms and discriminates nothing. "
                "Reported as a negative result.")
    else:
        c4 = ("A controlled **induction / passkey** demonstration of multi-hop "
              "routing.")
    return [c2, c4]


def _abstract(zs: List[Agg], sy: Optional[List[Agg]] = None) -> List[str]:
    c2, c4 = _contributions(zs, sy or [])
    return [
        "## Abstract", "",
        "Softmax attention routes information in a single hop and is brittle to "
        "input perturbation: a small shift in key embeddings can collapse "
        "probability mass onto the wrong tokens. **QGFD (Query–Graph Flow "
        "Diffusion)** reframes attention refinement as a short Markovian random "
        "walk over a key-similarity graph "
        "`P = softmax(KKᵀ/√d)`, mixing `p⁽ᵗ⁺¹⁾ = (1−α)·p⁽⁰⁾ + α·(p⁽ᵗ⁾P)`.", "",
        "Contributions (2 and 4 are written from the ingested numbers, not "
        "asserted):", "",
        "1. **Exact softmax-equivalence at α=0**, so QGFD is a safe drop-in.",
        f"2. {c2}",
        "3. A **single-GPU LoRA recipe** (q/k/v/o adapters + α warmup) that lets a "
        "pretrained model adapt to the diffused distribution.",
        f"4. {c4}",
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


def _scale_trend(zs: List[Agg]) -> List[str]:
    """
    Table 1a — the gap against model scale, and against the damage it repairs.

    Two things Table 1 cannot show. First, a gap in percentage *points* against a
    baseline that degraded by four figures reads far larger than it is, so the
    repaired fraction (gap / softmax Δ%) is given alongside it. Second, three
    models ordered by parameter count is not a regression, but a monotone decline
    ending in a sign flip is still the single most important thing in the run and
    must not be left for the reader to assemble from three separate rows.
    """
    lines = ["### Table 1a — Does the effect survive scale?", ""]
    rows = []
    for _, agg in zs:
        g = _gap_at_worst_noise(agg)
        if not g:
            continue
        m = agg["meta"]
        rate = max(agg["paired"]["robustness_gap_pct"], key=float)
        deg = agg["arms"]["softmax"]["robustness_delta_pct"][rate]["mean"]
        rows.append({
            "name": m["model_id"].split("/")[-1],
            "n_params": m.get("n_params"),
            "clean": agg["arms"]["softmax"]["clean_ppl"]["mean"],
            "rate": float(rate), "gap": g["mean"], "sig": _sig(g), "deg": deg,
            "frac": (g["mean"] / deg * 100.0) if deg else None,
            "cost": (agg["paired"].get("clean_ppl_qgfd_minus_softmax") or {}).get("mean"),
        })
    if not rows:
        return lines + ["_Not yet run._", ""]

    # Order by measured capability when parameter counts are missing (older
    # aggregates predate meta.n_params); lower clean PPL = more capable.
    known = all(r["n_params"] for r in rows)
    rows.sort(key=(lambda r: r["n_params"]) if known else (lambda r: -r["clean"]))
    lines += [
        "| Model | params | Clean PPL | noise | softmax Δ% | Gap (pp) | sig | "
        "Gap as % of the damage | Clean cost (PPL) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append("| " + " | ".join([
            f"`{r['name']}`",
            f"{r['n_params'] / 1e6:.0f}M" if r["n_params"] else "—",
            f"{r['clean']:.2f}", f"{r['rate'] * 100:.0f}%", f"{r['deg']:.0f}%",
            f"{r['gap']:+.2f}", r["sig"],
            f"{r['frac']:.2f}%" if r["frac"] is not None else "—",
            f"{r['cost']:+.4f}" if r["cost"] is not None else "—",
        ]) + " |")

    gaps = [r["gap"] for r in rows]
    monotone_down = all(b <= a for a, b in zip(gaps, gaps[1:])) and len(gaps) > 1
    flips = any(g <= 0 for g in gaps) and any(g > 0 for g in gaps)
    order = "parameter count" if known else "measured capability (clean PPL)"
    lines += ["", f"Rows are ordered by {order}, smallest/weakest first. "
              "**Gap as % of the damage** is the paired gap divided by the "
              "baseline's own degradation — the fraction of the damage QGFD "
              "actually repairs."]
    if monotone_down and flips:
        lines.append(
            f"\n**The effect does not survive scale.** The gap declines "
            f"monotonically across all {len(gaps)} models "
            f"({' → '.join(f'{g:+.2f}' for g in gaps)} pp) and changes sign on the "
            f"largest. On the evidence here the benefit is a small-model artefact, "
            f"and the paper must say so in the abstract rather than in a caveat. "
            f"E3 and E4 (Table 5) are what decide whether even the small-model gap "
            f"is a QGFD effect at all.")
    elif flips:
        lines.append(
            f"\n**The sign is not consistent across models** "
            f"({' → '.join(f'{g:+.2f}' for g in gaps)} pp). Any claim has to be "
            f"conditioned on the model, not stated in general.")
    elif monotone_down:
        lines.append(
            f"\n**The gap shrinks with scale** "
            f"({' → '.join(f'{g:+.2f}' for g in gaps)} pp) without yet reversing. "
            f"Extrapolating to a larger model is not supported either way; say that "
            f"instead of implying the effect holds.")
    else:
        lines.append(
            f"\nNo monotone trend in the gap across these models "
            f"({' → '.join(f'{g:+.2f}' for g in gaps)} pp). With n={len(gaps)} "
            f"models that is weak evidence of scale-independence, not evidence of it.")
    if max(abs(r["frac"] or 0.0) for r in rows) < 5.0:
        lines.append(
            "\nNote the last-but-one column: every gap here repairs **under 5%** of "
            "the perplexity damage the noise causes. A four-figure degradation is "
            "not meaningfully mitigated by any arm in this table.")
    lines.append("")
    return lines


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _stat(values: Sequence[float]) -> Dict:
    """Same shape and t-critical table as review_experiments._stat.

    Duplicated rather than imported because that module pulls in torch and
    transformers, and building the report must work from the JSON alone.
    """
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": 0.0, "sem": 0.0, "ci95": 0.0,
                "n": 0, "values": []}
    mean = sum(vals) / n
    if n == 1:
        std = sem = ci = 0.0
    else:
        std = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5
        sem = std / n ** 0.5
        ci = _T95.get(n - 1, 1.96) * sem
    return {"mean": mean, "std": std, "sem": sem, "ci95": ci, "n": n,
            "values": vals}


def _decompose(agg: Dict) -> Optional[Dict]:
    """
    Split the reported relative robustness gap into a denominator artefact and a
    residual, per seed, at the highest noise rate measured. Returns None when the
    aggregate lacks per-seed `values` or a second noise rate.

    The reported statistic is a difference of *relative* degradations,
    `100·(noisy−clean)/clean`. QGFD's clean perplexity is higher, so its
    denominator is larger, and a larger denominator gives a smaller Δ% **for the
    same noisy perplexity** — which manufactures a positive gap out of nothing.
    Holding the noisy perplexity at the softmax arm's value and swapping only the
    denominator isolates that component:

        artefact_pp = 100·noisy_sm·(1/clean_sm − 1/clean_qgfd)
        residual_pp = observed_pp − artefact_pp
                    = 100·(noisy_sm − noisy_qgfd)/clean_qgfd

    so the residual is positive if and only if QGFD's **absolute** perplexity
    under noise is genuinely lower. No new GPU time: this re-reads aggregates that
    already exist.
    """
    sm, qg = agg["arms"]["softmax"], agg["arms"]["qgfd"]
    rates = sm.get("robustness", {})
    if not rates:
        return None
    base, worst = min(rates, key=float), max(rates, key=float)
    if worst == base:
        return None
    try:
        c_s = sm["robustness"][base]["values"]
        n_s = sm["robustness"][worst]["values"]
        c_q = qg["robustness"][base]["values"]
        n_q = qg["robustness"][worst]["values"]
    except KeyError:
        return None
    if not c_s or not (len(c_s) == len(n_s) == len(c_q) == len(n_q)):
        return None
    seeds = range(len(c_s))
    observed = _stat([100.0 * (n_s[i] - c_s[i]) / c_s[i]
                      - 100.0 * (n_q[i] - c_q[i]) / c_q[i] for i in seeds])
    artefact = _stat([100.0 * n_s[i] * (1.0 / c_s[i] - 1.0 / c_q[i]) for i in seeds])
    residual = _stat([100.0 * (n_s[i] - n_q[i]) / c_q[i] for i in seeds])
    return {
        "name": agg["meta"]["model_id"].split("/")[-1],
        "n_params": agg["meta"].get("n_params"),
        "rate": float(worst), "observed": observed, "artefact": artefact,
        "residual": residual,
        "abs_gap": _stat([n_s[i] - n_q[i] for i in seeds]),
        "share": (100.0 * artefact["mean"] / observed["mean"]
                  if observed["mean"] > 0 else None),
    }


def _classify_residuals(rows: List[Dict]) -> Tuple[List, List, List, str]:
    """
    (significantly-positive rows, significantly-negative rows, artefact-dominated
    rows, verdict) for a set of `_decompose` results.

    The verdict is the single word the abstract and Table 1b must agree on, so it
    lives here rather than being re-derived in each place.
    """
    pos = [r for r in rows
           if r["residual"]["mean"] > 0 and _sig(r["residual"]) != "ns"]
    neg = [r for r in rows
           if r["residual"]["mean"] < 0 and _sig(r["residual"]) != "ns"]
    dominated = [r for r in rows if r["share"] is not None and r["share"] >= 50.0]
    if neg and not pos:
        verdict = "backwards"
    elif not pos and dominated:
        verdict = "not_survived"
    elif not pos:
        verdict = "no_positive"
    elif dominated:
        verdict = "partly"
    else:
        verdict = "survives"
    return pos, neg, dominated, verdict


def _absolute_verdict(zs: List[Agg]) -> Optional[str]:
    """E9's verdict for the abstract, or None when the decomposition is unavailable."""
    rows = [d for _, agg in zs if (d := _decompose(agg))]
    return _classify_residuals(rows)[3] if rows else None


def _denominator_check(zs: List[Agg]) -> List[str]:
    """Table 1b — E9. See `_decompose` for the arithmetic."""
    lines = ["### Table 1b — Is the gap a denominator artefact? (E9)", ""]
    rows = [d for _, agg in zs if (d := _decompose(agg))]
    if not rows:
        return lines + ["_Not yet run._ Needs a zero-shot aggregate with per-seed "
                        "`values` arrays and at least two noise rates.", ""]

    known = all(r["n_params"] for r in rows)
    rows.sort(key=(lambda r: r["n_params"]) if known
              else (lambda r: -r["observed"]["mean"]))
    lines += [
        "| Model | noise | Observed gap (pp) | Denominator-only (pp) | "
        "share | Residual (pp) | sig | Absolute PPL under noise (softmax − QGFD) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append("| " + " | ".join([
            f"`{r['name']}`", f"{r['rate'] * 100:.0f}%",
            f"{r['observed']['mean']:+.2f} ± {r['observed']['std']:.2f}",
            f"{r['artefact']['mean']:+.2f} ± {r['artefact']['std']:.2f}",
            f"{r['share']:.0f}%" if r["share"] is not None else "n/a",
            f"{r['residual']['mean']:+.2f} ± {r['residual']['std']:.2f}",
            _sig(r["residual"]),
            f"{r['abs_gap']['mean']:+.2f} ± {r['abs_gap']['std']:.2f}",
        ]) + " |")

    pos, neg, dominated, verdict = _classify_residuals(rows)
    lines += ["", "**Denominator-only** is what the gap would be if QGFD's "
              "perplexity under noise were *identical* to the softmax arm's and "
              "only its clean perplexity differed — pure arithmetic, no robustness. "
              "**Residual** is the rest, and it is positive exactly when QGFD's "
              "absolute perplexity under noise is lower (last column). `share` is "
              "the denominator-only component as a fraction of the reported gap, "
              "and is left blank where the reported gap is not positive."]
    if verdict == "backwards":
        lines.append(
            f"\n**The gap is an artefact, and the underlying sign is backwards.** On "
            f"{len(neg)}/{len(rows)} model(s) QGFD's *absolute* perplexity under "
            f"noise is significantly **higher** than the softmax arm's, while the "
            f"reported relative gap still reads positive. The reported statistic is "
            f"measuring QGFD's clean-perplexity cost, not its robustness. It cannot "
            f"be used as the headline; report the absolute column instead.")
    elif verdict == "not_survived":
        lines.append(
            f"\n**The gap does not survive the control.** On "
            f"{len(dominated)}/{len(rows)} model(s) at least half of the reported "
            f"gap is explained by QGFD's larger clean-perplexity denominator, and "
            f"no model shows a *positive* residual whose 95% CI excludes zero. "
            f"There is no evidence QGFD is more accurate under noise in absolute "
            f"terms. The headline statistic must be replaced by the absolute column "
            f"or abandoned.")
    elif verdict == "no_positive":
        lines.append(
            "\n**No model has a positive residual distinguishable from zero.** "
            "Whatever the denominator share, there is no evidence here that QGFD's "
            "absolute perplexity under noise is lower. Report the absolute column.")
    elif verdict == "partly":
        lines.append(
            f"\n**Partly artefact.** {len(pos)}/{len(rows)} model(s) keep a "
            f"significant positive residual, but on {len(dominated)} the "
            f"denominator alone accounts for half or more of the reported gap. "
            f"Quote the residual, not the raw gap.")
    else:
        lines.append(
            f"\n**The gap survives the control** on {len(pos)}/{len(rows)} "
            f"model(s): the residual is positive with a 95% CI excluding zero, so "
            f"QGFD's absolute perplexity under noise really is lower. Still quote "
            f"the residual alongside the raw gap — the denominator contributes to "
            f"both.")
    lines.append("")
    return lines


def _mech_verdict(key: str, r: Dict) -> Tuple[str, str]:
    """(finding, what it means for the paper) for one mechanism experiment."""
    if key == "E2":
        cur = r.get("clean_curvature") or {}
        fit = (cur.get("T1") or next(iter(cur.values()), {})).get("fit", {})
        k = fit.get("exponent_k")
        if k is None:
            return f"no fit ({fit.get('note', 'insufficient points')})", "inconclusive"
        ok = fit.get("consistent_with_quadratic")
        return (f"k = {k:.2f} (predicted 2.0), R² = {fit.get('r2', float('nan')):.3f}",
                "the O(α²) cost account holds" if ok else
                "cost is **not** quadratic — the curvature argument does not describe it")
    if key == "E3":
        p = r.get("paired") or {}
        if not p:
            return "no arms completed", "inconclusive"
        worst = min(p.values(), key=lambda v: v["qgfd_minus_temp_pp"])
        d = worst["qgfd_minus_temp_pp"]
        return (f"α={worst['alpha']:.3f}: QGFD {worst['qgfd_gap_pp']:+.2f} pp vs "
                f"entropy-matched τ={worst['matched_tau']:.3f} "
                f"{worst['temp_gap_pp']:+.2f} pp → **{d:+.2f} pp**",
                "QGFD beats the free control" if d > 0 else
                "**a free temperature rescale matches or beats QGFD — no mechanism "
                "contribution survives**")
    if key == "E4":
        v = r.get("verdict") or {}
        if not v:
            return "no verdict", "inconclusive"
        return (f"real {v['real_gap_pp']:+.2f} pp vs best control "
                f"`{v['best_control']}` {v['best_control_gap_pp']:+.2f} pp → margin "
                f"**{v['margin_pp']:+.2f} pp**", v["reading"])
    if key == "E6":
        v = r.get("verdict") or {}
        by = r.get("by_corruption") or {}
        det = "; ".join(f"{n} {d['robustness_gap_pp']:+.2f} pp" for n, d in by.items())
        return det or "no corruptions run", v.get("reading", "inconclusive")
    if key == "E7":
        v = r.get("verdict") or {}
        if not v:
            return "no verdict", "inconclusive"
        f = (f"best T = {v['best_T']} (induction {v['best_induction_acc']:.4f} vs "
             f"softmax {v['softmax_induction_acc']:.4f}, floor "
             f"{v['control_acc_floor']:.4f}), shape `{v['shape']}`")
        return f, v.get("warning") or v.get("reading", "")
    if key == "E8":
        c = r.get("comparison") or {}
        if not c:
            return "no comparison", "inconclusive"
        return (f"QGFD overhead {c['qgfd_overhead_x']:.2f}×; under noise "
                f"**{c['winner_under_noise']}** wins, cheaper: "
                f"{c['latency_cheaper']}", c["reading"])
    return "—", "—"


def _table_mechanism(mech: List[Agg]) -> List[str]:
    try:
        # Local, and tolerated failing: mechanism_experiments imports torch, and
        # building the report must stay possible on a machine that only has the JSON.
        from scripts.mechanism_experiments import MECHANISM_EXPERIMENTS
    except Exception:                                             # noqa: BLE001
        MECHANISM_EXPERIMENTS = {}

    lines = ["### Table 5 — Mechanism falsifiers (E2–E8)", "",
             "Each row is a control designed so that one specific outcome ends one "
             "specific claim. Every arm within a run shares one loaded checkpoint, "
             "one α mutated in place, and byte-identical corrupted text, so nothing "
             "here is confounded by seed variance. E1 (α=0 bit-exactness) is checked "
             "in the driver notebook, not here.", ""]
    if not mech:
        return lines + ["_Not yet run._ Run Track 5 of "
                        "`QGFD_Paper_Experiments.ipynb`, or "
                        "`python scripts/mechanism_experiments.py --model_id <id>`.", ""]
    for path, blob in mech:
        model = blob.get("config", {}).get("model_id", "?")
        req = blob.get("requested", [])
        res, errs = blob.get("results", {}), blob.get("errors", {})
        lines += [f"**`{model}`** — {len(res)}/{len(req)} completed"
                  + ("  _(quick sizing: plumbing check, not a result)_"
                     if blob.get("quick") else ""), "",
                  "| # | Experiment | Finding | Reading | A negative result kills |",
                  "| --- | --- | --- | --- | --- |"]
        for key in req:
            title, kills = MECHANISM_EXPERIMENTS.get(key, (key, "—"))
            if key in errs:
                finding, reading = f"**FAILED** — `{errs[key]}`", "no result"
            else:
                finding, reading = _mech_verdict(key, res.get(key, {}))
            lines.append(f"| {key} | {title} | {finding} | {reading} | {kills} |")
        lines.append("")
    return lines


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
             "* `robustness_gap_pct` is a difference of **relative** degradations, "
             "so it is sensitive to each arm's clean perplexity: raising QGFD's "
             "clean PPL enlarges its denominator and shrinks its Δ% at no change in "
             "accuracy. Table 1b (E9) separates that arithmetic from the real "
             "component and is the number any claim should rest on.",
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
                 repo_root: Optional[str] = None,
                 mechanism: Optional[List[Agg]] = None) -> str:
    repo_root = repo_root or os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))
    zs, ft, sy = found.get("zeroshot", []), found.get("finetune", []), \
        found.get("synthetic", [])
    mech = mechanism or []
    n_files = sum(len(v) for v in found.values()) + len(mech)

    lines = [
        "# QGFD: Query–Graph Flow Diffusion for Attention Refinement", "",
        "> Auto-generated by `scripts/build_report.py` — do not edit by hand; "
        "re-run it instead.", "",
        "| | |", "| --- | --- |",
        f"| Generated | {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} |",
        f"| Commit | `{_git_commit(repo_root)}` |",
        f"| Aggregates ingested | {n_files} "
        f"(zero-shot {len(zs)}, fine-tune {len(ft)}, synthetic {len(sy)}, "
        f"mechanism {len(mech)}) |",
        "",
        *_abstract(zs, sy),
        *_design(),
        *_protocol(found),
        "## Results", "",
        *_table_zeroshot(zs),
        *_scale_trend(zs),
        *_denominator_check(zs),
        *_table_attention_latency(zs),
        *_table_finetune(ft),
        *_table_synthetic(sy),
        *_table_mechanism(mech),
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
    for path, _ in mech:
        rel = os.path.relpath(path, repo_root)
        lines.append(f"* `{path if rel.startswith('..') else rel}` — mechanism")
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
    mech = discover_mechanism(a.scan)
    n = sum(len(v) for v in found.values()) + len(mech)
    for track, entries in found.items():
        for path, agg in entries:
            print(f"  [{track}] {agg['meta']['model_id']} "
                  f"n={agg['meta']['n_seeds']}  <- {path}")
    for path, blob in mech:
        res = blob.get("results", {})
        print(f"  [mechanism] {blob.get('config', {}).get('model_id', '?')} "
              f"{'+'.join(res)}  <- {path}")
    build_report(found, a.out, mechanism=mech)
    print(f"\nWrote {a.out} from {n} aggregate file(s).")
    if n == 0:
        print("WARNING: no aggregates found — the report is a skeleton. Run the "
              "track scripts with --seeds first.")


if __name__ == "__main__":
    main()
