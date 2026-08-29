"""
QGFD synthetic multi-hop probes  (paper track 3)
================================================
Two controlled tasks that isolate *multi-hop routing* rather than language
modelling quality, run for both arms (softmax vs QGFD), zero-shot and optionally
after the LoRA fine-tune of track 2.

    induction : a random word sequence S is presented twice. At every position of
                the second copy the model must emit the token that followed the
                same word in the first copy. This is the canonical 2-hop circuit
                ("match the earlier occurrence, then copy its successor") and it
                is exactly the routing QGFD's key-graph walk is meant to help.
                A single forward pass yields len(S)-1 scored predictions.

    passkey   : a 5-digit key is buried at a controlled depth inside filler text;
                the model must retrieve it at the end. Tests long-range retrieval
                against a distractor-heavy context.

Deviation from the original plan (worth knowing)
------------------------------------------------
`torchdire/benchmarks/dataset.py` already has `GraphMultiHopDataset` and
`PasskeyRetrievalDataset`, but both emit RANDOM TOKEN IDs over a synthetic vocab —
they were built for from-scratch models. Feeding random ids to a *pretrained* SLM
measures nothing (the ids are out-of-distribution noise), so both tasks here are
rebuilt as natural-language prompts over the model's own tokenizer instead.

Two further honesty notes:
* The induction task also scores the FIRST copy, where the answer is
  unpredictable. That is a chance-level control: if second-copy accuracy is not
  far above it, nothing was learned in-context and the task is uninformative.
* Passkey decoding uses an explicit greedy loop with use_cache=False (full
  re-forward per token). QGFD is defined over the materialised probability
  matrix, so incremental-decode caching is the one place train/eval mechanism
  could silently diverge; re-forwarding keeps it identical to training.

Usage
-----
    python scripts/eval_synthetic.py --quick --model_id JackFram/llama-160m --device cpu
    python scripts/eval_synthetic.py --model_id HuggingFaceTB/SmolLM2-135M --seeds 0,1,2
    python scripts/eval_synthetic.py --model_id HuggingFaceTB/SmolLM2-135M --post_lora
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.review_experiments import (  # noqa: E402
    _signif,
    _stat,
    fmt_stat,
    load_wikitext,
    resolve_device,
)
from scripts.finetune_qgfd import (  # noqa: E402
    FinetuneConfig,
    build_arm_model,
    train_arm,
)

# __CHUNK2__

# Candidate induction vocabulary. Filtered at runtime down to the words that are
# a SINGLE token under the model's own tokenizer (see single_token_vocab), which
# keeps "one word = one hop" true for any BPE vocabulary.
_CANDIDATE_WORDS = """
apple river stone tiger cloud bridge candle forest marble silver garden window
mountain harbor lantern copper velvet meadow anchor pepper thunder ribbon saddle
tunnel violin walnut orchid pyramid compass diamond emerald falcon glacier hammer
island jungle kettle ladder magnet needle island oyster palace quilt rocket
shadow temple umbrella valley whistle yellow zebra basket cactus dolphin engine
feather guitar helmet igloo jacket kitten lemon monkey noodle orange pencil
quiver rabbit summer turtle union violet wagon xylophone yogurt zipper autumn
bottle circus desert eagle flower grape hotel indigo juice koala lemon mirror
nectar ocean piano queen rocket sunset trumpet velvet winter cabin dragon
fabric ginger honey ivory jelly kernel lilac mango nickel olive parrot quartz
""".split()


@dataclass
class SyntheticConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-135M"
    dtype: str = "bfloat16"
    device: str = "auto"
    backend: str = "operator"

    # --- QGFD (arm "qgfd" only) --------------------------------------------
    diffusion_steps: int = 1
    target_alpha: float = 0.05
    detach_P: bool = True
    mode: str = "full"
    max_full_seq_len: int = 512
    full_fallback_mode: str = "conv"

    # --- induction ----------------------------------------------------------
    induction_num_examples: int = 64
    induction_seq_len: int = 48         # words per copy; 2 copies <= ~100 tokens
    induction_min_ctx: int = 1          # skip the first N second-copy positions
    # Fraction of the SECOND copy replaced by unrelated words. Corrupted positions
    # are excluded from scoring; the rest must route through a garbled context.
    induction_noise_rates: Tuple[float, ...] = (0.0, 0.2, 0.4)

    # --- passkey ------------------------------------------------------------
    passkey_num_examples: int = 24      # per depth
    passkey_context_tokens: int = 384   # <= max_full_seq_len keeps mode="full"
    passkey_depths: Tuple[float, ...] = (0.1, 0.5, 0.9)
    passkey_digits: int = 5
    passkey_max_new_tokens: int = 8

    seed: int = 0
    out_dir: str = "./qgfd_synthetic_results"

    # --- post-LoRA probe (only used with post_lora=True) -------------------
    ft_max_steps: int = 300
    ft_batch_size: int = 2
    ft_grad_accum: int = 8
    ft_block_size: int = 256
    ft_learning_rate: float = 2e-4
    ft_warmup_steps: int = 100
    ft_train_num_texts: int = 1500
    ft_lora_r: int = 16
    ft_gradient_checkpointing: bool = True
    ft_eval_ppl_num_texts: int = 60
    ft_eval_robustness_num_texts: int = 40
    ft_eval_max_length: int = 512

    def finetune_config(self) -> FinetuneConfig:
        """
        A FinetuneConfig carrying the same model/operator settings.

        Reused two ways: `build_arm_model` for the zero-shot arms (so the operator
        is installed and verified by exactly the code track 2 trains through), and
        `train_arm` for the post-LoRA arms.
        """
        return FinetuneConfig(
            model_id=self.model_id, dtype=self.dtype, device=self.device,
            backend=self.backend, diffusion_steps=self.diffusion_steps,
            target_alpha=self.target_alpha, detach_P=self.detach_P, mode=self.mode,
            max_full_seq_len=self.max_full_seq_len,
            full_fallback_mode=self.full_fallback_mode,
            warmup_steps=self.ft_warmup_steps, max_steps=self.ft_max_steps,
            batch_size=self.ft_batch_size, grad_accum=self.ft_grad_accum,
            block_size=self.ft_block_size, learning_rate=self.ft_learning_rate,
            train_num_texts=self.ft_train_num_texts, lora_r=self.ft_lora_r,
            gradient_checkpointing=self.ft_gradient_checkpointing,
            eval_ppl_num_texts=self.ft_eval_ppl_num_texts,
            eval_robustness_num_texts=self.ft_eval_robustness_num_texts,
            eval_max_length=self.ft_eval_max_length,
            seed=self.seed, out_dir=self.out_dir,
        )

# __CHUNK3__


# --------------------------------------------------------------------------- #
# Task 1: induction heads
# --------------------------------------------------------------------------- #
def single_token_vocab(tok, base: str = "Sequence:") -> Tuple[List[int], List[int]]:
    """
    Find candidate words that cost exactly ONE token when appended to `base`.

    Deliberately empirical rather than assuming `" word"` is one token: SentencePiece
    (Llama) and byte-level BPE (GPT-2/Qwen) disagree about leading spaces and dummy
    prefixes. Appending and re-tokenizing measures the real thing, and lets the rest
    of the task be built directly in id space — so the id sequence we feed the model
    is genuinely the tokenization of the text it represents.
    """
    base_ids = tok(base, add_special_tokens=False)["input_ids"]
    n = len(base_ids)
    ids, seen = [], set()
    for w in _CANDIDATE_WORDS:
        if w in seen:
            continue
        seen.add(w)
        full = tok(f"{base} {w}", add_special_tokens=False)["input_ids"]
        if len(full) == n + 1 and full[:n] == base_ids and full[-1] not in ids:
            ids.append(full[-1])
    return base_ids, ids


def make_induction_examples(tok, cfg: SyntheticConfig, rng: random.Random,
                            noise_rate: float = 0.0):
    """
    Build `induction_num_examples` sequences of the form  base + S + S'.

    S' is S with `noise_rate` of its tokens replaced by words absent from S. The
    corrupted positions are excluded from scoring via the returned validity mask:
    a corrupted position has no correct answer (its query word never appeared), so
    scoring it would measure nothing. Uncorrupted positions remain well-posed but
    now have to route through a partly-garbled context — which is precisely the
    regime the paper's robustness claim is about.

    Every example has the same length, so the scored positions are shared and the
    whole set evaluates in batched forward passes.
    """
    base_ids, vocab = single_token_vocab(tok)
    k = cfg.induction_seq_len
    if len(vocab) < k + 4:
        raise RuntimeError(
            f"Only {len(vocab)} single-token words available for this tokenizer, "
            f"need induction_seq_len+4={k + 4}. Lower --induction_seq_len.")

    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    off = len(bos) + len(base_ids)
    scored = list(range(cfg.induction_min_ctx, k - 1))     # i -> target S[i+1]
    if not scored:
        raise RuntimeError("induction_seq_len too small for induction_min_ctx.")

    rows, targets, valid = [], [], []
    for _ in range(cfg.induction_num_examples):
        S = rng.sample(vocab, k)
        second, corrupted = list(S), set()
        if noise_rate > 0:
            pool = [v for v in vocab if v not in set(S)]
            corrupted = set(rng.sample(range(k), int(round(noise_rate * k))))
            for j in corrupted:
                second[j] = rng.choice(pool)
        rows.append(bos + base_ids + S + second)
        targets.append([S[i + 1] for i in scored])
        valid.append([i not in corrupted for i in scored])

    return (torch.tensor(rows, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(valid, dtype=torch.bool),
            torch.tensor([off + k + i for i in scored], dtype=torch.long),
            torch.tensor([off + i for i in scored], dtype=torch.long))


@torch.no_grad()
def _induction_pass(model, tok, device: str, cfg: SyntheticConfig,
                    noise_rate: float, micro_batch: int) -> Dict:
    rng = random.Random(cfg.seed)
    ids, tgt, valid, pos, ctrl = make_induction_examples(tok, cfg, rng, noise_rate)
    if not bool(valid.any()):
        raise RuntimeError(f"noise_rate={noise_rate} corrupted every scored position.")
    pos, ctrl = pos.to(device), ctrl.to(device)
    hits = ctrl_hits = total = 0
    for s in range(0, ids.shape[0], micro_batch):
        logits = model(input_ids=ids[s:s + micro_batch].to(device)).logits.float()
        t = tgt[s:s + micro_batch].to(device)
        v = valid[s:s + micro_batch].to(device)
        hits += ((logits[:, pos, :].argmax(-1) == t) & v).sum().item()
        ctrl_hits += ((logits[:, ctrl, :].argmax(-1) == t) & v).sum().item()
        total += int(v.sum().item())
    return {"acc": hits / total, "control_acc": ctrl_hits / total,
            "n_predictions": total, "n_examples": int(ids.shape[0]),
            "seq_len_tokens": int(ids.shape[1])}


def eval_induction(model, tok, device: str, cfg: SyntheticConfig,
                   micro_batch: int = 8) -> Dict:
    """
    Induction accuracy at each context-corruption rate.

    Top-level `acc`/`control_acc` are the CLEAN (0% corruption) numbers; `by_noise`
    holds the sweep. `control_acc` scores the first copy, where S[i+1] is genuinely
    unpredictable — the chance-level floor for the task.
    """
    rates = tuple(dict.fromkeys((0.0,) + tuple(cfg.induction_noise_rates)))
    by_noise = {f"{r:.2f}": _induction_pass(model, tok, device, cfg, r, micro_batch)
                for r in rates}
    clean = by_noise["0.00"]
    return {"acc": clean["acc"], "control_acc": clean["control_acc"],
            "n_predictions": clean["n_predictions"],
            "n_examples": clean["n_examples"],
            "seq_len_tokens": clean["seq_len_tokens"],
            "noise_rates": [f"{r:.2f}" for r in rates],
            "by_noise": by_noise}

# __CHUNK4__


# --------------------------------------------------------------------------- #
# Task 2: passkey retrieval
# --------------------------------------------------------------------------- #
_FILLER = ("The grass is green. The sky is blue. The sun is yellow. "
           "Here we go. There and back again. ")
_HEAD = ("There is an important piece of information hidden inside a lot of "
         "irrelevant text. Find it and memorize it. I will quiz you about it.\n")
_TAIL = "\nWhat is the pass key? The pass key is"


def make_passkey_prompt(tok, cfg: SyntheticConfig, rng: random.Random,
                        depth: float) -> Tuple[str, str]:
    """A key buried at `depth` (0 = start, 1 = end) of the filler block."""
    key = str(rng.randint(1, 9)) + "".join(
        str(rng.randint(0, 9)) for _ in range(cfg.passkey_digits - 1))
    n_filler_tokens = len(tok(_FILLER, add_special_tokens=False)["input_ids"])
    n_rep = max(2, cfg.passkey_context_tokens // max(1, n_filler_tokens))
    before = min(n_rep - 1, max(1, int(round(n_rep * depth))))
    info = f"The pass key is {key}. Remember it. {key} is the pass key.\n"
    prompt = (_HEAD + _FILLER * before + info + _FILLER * (n_rep - before) + _TAIL)
    return prompt, key


@torch.no_grad()
def greedy_generate(model, tok, prompt: str, device: str, max_new_tokens: int) -> str:
    """
    Explicit greedy loop with no KV cache: every step re-forwards the full prefix.

    Slower than .generate(), but QGFD is defined over the materialised attention
    probability matrix, and incremental decode is the one place its mechanism could
    silently differ from training. A few extra forwards is a cheap price for
    knowing the probe measures the same operator the model was trained with.
    """
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    produced: List[int] = []
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits[:, -1, :].float()
        nxt = logits.argmax(-1, keepdim=True)
        tid = int(nxt.item())
        if tok.eos_token_id is not None and tid == tok.eos_token_id:
            break
        produced.append(tid)
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(produced, skip_special_tokens=True)


@torch.no_grad()
def eval_passkey(model, tok, device: str, cfg: SyntheticConfig) -> Dict:
    """
    Strict metric (`acc`): the continuation must START with the key. `contains`
    is the lenient variant, reported alongside because a model that emits the key
    after a preamble has still retrieved it — but only the strict number is used
    for the paper's headline table.
    """
    rng = random.Random(cfg.seed + 1000)
    by_depth, hits, total, ctx_tokens = {}, 0, 0, None
    for depth in cfg.passkey_depths:
        h = c = 0
        for _ in range(cfg.passkey_num_examples):
            prompt, key = make_passkey_prompt(tok, cfg, rng, depth)
            if ctx_tokens is None:
                ctx_tokens = len(tok(prompt)["input_ids"])
            cont = greedy_generate(model, tok, prompt, device,
                                   cfg.passkey_max_new_tokens)
            h += int(cont.strip().startswith(key))
            c += int(key in cont)
        n = cfg.passkey_num_examples
        by_depth[f"{depth:.2f}"] = {"acc": h / n, "contains": c / n, "n": n}
        hits += h
        total += n
    return {"acc": hits / total, "by_depth": by_depth, "n": total,
            "context_tokens": ctx_tokens}

# __CHUNK5__


# --------------------------------------------------------------------------- #
# Per-seed orchestration
# --------------------------------------------------------------------------- #
def eval_arm(arm: str, cfg: SyntheticConfig, post_lora: bool = False,
             train_texts: Optional[List[str]] = None,
             eval_texts: Optional[List[str]] = None) -> Dict:
    """Install (or train) one arm, then run both synthetic probes on it."""
    import gc

    fcfg = cfg.finetune_config()
    print(f"\n--- arm: {arm}{' (post-LoRA)' if post_lora else ' (zero-shot)'} ---",
          flush=True)
    ft = None
    if post_lora:
        ft, tok, model, device = train_arm(arm, fcfg, train_texts, eval_texts,
                                          return_model=True)
    else:
        tok, model, device = build_arm_model(arm, fcfg)
    model.eval()

    print(f"  [{arm}] induction ...", flush=True)
    induction = eval_induction(model, tok, device, cfg)
    print(f"  [{arm}] induction acc = {induction['acc']:.4f} "
          f"(control {induction['control_acc']:.4f})")
    for r, d in induction["by_noise"].items():
        if r != "0.00":
            print(f"    ctx corruption {r}: acc = {d['acc']:.4f} "
                  f"(n={d['n_predictions']})")
    print(f"  [{arm}] passkey ...", flush=True)
    passkey = eval_passkey(model, tok, device, cfg)
    print(f"  [{arm}] passkey acc  = {passkey['acc']:.4f} "
          f"@ {passkey['context_tokens']} ctx tokens")

    provenance = _operator_provenance(model, fcfg, arm)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out = {"kind": arm, "post_lora": post_lora,
           "induction": induction, "passkey": passkey,
           "operator": provenance}
    if ft is not None:
        out["finetune"] = ft          # clean ppl / robustness / alpha of this arm
    return out


def _operator_provenance(model, fcfg: FinetuneConfig, arm: str) -> Dict:
    """
    Record the operator's live settings alongside the scores.

    Exact-match accuracy is a coarse metric: with alpha=0.05 the two arms disagree
    on only ~1% of argmax predictions, so identical accuracies are an expected
    outcome at small n — not evidence QGFD was off. This block is what lets a
    reader tell those two cases apart.
    """
    if arm != "qgfd":
        return {"qgfd_active": False}
    if fcfg.backend == "operator":
        from torchdire import collect_qgfd_operators as collect
    else:
        from torchdire import collect_qgfd_kernels as collect
    mods = collect(model)
    if not mods:
        return {"qgfd_active": False, "warning": "no QGFD modules found"}
    a = mods[0].get_alpha()
    return {"qgfd_active": True, "n_modules": len(mods),
            "alpha_eval_mode": float(a.mean().item()) if isinstance(a, torch.Tensor)
            else float(a),
            "diffusion_steps": mods[0].diffusion_steps,
            "mode": getattr(mods[0], "mode", None),
            "is_causal": getattr(mods[0], "is_causal", None)}


def run_seed(cfg: SyntheticConfig, arms: Sequence[str] = ("softmax", "qgfd"),
             post_lora: bool = False) -> Dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_texts = eval_texts = None
    if post_lora:
        fcfg = cfg.finetune_config()
        print(f"Loading WikiText-2 for the LoRA arms "
              f"({fcfg.train_num_texts} train paragraphs) ...", flush=True)
        train_texts = load_wikitext(fcfg.train_num_texts, split="train")
        eval_texts = load_wikitext(max(fcfg.eval_ppl_num_texts,
                                      fcfg.eval_robustness_num_texts), split="test")

    result = {
        "config": asdict(cfg),
        "post_lora": post_lora,
        "device": resolve_device(cfg.device),
        "arms": {a: eval_arm(a, cfg, post_lora, train_texts, eval_texts)
                 for a in arms},
    }
    path = os.path.join(cfg.out_dir, "synthetic_results.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved synthetic results -> {path}")
    _print_seed_summary(result)
    return result


def _print_seed_summary(result: Dict) -> None:
    arms = list(result["arms"])
    print(f"\n{'=' * 74}\nSYNTHETIC MULTI-HOP — seed {result['config']['seed']}"
          f"{'  [post-LoRA]' if result['post_lora'] else '  [zero-shot]'}\n{'=' * 74}")
    print(f"{'metric':<34}" + "".join(f"{a:>18}" for a in arms))
    rows = [("Induction acc (clean)", lambda r: r["induction"]["acc"]),
            ("  (control: first copy)", lambda r: r["induction"]["control_acc"])]
    for r_ in result["arms"][arms[0]]["induction"]["by_noise"]:
        if r_ != "0.00":
            rows.append((f"  induction @ ctx noise {r_}",
                         lambda r, k=r_: r["induction"]["by_noise"][k]["acc"]))
    rows.append(("Passkey acc (strict)", lambda r: r["passkey"]["acc"]))
    for d in result["arms"][arms[0]]["passkey"]["by_depth"]:
        rows.append((f"  passkey @ depth {d}",
                     lambda r, d=d: r["passkey"]["by_depth"][d]["acc"]))
    for label, fn in rows:
        print(f"{label:<34}" + "".join(f"{fn(result['arms'][a]):>18.4f}"
                                       for a in arms))

# __CHUNK6__


# --------------------------------------------------------------------------- #
# Multi-seed aggregation
# --------------------------------------------------------------------------- #
def aggregate_synthetic(runs: Sequence[Dict], seeds: Sequence[int]) -> Dict:
    """
    mean +/- std over seeds, plus PAIRED (qgfd - softmax) per-seed differences.

    The paired statistic is the one to trust: within a seed both arms see the
    identical prompts, so the difference cancels the (large) between-seed variance
    of which random word sequences and keys were drawn.
    """
    if not runs:
        raise ValueError("aggregate_synthetic() needs at least one run")
    arms = list(runs[0]["arms"])
    depths = list(runs[0]["arms"][arms[0]]["passkey"]["by_depth"])
    inoise = list(runs[0]["arms"][arms[0]]["induction"]["by_noise"])
    c0 = runs[0]["config"]

    def col(arm, fn):
        return [fn(r["arms"][arm]) for r in runs]

    agg = {
        "meta": {
            "track": "synthetic",
            "seeds": list(seeds),
            "n_seeds": len(runs),
            "post_lora": runs[0]["post_lora"],
            "model_id": c0["model_id"],
            "backend": c0["backend"],
            "device": runs[0]["device"],
            "diffusion_steps": c0["diffusion_steps"],
            "target_alpha": c0["target_alpha"],
            "induction_seq_len": c0["induction_seq_len"],
            "induction_predictions_per_seed":
                runs[0]["arms"][arms[0]]["induction"]["n_predictions"],
            "passkey_context_tokens":
                runs[0]["arms"][arms[0]]["passkey"]["context_tokens"],
            "passkey_n_per_seed": runs[0]["arms"][arms[0]]["passkey"]["n"],
            "ci_method": "t-based 95% CI (small n); std is the sample std",
            "operator": {a: runs[0]["arms"][a]["operator"] for a in arms},
            "coarse_metric_note": (
                "exact-match accuracy is coarse: at alpha=0.05 the arms disagree on "
                "~1% of argmax predictions, so identical scores at small n are "
                "expected and do NOT mean QGFD was inactive — check meta.operator."),
            "control_note": ("induction_control_acc scores the FIRST copy, where the "
                             "target is unpredictable. If acc is not well above it, "
                             "no in-context induction happened."),
        },
        "arms": {}, "paired": {},
    }
    for a in arms:
        agg["arms"][a] = {
            "induction_acc": _stat(col(a, lambda r: r["induction"]["acc"])),
            "induction_control_acc":
                _stat(col(a, lambda r: r["induction"]["control_acc"])),
            "induction_by_noise": {
                r: _stat(col(a, lambda x, r=r: x["induction"]["by_noise"][r]["acc"]))
                for r in inoise
            },
            "passkey_acc": _stat(col(a, lambda r: r["passkey"]["acc"])),
            "passkey_by_depth": {
                d: _stat(col(a, lambda r, d=d: r["passkey"]["by_depth"][d]["acc"]))
                for d in depths
            },
        }
    if {"softmax", "qgfd"} <= set(arms):
        def gap(fn):
            return _stat([fn(r["arms"]["qgfd"]) - fn(r["arms"]["softmax"])
                          for r in runs])
        agg["paired"] = {
            "induction_gap": gap(lambda r: r["induction"]["acc"]),
            "induction_gap_by_noise": {
                r: gap(lambda x, r=r: x["induction"]["by_noise"][r]["acc"])
                for r in inoise
            },
            "passkey_gap": gap(lambda r: r["passkey"]["acc"]),
            "passkey_gap_by_depth": {
                d: gap(lambda r, d=d: r["passkey"]["by_depth"][d]["acc"])
                for d in depths
            },
            "note": "qgfd - softmax, computed within each seed on identical prompts",
        }
    return agg

# __CHUNK7__


def _print_synthetic_aggregate(agg: Dict) -> None:
    m, arms = agg["meta"], list(agg["arms"])
    print(f"\n{'=' * 78}")
    print(f"SYNTHETIC MULTI-HOP SUMMARY — {m['model_id']}  "
          f"(n={m['n_seeds']} seeds: {m['seeds']}"
          f"{', post-LoRA' if m['post_lora'] else ', zero-shot'})")
    print(f"{'=' * 78}")
    print(f"{'metric':<34}" + "".join(f"{a:>20}" for a in arms))
    rows = [("Induction acc (clean)", "induction_acc"),
            ("  (control: first copy)", "induction_control_acc")]
    for label, key in rows:
        print(f"{label:<34}" + "".join(
            f"{fmt_stat(agg['arms'][a][key], prec=4):>20}" for a in arms))
    for r in agg["arms"][arms[0]]["induction_by_noise"]:
        if r == "0.00":
            continue
        print(f"{'  induction @ ctx noise ' + r:<34}" + "".join(
            f"{fmt_stat(agg['arms'][a]['induction_by_noise'][r], prec=4):>20}"
            for a in arms))
    print(f"{'Passkey acc (strict)':<34}" + "".join(
        f"{fmt_stat(agg['arms'][a]['passkey_acc'], prec=4):>20}" for a in arms))
    for d in agg["arms"][arms[0]]["passkey_by_depth"]:
        print(f"{'  passkey @ depth ' + d:<34}" + "".join(
            f"{fmt_stat(agg['arms'][a]['passkey_by_depth'][d], prec=4):>20}"
            for a in arms))
    if agg["paired"]:
        print("\nPaired (qgfd - softmax), positive = QGFD better:")
        pairs = [("induction", agg["paired"]["induction_gap"])]
        pairs += [(f"induction n{r}", s) for r, s
                  in agg["paired"]["induction_gap_by_noise"].items() if r != "0.00"]
        pairs.append(("passkey", agg["paired"]["passkey_gap"]))
        for label, s in pairs:
            print(f"  {label:<16} {fmt_stat(s, prec=4)}  "
                  f"[95% CI +/-{s['ci95']:.4f}] {_signif(s)}")
    print(f"\n{m['control_note']}")
    print(f"NOTE: {m['coarse_metric_note']}")
    op = m["operator"].get("qgfd", {})
    print(f"qgfd operator: {op}")
    print(f"CIs: {m['ci_method']}")


def run_all_seeds(cfg: SyntheticConfig, seeds: Sequence[int] = (0, 1, 2),
                  arms: Sequence[str] = ("softmax", "qgfd"),
                  post_lora: bool = False) -> Dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    runs, seeds = [], list(seeds)
    for i, s in enumerate(seeds):
        print(f"\n{'#' * 74}\n# SEED {s}  ({i + 1}/{len(seeds)})\n{'#' * 74}",
              flush=True)
        runs.append(run_seed(
            replace(cfg, seed=s, out_dir=os.path.join(cfg.out_dir, f"seed{s}")),
            arms=arms, post_lora=post_lora))
    agg = aggregate_synthetic(runs, seeds)
    path = os.path.join(cfg.out_dir, "synthetic_aggregated.json")
    with open(path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved aggregated synthetic results -> {path}")
    _print_synthetic_aggregate(agg)
    return agg


def apply_quick(cfg: SyntheticConfig) -> SyntheticConfig:
    """Tiny CPU smoke settings — enough to exercise every code path."""
    return replace(
        cfg,
        induction_num_examples=4, induction_seq_len=16,
        induction_noise_rates=(0.0, 0.25),
        passkey_num_examples=2, passkey_context_tokens=96,
        passkey_depths=(0.1, 0.9), passkey_max_new_tokens=6,
        ft_max_steps=4, ft_warmup_steps=2, ft_block_size=64, ft_batch_size=1,
        ft_grad_accum=1, ft_train_num_texts=40, ft_lora_r=4,
        ft_gradient_checkpointing=False,
        ft_eval_ppl_num_texts=4, ft_eval_robustness_num_texts=4,
        ft_eval_max_length=128,
    )

# __CHUNK8__


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="QGFD synthetic multi-hop probes")
    p.add_argument("--model_id", default=SyntheticConfig.model_id)
    p.add_argument("--backend", default="operator", choices=["operator", "kernel"])
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--arms", default="softmax,qgfd",
                   help="Comma-separated subset of softmax,qgfd")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seeds, e.g. '0,1,2'. Omit for a single seed.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--post_lora", action="store_true",
                   help="LoRA fine-tune each arm first, then probe (much slower)")
    p.add_argument("--diffusion_steps", type=int, default=SyntheticConfig.diffusion_steps)
    p.add_argument("--target_alpha", type=float, default=SyntheticConfig.target_alpha)
    p.add_argument("--induction_num_examples", type=int,
                   default=SyntheticConfig.induction_num_examples)
    p.add_argument("--induction_seq_len", type=int,
                   default=SyntheticConfig.induction_seq_len)
    p.add_argument("--passkey_num_examples", type=int,
                   default=SyntheticConfig.passkey_num_examples)
    p.add_argument("--passkey_context_tokens", type=int,
                   default=SyntheticConfig.passkey_context_tokens)
    p.add_argument("--out_dir", default=SyntheticConfig.out_dir)
    p.add_argument("--quick", action="store_true", help="Tiny CPU smoke run")
    a = p.parse_args(argv)

    cfg = SyntheticConfig(
        model_id=a.model_id, backend=a.backend, dtype=a.dtype, device=a.device,
        seed=a.seed, diffusion_steps=a.diffusion_steps, target_alpha=a.target_alpha,
        induction_num_examples=a.induction_num_examples,
        induction_seq_len=a.induction_seq_len,
        passkey_num_examples=a.passkey_num_examples,
        passkey_context_tokens=a.passkey_context_tokens,
        out_dir=a.out_dir,
    )
    if a.quick:
        cfg = apply_quick(cfg)
    arms = tuple(x.strip() for x in a.arms.split(",") if x.strip())

    if a.seeds:
        run_all_seeds(cfg, seeds=[int(s) for s in a.seeds.split(",")],
                      arms=arms, post_lora=a.post_lora)
    else:
        run_seed(cfg, arms=arms, post_lora=a.post_lora)


if __name__ == "__main__":
    main()
