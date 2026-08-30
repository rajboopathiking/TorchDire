"""
Gated, trust-aware QGFD — the redesign experiments.
====================================================
The fixed-alpha operator does not work. Across 3 models x 3 seeds x 4 noise
rates, QGFD's ABSOLUTE perplexity under noise was lower in 0 of 36 measurements
(see E9 / `scripts/build_report.py::_decompose`). The reported "robustness gap"
was a difference of *relative* degradations, inflated by QGFD's own higher clean
perplexity.

The diagnosis points at the fix. At alpha = 0 QGFD *is* softmax, so a fixed
alpha > 0 is a constant perturbation of a converged optimum: the loss change is
`dL ~ 1/2 d'Hd = O(alpha^2) >= 0` in every direction. A perturbation applied
everywhere, always, cannot pay for itself. The repair is to make the mixing
coefficient a LEARNED, ZERO-INITIALISED, PER-HEAD, PER-POSITION gate:

    p' = (1 - g_ht) * p0 + g_ht * (p0 P),    g_ht = g_max * tanh(z_ht)
    z_ht = a_h * Hhat(p0)_t + b_h * margin(p0)_t + r_h,   a = b = r = 0 at init

so training starts at exact softmax and every head must *earn* any diffusion it
uses. Signed (tanh, not sigmoid) for two reasons: g == 0 is then reachable
exactly WITH a live gradient (tanh'(0) = 1), and the evidence so far is
consistent with heads wanting to SHARPEN, which needs g < 0.

Stage layout (see docs/gated-qgfd-experiments.md for the protocol and budget):

  Stage 0  ~40 min, NO TRAINING. Does the headroom exist at all?
           0a  dCE_noisy/dalpha_h for every head from ONE backward pass.
           0b  per-layer finite-difference alpha oracle.
           0c  corruption-family screen (token-preserving vs tokenisation-breaking).
           Hard stop rule: no head with a negative noisy-CE gradient AND no
           corruption family with an absolute win => publish the negative result
           and do not build the gate.

  Stage 1  gate only, everything else frozen. Smallest model, 3 seeds.
  Stage 2  + trust-aware sparse P (top-k, self-loop, reliability) + Q/K LoRA.
  Stage 3  recipe transfer to the two larger models, 1 seed, no re-tuning.

Metric discipline, learned the hard way
---------------------------------------
* The primary number is PAIRED PER-WINDOW CROSS-ENTROPY IN NATS on byte-identical
  corrupted text. Never a relative delta of perplexities: PPL = exp(CE), so any
  ratio statistic re-imports exactly the denominator sensitivity that produced
  the phantom +12.6 pp gap.
* Pairing is at the WINDOW level (hundreds of windows), not the seed level
  (n = 3). Seeds measure training variance; windows measure the effect.
* Success = constrained Pareto: clean CE regression <= eps AND noisy CE strictly
  lower, with the paired CI excluding zero.
* The uniform-P / shuffled-P controls from E4 stay in the design. A gate that
  still wins with a structure-free P has learned "smooth when uncertain", which
  is an adaptive temperature schedule and folds into W_Q.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdire import QGFDOperator, SoftmaxOperator, wrap_model_with_qgfd_operator
from torchdire.nn.attention_adapters import AttentionOperatorAdapter

from scripts.review_experiments import (
    ExperimentConfig,
    load_model_and_tokenizer,
    load_wikitext,
    perturb_text,
    resolve_device,
    verify_patch,
    free,
)
from scripts.mechanism_experiments import (
    CORRUPTIONS,
    StructuredPOperator,
    precompute_corruptions,
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class GatedConfig:
    """Everything the four stages need. Defaults are the T4-sized settings."""

    model_id: str = "HuggingFaceTB/SmolLM2-135M"
    # fp16, NOT bf16. Turing (T4, sm_75) has fp16 tensor cores and no bf16 ones,
    # so bf16 is emulated: "correct but slower". The operator keeps p0, P and the
    # loss in fp32 regardless, which is where precision actually matters.
    dtype: str = "float16"
    device: str = "auto"

    # Sequence length. 128 for method development: eager attention is O(n^2)
    # materialised and the diffusion product is O(n^3), so this is the single
    # biggest lever after dtype.
    seq_len: int = 128

    # ---- gate ----
    g_max: float = 0.05
    signed: bool = True            # tanh gate: exactly 0 at init, gradient alive
    use_entropy: bool = True
    use_margin: bool = True

    # ---- trust graph (Stage 2 only) ----
    top_k: int = 8
    self_loop: float = 1.0
    reliability: bool = True

    # ---- training ----
    gate_lr: float = 3e-2          # ~1k parameters; a normal LM lr is far too small
    lora_lr: float = 1e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_targets: Tuple[str, ...] = ("q_proj", "k_proj")
    max_steps: int = 300
    grad_accum: int = 4
    lambda_noisy_max: float = 1.0
    lambda_ramp_frac: float = 0.3  # fraction of max_steps spent ramping lambda_n
    gamma_l1: float = 1e-2         # gate L1; the "stay at softmax unless you earn it" term
    grad_checkpointing: bool = False

    # ---- corruption curriculum ----
    curriculum: Tuple[float, float] = (0.02, 0.08)
    train_corruption: str = "word_drop"
    eval_corruption: str = "word_drop"
    eval_rate: float = 0.10

    # ---- data / eval ----
    n_train_texts: int = 400
    n_eval_texts: int = 60
    eval_max_windows: int = 200
    epsilon_clean_nats: float = 0.003   # ~0.3% relative PPL at CE ~ 3 nats

    # ---- Stage 0 ----
    probe_windows: int = 16
    probe_alpha: float = 0.05
    probe_families: Tuple[str, ...] = ("word_drop", "word_swap", "token_replace",
                                       "repeat_token", "char", "ocr")

    seeds: Tuple[int, ...] = (0, 1, 2)
    out_dir: str = "./qgfd_gated_results"
    quick: bool = False

    def eval_config(self) -> ExperimentConfig:
        """An ExperimentConfig for the loaders/verifiers we reuse verbatim."""
        return ExperimentConfig(
            model_id=self.model_id, dtype=self.dtype, device=self.device,
            diffusion_steps=1, target_alpha=self.g_max,
            ppl_max_length=self.seq_len, ppl_stride=self.seq_len,
            robustness_num_texts=self.n_eval_texts,
            attn_seq_len=self.seq_len, out_dir=self.out_dir,
        )


# --------------------------------------------------------------------------- #
# Corruption families
# --------------------------------------------------------------------------- #
# Two groups, and the distinction is the whole point of the screen:
#
#   TOKENISATION-PRESERVING — word_drop, word_swap, token_replace, repeat_token.
#     Every surviving word keeps its exact token ids. These are the families that
#     actually instantiate the QGFD hypothesis ("the key at position j is wrong,
#     route some mass to keys that look like the right one").
#
#   TOKENISATION-BREAKING — char, ocr.
#     Character edits re-segment words into subword fragments, so a result here is
#     confounded with fragmentation repair. `char` is the family the original
#     headline used, which is one reason that headline was hard to interpret.
_OCR_CONFUSIONS = {"l": "1", "1": "l", "O": "0", "0": "O", "S": "5", "5": "S",
                   "B": "8", "8": "B", "I": "l", "o": "c", "e": "c", "m": "rn",
                   "n": "h", "u": "v", "v": "y", "g": "9", "q": "g"}


def perturb_token_replace(text: str, rate: float, rng: random.Random) -> str:
    """Replace whole words with other words drawn from the same text.

    Information is destroyed and the local context becomes misleading, but every
    token id in the output already appeared in the input's vocabulary usage, so
    the tokeniser is not stressed at all.
    """
    words = text.split()
    if len(words) < 4 or rate <= 0:
        return text
    pool = list(words)
    out = [rng.choice(pool) if rng.random() < rate else w for w in words]
    return " ".join(out)


def perturb_repeat_token(text: str, rate: float, rng: random.Random) -> str:
    """Duplicate words. Tests whether attention can discount redundant keys."""
    words = text.split()
    if not words or rate <= 0:
        return text
    out = []
    for w in words:
        out.append(w)
        if rng.random() < rate:
            out.append(w)
    return " ".join(out)


def perturb_ocr(text: str, rate: float, rng: random.Random) -> str:
    """OCR-style confusable substitution. Character-level, so it re-tokenises."""
    if rate <= 0:
        return text
    return "".join(_OCR_CONFUSIONS.get(ch, ch) if (ch in _OCR_CONFUSIONS
                                                  and rng.random() < rate) else ch
                   for ch in text)


TOKEN_PRESERVING = ("word_drop", "word_swap", "token_replace", "repeat_token")
TOKENISATION_BREAKING = ("char", "ocr")


def register_corruptions() -> Dict[str, object]:
    """Add this module's families to the shared `CORRUPTIONS` registry."""
    CORRUPTIONS.setdefault("token_replace", perturb_token_replace)
    CORRUPTIONS.setdefault("repeat_token", perturb_repeat_token)
    CORRUPTIONS.setdefault("ocr", perturb_ocr)
    return CORRUPTIONS


register_corruptions()


# --------------------------------------------------------------------------- #
# The primary metric: paired per-window cross-entropy in nats
# --------------------------------------------------------------------------- #
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 29: 2.045}


def _t95(dof: int) -> float:
    if dof >= 30:
        return 1.96
    if dof in _T95:
        return _T95[dof]
    return next(v for k, v in sorted(_T95.items()) if k >= dof)


@torch.no_grad()
def window_ce(model, tok, texts: Sequence[str], device: str, seq_len: int,
              max_windows: Optional[int] = None) -> List[Tuple[float, int]]:
    """Per-window mean cross-entropy in nats, plus that window's token count.

    Returned as a LIST so two arms can be paired window-by-window. Both arms see
    the identical token stream (same tokeniser, byte-identical text), so window i
    in one arm and window i in the other are the same prediction problem and the
    difference has no corpus-sampling variance in it at all.
    """
    enc = tok("\n\n".join(texts), return_tensors="pt").input_ids.to(device)
    n = enc.size(1)
    out: List[Tuple[float, int]] = []
    prev_end = 0
    for begin in range(0, n, seq_len):
        end = min(begin + seq_len, n)
        trg_len = end - prev_end
        ids = enc[:, begin:end]
        targets = ids.clone()
        targets[:, :-trg_len] = -100
        n_tok = int((targets != -100).sum().item())
        if n_tok < 8:                       # a 1-7 token tail is pure noise
            break
        loss = model(ids, labels=targets).loss.float().item()
        out.append((loss, n_tok))
        prev_end = end
        if end == n or (max_windows is not None and len(out) >= max_windows):
            break
    return out


def token_weighted_ce(windows: Sequence[Tuple[float, int]]) -> float:
    tot = sum(n for _, n in windows)
    return sum(ce * n for ce, n in windows) / max(1, tot)


def paired_delta(a: Sequence[Tuple[float, int]],
                 b: Sequence[Tuple[float, int]]) -> Dict:
    """Paired statistics on `a - b` in nats/token. Negative means `a` is better.

    `n` here is the number of WINDOWS, typically 100-300, so the CI is roughly an
    order of magnitude tighter than a 3-seed aggregate over the same compute.
    """
    m = min(len(a), len(b))
    if m < 2:
        return {"n": m, "mean": None, "ci95": None, "sig": False, "win_frac": None}
    d = [a[i][0] - b[i][0] for i in range(m)]
    mean = sum(d) / m
    var = sum((x - mean) ** 2 for x in d) / (m - 1)
    se = math.sqrt(var / m)
    ci = _t95(m - 1) * se
    return {
        "n": m,
        "mean": mean,
        "sd": math.sqrt(var),
        "se": se,
        "ci95": ci,
        # Note the absent `ci95 > 0` guard that `review_experiments._signif` has.
        # There it protects against a degenerate n=1 aggregate; here n >= 2 is
        # already enforced, and a paired difference with zero variance and a
        # non-zero mean is the *most* significant case there is, not the least.
        "sig": bool(abs(mean) > ci),
        "win_frac": sum(1 for x in d if x < 0) / m,
        "ce_a": token_weighted_ce(a[:m]),
        "ce_b": token_weighted_ce(b[:m]),
        "ppl_a": math.exp(token_weighted_ce(a[:m])),
        "ppl_b": math.exp(token_weighted_ce(b[:m])),
    }


def fmt_delta(d: Dict, unit: str = "nats") -> str:
    if d.get("mean") is None:
        return "n/a"
    star = "*" if d["sig"] else "ns"
    return (f"{d['mean']:+.4f} +/- {d['ci95']:.4f} {unit} [{star}] "
            f"(win {100.0 * d['win_frac']:.0f}% of {d['n']} windows)")


def _auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """Mann-Whitney AUC: P(pos > neg), with ties counted as 1/2."""
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


# --------------------------------------------------------------------------- #
# Per-layer operator installation
# --------------------------------------------------------------------------- #
# `wrap_model_with_qgfd_operator` installs ONE shared operator instance on every
# layer, which is right for a global alpha and wrong for anything per-layer. The
# probes and the gate both need one instance per layer, so we patch once to get
# the adapters and then swap each adapter's `prob_operator` for its own object.
def install_per_layer(model, factory, device: str, verify_tok=None,
                      label: str = "gated") -> List[nn.Module]:
    """`factory(layer_index, num_heads) -> operator`. Returns the operators."""
    model = wrap_model_with_qgfd_operator(model, SoftmaxOperator(), verbose=False)
    adapters = [m for m in model.modules() if isinstance(m, AttentionOperatorAdapter)]
    if not adapters:
        raise RuntimeError(
            f"[{label}] no attention layer was patched for "
            f"'{getattr(model.config, 'model_type', '?')}' — QGFD would be a no-op. "
            f"Use a Llama / Mistral / Qwen2 checkpoint."
        )
    n_heads = int(model.config.num_attention_heads)
    ops = []
    for i, ad in enumerate(adapters):
        op = factory(i, n_heads).to(device)
        ad.prob_operator = op
        ops.append(op)
    # A freshly constructed nn.Module has `training=True`, and assigning it as a
    # child does not inherit the parent's mode -- so without this the operators
    # take `get_alpha()`'s training branch during evaluation. With warmup_steps=0
    # the value is the same either way, but relying on that is a trap.
    model.eval()
    if verify_tok is not None:
        verify_patch(model, verify_tok, device, label)
    return ops


def n_query_heads(model) -> int:
    return int(model.config.num_attention_heads)


def load_arm(cfg: GatedConfig):
    """Load the checkpoint with this module's dtype/device choices."""
    tok, model, device = load_model_and_tokenizer(cfg.eval_config())
    model.eval()
    return tok, model, device


def eval_texts(cfg: GatedConfig) -> List[str]:
    return load_wikitext(cfg.n_eval_texts)


def train_texts(cfg: GatedConfig) -> List[str]:
    return load_wikitext(cfg.n_train_texts, split="train")


# --------------------------------------------------------------------------- #
# Stage 0a — dCE/dalpha for every head, from one backward pass
# --------------------------------------------------------------------------- #
# `QGFDOperator(learnable_alpha=True)` already carries a per-head
# `alpha_param` of shape (num_heads,), and in eval mode `get_alpha()` returns it
# clamped, with no warmup factor. So the derivative of the loss with respect to
# every head's mixing coefficient falls out of ONE forward+backward per window.
#
# One wrinkle: at alpha == 0 exactly, `forward` short-circuits to `p = p0` and no
# graph is built, so the gradient would be None. We therefore evaluate at
# alpha = 1e-6, which is numerically indistinguishable from zero (the TV
# displacement is bounded by alpha) but keeps the diffusion branch live.
_PROBE_EPS = 1e-6


def head_alpha_gradients(cfg: GatedConfig, corruption: str = "word_drop",
                         rate: float = 0.10) -> Dict:
    """Per-head dCE/dalpha at alpha ~ 0, on clean and on corrupted text.

    Reading:
      dCE_noisy/dalpha_h < 0  -> diffusion on head h lowers noisy loss at the
                                 margin. Real headroom; a gate can find it.
      dCE_noisy/dalpha_h > 0  -> the optimiser wants alpha NEGATIVE on that head,
                                 i.e. it wants to SHARPEN, not diffuse. That
                                 inverts the mechanism claim, and sharpening is a
                                 temperature change that folds into W_Q.
      all >= 0                -> no gate can help. Stop.
    """
    tok, model, device = load_arm(cfg)
    n_h = n_query_heads(model)
    ops = install_per_layer(
        model,
        lambda i, h: QGFDOperator(
            diffusion_steps=1, target_alpha=_PROBE_EPS, warmup_steps=0,
            detach_P=True, is_causal=True, learnable_alpha=True, num_heads=h,
            max_alpha=0.5, max_full_seq_len=max(512, cfg.seq_len),
            full_fallback_mode="disable",
        ),
        device, verify_tok=tok, label="stage0a",
    )
    for p in model.parameters():
        p.requires_grad_(False)
    for op in ops:
        op.alpha_param.requires_grad_(True)

    texts = eval_texts(cfg)
    rng = random.Random(cfg.eval_config().robustness_seed)
    noisy = [CORRUPTIONS[corruption](t, rate, rng) for t in texts]

    grads: Dict[str, List[List[float]]] = {}
    for label, corpus in (("clean", texts), ("noisy", noisy)):
        for op in ops:
            if op.alpha_param.grad is not None:
                op.alpha_param.grad = None
        n_win = _accumulate_alpha_grad(model, tok, corpus, device, cfg)
        grads[label] = [
            (op.alpha_param.grad / max(1, n_win)).detach().float().cpu().tolist()
            if op.alpha_param.grad is not None else [0.0] * n_h
            for op in ops
        ]
        grads[label + "_windows"] = n_win

    free(model)
    return _summarise_head_grads(cfg, corruption, rate, grads, len(ops), n_h)


def _accumulate_alpha_grad(model, tok, corpus, device, cfg: GatedConfig) -> int:
    """Sum dCE/dalpha over up to `cfg.probe_windows` windows. Returns the count."""
    enc = tok("\n\n".join(corpus), return_tensors="pt").input_ids.to(device)
    n, done = enc.size(1), 0
    for begin in range(0, n - 1, cfg.seq_len):
        ids = enc[:, begin:begin + cfg.seq_len]
        if ids.size(1) < 16:
            break
        with torch.enable_grad():
            loss = model(ids, labels=ids).loss.float()
        loss.backward()
        done += 1
        if done >= cfg.probe_windows:
            break
    return done


def _summarise_head_grads(cfg: GatedConfig, corruption: str, rate: float,
                          grads: Dict, n_layers: int, n_heads: int) -> Dict:
    flat_n = [g for layer in grads["noisy"] for g in layer]
    flat_c = [g for layer in grads["clean"] for g in layer]
    neg = [g for g in flat_n if g < 0.0]
    # A head is "promising" when diffusion helps under noise and does not hurt
    # much when clean -- that is exactly the trade a gate is able to exploit.
    promising = [
        {"layer": li, "head": hi,
         "d_noisy": grads["noisy"][li][hi], "d_clean": grads["clean"][li][hi]}
        for li in range(n_layers) for hi in range(n_heads)
        if grads["noisy"][li][hi] < 0.0
    ]
    promising.sort(key=lambda r: r["d_noisy"])
    per_layer = [sum(layer) / len(layer) for layer in grads["noisy"]]
    return {
        "experiment": "S0a_head_alpha_gradients",
        "meta": {"model_id": cfg.model_id, "corruption": corruption,
                 "rate": float(rate), "alpha_eval_point": _PROBE_EPS,
                 "seq_len": cfg.seq_len, "windows": grads.get("noisy_windows"),
                 "n_layers": n_layers, "n_heads": n_heads},
        "grad_noisy_by_layer_head": grads["noisy"],
        "grad_clean_by_layer_head": grads["clean"],
        "grad_noisy_mean_by_layer": per_layer,
        "summary": {
            "n_heads_total": len(flat_n),
            "n_heads_negative_noisy": len(neg),
            "frac_negative_noisy": len(neg) / max(1, len(flat_n)),
            "mean_grad_noisy": sum(flat_n) / max(1, len(flat_n)),
            "mean_grad_clean": sum(flat_c) / max(1, len(flat_c)),
            "min_grad_noisy": min(flat_n) if flat_n else None,
            "top_heads": promising[:12],
            "best_layers": sorted(range(n_layers), key=lambda i: per_layer[i])[:6],
        },
        "verdict": _grad_verdict(flat_n),
    }


def _grad_verdict(flat_noisy: Sequence[float]) -> str:
    if not flat_noisy:
        return "no_data"
    neg = sum(1 for g in flat_noisy if g < 0.0)
    frac = neg / len(flat_noisy)
    if neg == 0:
        # Every head wants alpha to move the other way. Not a null result -- an
        # inverted one, and the inverted direction is free (temperature).
        return "sharpen"
    if frac < 0.05:
        return "marginal"
    return "headroom"


# --------------------------------------------------------------------------- #
# Stage 0b — per-layer finite-difference alpha oracle
# --------------------------------------------------------------------------- #
def _set_alpha(ops: Sequence[nn.Module], value: float,
               only: Optional[int] = None) -> None:
    """Mutate alpha in place. In eval mode `get_alpha()` reads `target_alpha`
    directly, so there is no need to rebuild the model per value -- which keeps
    checkpoint loading out of the inner loop entirely."""
    for i, op in enumerate(ops):
        op.target_alpha = float(value) if (only is None or i == only) else 0.0


def layer_alpha_oracle(cfg: GatedConfig, corruption: str = "word_drop",
                       rate: float = 0.10, max_windows: int = 40) -> Dict:
    """Absolute noisy CE with alpha on exactly one layer at a time.

    Stage 0a is a first-order reading at a point; this confirms it at the actual
    alpha the paper would use, and catches any layer where the second-order term
    rescues a positive gradient. One model instance, n_layers + 1 evaluations.
    """
    tok, model, device = load_arm(cfg)
    ops = install_per_layer(
        model,
        lambda i, h: QGFDOperator(
            diffusion_steps=1, target_alpha=0.0, warmup_steps=0, detach_P=True,
            is_causal=True, max_alpha=0.5,
            max_full_seq_len=max(512, cfg.seq_len), full_fallback_mode="disable",
        ),
        device, verify_tok=tok, label="stage0b",
    )
    texts = eval_texts(cfg)
    rng = random.Random(cfg.eval_config().robustness_seed)
    noisy = [CORRUPTIONS[corruption](t, rate, rng) for t in texts]

    _set_alpha(ops, 0.0)
    base = window_ce(model, tok, noisy, device, cfg.seq_len, max_windows)
    base_ce = token_weighted_ce(base)

    rows = []
    for i in range(len(ops)):
        _set_alpha(ops, cfg.probe_alpha, only=i)
        w = window_ce(model, tok, noisy, device, cfg.seq_len, max_windows)
        d = paired_delta(w, base)
        rows.append({"layer": i, "ce": token_weighted_ce(w),
                     "delta_ce": d["mean"], "ci95": d["ci95"], "sig": d["sig"]})
    _set_alpha(ops, 0.0)
    free(model)

    helpful = [r for r in rows if r["delta_ce"] is not None
               and r["delta_ce"] < 0 and r["sig"]]
    return {
        "experiment": "S0b_layer_alpha_oracle",
        "meta": {"model_id": cfg.model_id, "corruption": corruption,
                 "rate": float(rate), "alpha": cfg.probe_alpha,
                 "seq_len": cfg.seq_len, "windows": len(base)},
        "baseline_noisy_ce": base_ce,
        "by_layer": rows,
        "summary": {
            "n_layers_helpful": len(helpful),
            "helpful_layers": [r["layer"] for r in helpful],
            "best_layer": min(rows, key=lambda r: r["delta_ce"])["layer"] if rows else None,
            "best_delta_ce": min((r["delta_ce"] for r in rows), default=None),
        },
        "verdict": "headroom" if helpful else "none",
    }


# --------------------------------------------------------------------------- #
# Stage 0c — corruption-family screen
# --------------------------------------------------------------------------- #
def corruption_screen(cfg: GatedConfig, families: Optional[Sequence[str]] = None,
                      rate: float = 0.10, max_windows: int = 60) -> Dict:
    """Fixed alpha, absolute paired CE, one row per corruption family.

    This is the one experiment that could still produce a positive result with
    the EXISTING fixed operator, and it is the reason the original headline was
    ambiguous: `char` noise re-tokenises, so a win there is a win at subword
    fragmentation repair, not at key perturbation.
    """
    families = tuple(families or cfg.probe_families)
    tok, model, device = load_arm(cfg)
    ops = install_per_layer(
        model,
        lambda i, h: QGFDOperator(
            diffusion_steps=1, target_alpha=0.0, warmup_steps=0, detach_P=True,
            is_causal=True, max_alpha=0.5,
            max_full_seq_len=max(512, cfg.seq_len), full_fallback_mode="disable",
        ),
        device, verify_tok=tok, label="stage0c",
    )
    texts = eval_texts(cfg)
    seed = cfg.eval_config().robustness_seed

    rows = []
    for fam in families:
        if fam not in CORRUPTIONS:
            rows.append({"corruption": fam, "error": "unknown family"})
            continue
        # Corrupt ONCE, then score both arms on the byte-identical string.
        variants = precompute_corruptions(texts, [float(rate)], fam, seed)
        corrupted = variants[float(rate)]
        _set_alpha(ops, 0.0)
        sm = window_ce(model, tok, corrupted, device, cfg.seq_len, max_windows)
        _set_alpha(ops, cfg.probe_alpha)
        qg = window_ce(model, tok, corrupted, device, cfg.seq_len, max_windows)
        d = paired_delta(qg, sm)
        rows.append({
            "corruption": fam,
            "preserves_tokenisation": fam in TOKEN_PRESERVING,
            "softmax_ce": d["ce_b"], "qgfd_ce": d["ce_a"],
            "delta_ce": d["mean"], "ci95": d["ci95"], "sig": d["sig"],
            "win_frac": d["win_frac"], "n_windows": d["n"],
        })
    _set_alpha(ops, 0.0)
    free(model)

    wins = [r for r in rows
            if r.get("delta_ce") is not None and r["delta_ce"] < 0 and r["sig"]]
    return {
        "experiment": "S0c_corruption_screen",
        "meta": {"model_id": cfg.model_id, "alpha": cfg.probe_alpha,
                 "rate": float(rate), "seq_len": cfg.seq_len,
                 "families": list(families)},
        "note": ("delta_ce = QGFD - softmax in nats/token on byte-identical "
                 "corrupted text; negative means QGFD is genuinely better in "
                 "ABSOLUTE terms, which is the only version of the claim that "
                 "survives the E9 denominator control."),
        "by_corruption": rows,
        "summary": {"winning_families": [r["corruption"] for r in wins],
                    "token_preserving_wins": [r["corruption"] for r in wins
                                              if r["preserves_tokenisation"]]},
        "verdict": "headroom" if wins else "none",
    }


# --------------------------------------------------------------------------- #
# Stage 0 runner and the stop rule
# --------------------------------------------------------------------------- #
def stage0(cfg: GatedConfig) -> Dict:
    """The whole screen. ~40 min on a T4, no training, no gate code involved.

    The stop rule is decided BEFORE the numbers are seen, which is the only way a
    stop rule does any work:

      stop     no head has a negative noisy-CE gradient AND no corruption family
               shows an absolute win. Write the negative result; do not build the
               gate. Nothing downstream can rescue this, because a gate with
               g >= 0 can only reach states that a fixed alpha already probed.
      pivot    gradients are uniformly positive -- the model wants to SHARPEN.
               Switch to the temperature/sharpening story, which is free and
               folds into W_Q.
      proceed  some heads or some family show real headroom. Go to Stage 1 and
               restrict the gate to the layers Stage 0b flagged.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    t0 = time.time()
    out: Dict[str, Dict] = {}
    fam = cfg.train_corruption
    out["s0a"] = head_alpha_gradients(cfg, corruption=fam, rate=cfg.eval_rate)
    out["s0b"] = layer_alpha_oracle(cfg, corruption=fam, rate=cfg.eval_rate)
    out["s0c"] = corruption_screen(cfg, rate=cfg.eval_rate)

    grad_v = out["s0a"]["verdict"]
    any_family = out["s0c"]["verdict"] == "headroom"
    any_layer = out["s0b"]["verdict"] == "headroom"
    if grad_v == "sharpen" and not any_family and not any_layer:
        decision, why = "pivot", (
            "No head wants alpha > 0 anywhere: every dCE_noisy/dalpha_h is "
            "positive, so the optimiser is asking for anti-diffusion. That is a "
            "temperature change, it is free, and it folds into W_Q. Report it as "
            "an inverted mechanism, not as a null.")
    elif grad_v in ("sharpen", "marginal") and not any_family and not any_layer:
        decision, why = "stop", (
            "No first-order headroom, no layer helps in isolation, and no "
            "corruption family gives an absolute win. A zero-initialised gate "
            "cannot reach anything a fixed alpha did not already probe, so "
            "Stages 1-3 would be spending GPU quota to confirm this. Publish the "
            "negative result plus the relative-metric critique.")
    else:
        decision, why = "proceed", (
            "There is exploitable structure: "
            f"{out['s0a']['summary']['n_heads_negative_noisy']}/"
            f"{out['s0a']['summary']['n_heads_total']} heads have a negative "
            f"noisy-CE gradient, "
            f"{out['s0b']['summary']['n_layers_helpful']} layer(s) help in "
            f"isolation, and the winning families are "
            f"{out['s0c']['summary']['winning_families'] or 'none'}. Restrict the "
            "Stage 1 gate to those layers.")

    out["decision"] = {"decision": decision, "why": why,
                       "elapsed_s": time.time() - t0}
    _save(out, cfg, "stage0.json")
    return out


# --------------------------------------------------------------------------- #
# The gated operator
# --------------------------------------------------------------------------- #
class GatedQGFDOperator(StructuredPOperator):
    """QGFD with a learned, zero-initialised, per-head per-position gate.

        p' = (1 - g) * p0 + g * (p0 P)
        g  = g_max * tanh(a_h * Hhat(p0) + b_h * margin(p0) + r_h)

    Design notes that are decisions, not details:

    * **tanh, not sigmoid.** The proposal had `g = g_max * sigmoid(.) in [0, g_max]`
      and asked for `g = 0` at init. Those are incompatible: a one-sided gate
      reaches 0 only at a boundary, where the gradient is 0 and training never
      starts. `g_max * tanh(z)` is exactly 0 at `z = 0` with `tanh'(0) = 1`. It
      also makes `g < 0` reachable, i.e. sharpening -- which is the direction the
      evidence so far actually points at.
    * **rho is dropped.** Damped diffusion `p_diff = (1-rho) p0 + rho (p0 P)`
      composed with the gate gives `(1 - g*rho) p0 + g*rho (p0 P)`, so rho is
      exactly reparameterisable into g. It is a redundant hyper-parameter.
    * **Two features, not four.** Entropy and the top-2 probability margin both
      fall out of the p0 that is already materialised, so they cost O(n^2) against
      the diffusion's O(n^3). Cross-layer disagreement needs plumbing across
      module boundaries and is deferred until the cheap features earn it.
    * **Exactness.** Under `torch.no_grad()` a gate that is identically zero
      short-circuits to `p0`, so every reported eval number at init is bit-exact
      softmax. With grad enabled the mixing always runs, because that is what
      keeps `dL/dg` alive at `g = 0`.

    Inherits `p_structure in {"real", "uniform", "shuffled"}` from
    `StructuredPOperator`, so the E4 controls apply to the gate unchanged.
    """

    def __init__(self, num_heads: int, g_max: float = 0.05, signed: bool = True,
                 use_entropy: bool = True, use_margin: bool = True,
                 feature_scale: float = 1.0, **kwargs):
        kwargs.setdefault("diffusion_steps", 1)
        kwargs.setdefault("detach_P", True)
        kwargs.setdefault("is_causal", True)
        kwargs.setdefault("full_fallback_mode", "disable")
        kwargs.setdefault("max_full_seq_len", 4096)
        kwargs["target_alpha"] = 0.0
        kwargs["warmup_steps"] = 0
        kwargs["learnable_alpha"] = False
        kwargs["num_heads"] = int(num_heads)
        super().__init__(**kwargs)
        assert self.diffusion_steps == 1, (
            "the gated operator implements T=1 only; T>1 is not on the table "
            "until one-step adaptive diffusion beats the baseline")
        H = int(num_heads)
        self.g_max = float(g_max)
        self.signed = bool(signed)
        self.use_entropy = bool(use_entropy)
        self.use_margin = bool(use_margin)
        self.feature_scale = float(feature_scale)
        self.w_entropy = nn.Parameter(torch.zeros(H))
        self.w_margin = nn.Parameter(torch.zeros(H))
        self.gate_bias = nn.Parameter(torch.zeros(H))
        # Differentiable L1 handle for the loss, and non-differentiable telemetry.
        self.gate_l1: Optional[torch.Tensor] = None
        self.track_gate = False
        self._gate_log: List[torch.Tensor] = []
        self._absmean_log: List[float] = []

    def gate_parameters(self) -> List[nn.Parameter]:
        return [self.w_entropy, self.w_margin, self.gate_bias]


    # -- features -------------------------------------------------------- #
    @staticmethod
    def _valid_counts(p0: torch.Tensor, valid_mask: Optional[torch.Tensor]
                      ) -> torch.Tensor:
        """Number of attendable keys per query position, shape broadcastable to
        (B, H, Lq). Needed to normalise entropy: position t of a causal row can
        hold at most log(t+1) nats, so raw entropy is not comparable across t."""
        B, H, Lq, Lk = p0.shape
        if valid_mask is not None:
            return valid_mask.to(torch.float32).sum(-1).clamp_min(2.0)
        counts = torch.arange(1, Lq + 1, device=p0.device, dtype=torch.float32)
        if Lq != Lk:                       # cached-KV decode: full prefix visible
            counts = torch.full((Lq,), float(Lk), device=p0.device)
        return counts.view(1, 1, Lq).clamp_min(2.0)

    def _gate(self, p0: torch.Tensor, valid_mask: Optional[torch.Tensor]
              ) -> torch.Tensor:
        """(B, H, Lq, 1) mixing coefficient in [-g_max, g_max]."""
        counts = self._valid_counts(p0, valid_mask)
        z = self.gate_bias.to(p0.dtype).view(1, -1, 1).expand(
            p0.size(0), -1, p0.size(2))
        if self.use_entropy:
            p = p0.clamp_min(1e-12)
            ent = -(p * p.log()).sum(-1)                       # (B, H, Lq)
            ent_hat = ent / counts.log()                       # ~[0, 1]
            z = z + self.w_entropy.to(p0.dtype).view(1, -1, 1) * (
                self.feature_scale * ent_hat)
        if self.use_margin:
            top2 = p0.topk(2, dim=-1).values
            margin = top2[..., 0] - top2[..., 1]               # [0, 1]
            z = z + self.w_margin.to(p0.dtype).view(1, -1, 1) * (
                self.feature_scale * margin)
        g = torch.tanh(z) if self.signed else torch.sigmoid(z)
        return (self.g_max * g).unsqueeze(-1)

    # -- forward --------------------------------------------------------- #
    def forward(self, scores, key_states=None, attention_mask=None,
                head_mask=None):
        assert scores.dim() == 4, f"expected [B,H,Q,K], got {tuple(scores.shape)}"
        B, H, Lq, Lk = scores.shape
        if attention_mask is not None:
            scores = self.apply_attention_mask(scores, attention_mask)
        p0 = F.softmax(scores.to(torch.float32), dim=-1)

        if not self.enable_qgfd:
            self.gate_l1 = None
            return p0.to(scores.dtype)

        valid_mask = self._build_valid_mask(scores, p0, attention_mask)
        g = self._gate(p0, valid_mask)
        self.gate_l1 = g.abs().mean()

        # Exactness at init: with no graph to build, an identically-zero gate must
        # return p0 untouched (not p0 renormalised), or the alpha=0 equivalence
        # claim quietly acquires an fp32 rounding error.
        if not torch.is_grad_enabled() and not bool((g != 0).any()):
            p = p0
        else:
            if key_states is None:
                raise ValueError("gated QGFD requires key_states")
            P = self.build_transition_from_keys(
                key_states, target_heads=H, is_causal=self.is_causal,
                dtype=p0.dtype)
            p = (1.0 - g) * p0 + g * torch.matmul(p0, P)
            # A signed gate can drive individual entries negative wherever the
            # graph routes mass to a key that p0 gave ~0 to; clamp and renormalise
            # so the output is always a distribution.
            p = p.clamp_min(0.0)
            if valid_mask is not None:
                p = p * valid_mask.to(p.dtype)
            p = p / p.sum(-1, keepdim=True).clamp_min(self._eps(p))

        if self.track_gate:
            self._gate_log.append(g.detach().float().mean(dim=(0, 2, 3)).cpu())
            self._absmean_log.append(float(g.detach().abs().mean()))
        if head_mask is not None:
            p = p * (head_mask.view(1, -1, 1, 1) if head_mask.dim() == 1
                     else head_mask)
        return p.to(scores.dtype)

    # -- telemetry ------------------------------------------------------- #
    def start_tracking(self) -> None:
        self.track_gate = True
        self._gate_log, self._absmean_log = [], []

    def stop_tracking(self) -> Dict:
        """Per-head mean signed gate, and one |g| scalar per forward pass."""
        self.track_gate = False
        log, absl = self._gate_log, self._absmean_log
        self._gate_log, self._absmean_log = [], []
        if not log:
            return {"per_head": None, "per_forward_absmean": []}
        stacked = torch.stack(log)
        return {"per_head": stacked.mean(0).tolist(),
                "frac_head_inert": float(
                    (stacked.mean(0).abs() < 0.01 * self.g_max).float().mean()),
                "per_forward_absmean": absl}


# --------------------------------------------------------------------------- #
# Stage 2: the trust-aware transition graph
# --------------------------------------------------------------------------- #
class TrustGatedOperator(GatedQGFDOperator):
    """Gated QGFD whose graph is sparse, self-preferring and reliability-weighted.

        P_ij = softmax_j( k_i.k_j / (|k| |k| tau) + s * [i == j] + lambda * (-z_j) )
               restricted to the top-k neighbours inside the causal prefix

    Deviations from the proposal, each with a reason:

    * **`agreement_ij` is deferred.** It needs the previous layer's attention
      inside this layer's operator, i.e. plumbing across a module boundary that
      does not exist yet. It is the most expensive feature in the design and the
      least justified until the cheap ones show something.
    * **`log r_j` is a free signal, not a learned sub-model.** `z_j` is the
      standardised key norm; unusually long keys are the ones that dominate
      similarity scores for reasons unrelated to content, so `-z_j` is a
      reasonable reliability prior. `lambda` is one learned scalar per layer,
      initialised to 0, so the term starts inert and has to earn its place.
    * **Masking order.** Causal mask and top-k are applied to the LOGITS, before
      normalisation, so every row is a proper distribution over exactly the edges
      that survived. Masking after softmax would leave rows summing to < 1.

    top-k is also the reason Stage 2 is affordable: `p0 P` drops from O(n^3) to
    O(n^2 k).
    """

    def __init__(self, num_heads: int, top_k: int = 8, self_loop: float = 1.0,
                 reliability: bool = True, **kwargs):
        super().__init__(num_heads=num_heads, **kwargs)
        self.top_k = int(top_k)
        self.self_loop = float(self_loop)
        self.use_reliability = bool(reliability)
        self.w_trust = nn.Parameter(torch.zeros(1))

    def gate_parameters(self) -> List[nn.Parameter]:
        return super().gate_parameters() + [self.w_trust]

    def build_transition_from_keys(self, K, target_heads=None, is_causal=None,
                                   dtype=None):
        if is_causal is None:
            is_causal = getattr(self, "is_causal", False)
        if self.p_structure == "uniform":
            return self._uniform_P(K, target_heads, is_causal, dtype)
        P = self._trust_P(K, target_heads, is_causal, dtype)
        if self.p_structure == "shuffled":
            P = self._shuffle_rows(P, is_causal)
        return P

    def _trust_P(self, K, target_heads, is_causal, dtype):
        out_dtype = dtype if dtype is not None else K.dtype
        K32 = K.to(torch.float32)
        B, H_k, Lk, _ = K32.shape
        Kn = F.normalize(K32, p=2, dim=-1, eps=1e-6)
        sim = torch.matmul(Kn, Kn.transpose(-1, -2)) / self.temp

        if self.use_reliability:
            norm = K32.norm(dim=-1)                                  # (B,H_k,Lk)
            z = (norm - norm.mean(-1, keepdim=True)) / norm.std(
                -1, keepdim=True).clamp_min(1e-6)
            sim = sim + self.w_trust.to(sim.dtype) * (-z).unsqueeze(-2)

        if self.self_loop:
            eye = torch.eye(Lk, device=sim.device, dtype=sim.dtype)
            sim = sim + self.self_loop * eye

        neg = torch.finfo(sim.dtype).min
        if is_causal:
            causal = torch.tril(torch.ones((Lk, Lk), device=sim.device,
                                           dtype=torch.bool))
            sim = sim.masked_fill(~causal[None, None], neg)

        if 0 < self.top_k < Lk:
            # `topk` on a row whose prefix is shorter than k returns the masked
            # sentinel as its k-th value, so the threshold degenerates to `neg`
            # and the row is left intact. That is the behaviour we want.
            kth = sim.topk(self.top_k, dim=-1).values[..., -1:]
            sim = sim.masked_fill(sim < kth, neg)

        P = F.softmax(sim, dim=-1)
        if Lk > 1:                       # row 0 attends only to itself
            row0 = torch.zeros_like(P[:, :, :1, :])
            row0[..., 0] = 1.0
            P = torch.cat([row0, P[:, :, 1:, :]], dim=2)
        if target_heads is not None and H_k != target_heads:
            P = torch.repeat_interleave(P, target_heads // H_k, dim=1)
        P = P.to(out_dtype)
        return P.detach() if self.detach_P else P


# --------------------------------------------------------------------------- #
# Building and training the gate
# --------------------------------------------------------------------------- #
def make_gated_ops(cfg: GatedConfig, model, device: str, tok=None,
                   trust: bool = False, p_structure: str = "real",
                   layers: Optional[Sequence[int]] = None,
                   label: str = "gated") -> List[GatedQGFDOperator]:
    """Install one gated operator per layer. `layers=None` means every layer;
    otherwise the others are installed inert (`enable_qgfd=False`) so the model
    keeps a single uniform operator type and the telemetry stays comparable."""
    allowed = None if layers is None else set(int(i) for i in layers)

    def factory(i: int, h: int):
        common = dict(
            g_max=cfg.g_max, signed=cfg.signed, use_entropy=cfg.use_entropy,
            use_margin=cfg.use_margin, p_structure=p_structure,
            max_full_seq_len=max(512, cfg.seq_len),
            enable_qgfd=(allowed is None or i in allowed),
        )
        if trust:
            return TrustGatedOperator(num_heads=h, top_k=cfg.top_k,
                                      self_loop=cfg.self_loop,
                                      reliability=cfg.reliability, **common)
        return GatedQGFDOperator(num_heads=h, **common)

    return install_per_layer(model, factory, device, verify_tok=tok, label=label)


def _blocks(tok, texts: Sequence[str], seq_len: int) -> torch.Tensor:
    ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
    n = (ids.size(0) // seq_len) * seq_len
    if n == 0:
        raise RuntimeError("training corpus is shorter than one block")
    return ids[:n].view(-1, seq_len)


def _attach_lora(cfg: GatedConfig, model):
    """Q/K adapters only. Not Q/K/V/O: if the hypothesis is about the key graph,
    V and O adapters let the model win for unrelated reasons and the attribution
    is gone. Returns the trainable LoRA parameters (cast to fp32)."""
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=list(cfg.lora_targets))
    model = get_peft_model(model, lcfg)
    params = []
    for n, p in model.named_parameters():
        if "lora_" in n:
            # Left in the base model's dtype on purpose: peft's LoRA forward
            # matmuls the adapter against fp16 activations, and an fp32 adapter
            # weight raises a dtype mismatch there. GradScaler covers underflow.
            p.requires_grad_(True)
            params.append(p)
    if not params:
        raise RuntimeError(f"LoRA matched no module in {cfg.lora_targets}")
    return model, params


def _grad_scaler(enabled: bool):
    """`torch.cuda.amp.GradScaler` is deprecated in torch >= 2.4."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_gate(cfg: GatedConfig, model, tok, ops: Sequence[GatedQGFDOperator],
               device: str, texts: Sequence[str], seed: int,
               use_lora: bool = False, verbose: bool = True) -> Dict:
    """Paired clean/noisy training of the gate (Stage 1) or gate + Q/K LoRA (2).

        L = CE_clean + lambda_n(t) * CE_noisy + gamma * mean|g|

    No KL-to-teacher term and no teacher network. At `g = 0` the student IS the
    frozen softmax model, exactly, so the teacher is a third forward pass and a
    second set of weights in 16 GB to reproduce a constraint that the clean CE
    term plus the eps-constrained checkpoint selection already enforce.

    `lambda_n` ramps from 0 and the corruption rate follows the curriculum, so the
    gate first learns not to break clean text and only then learns when to fire.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)

    lora_params: List[nn.Parameter] = []
    if use_lora:
        model, lora_params = _attach_lora(cfg, model)

    gate_params = [p for op in ops if op.enable_qgfd
                   for p in op.gate_parameters()]
    for p in gate_params:                 # peft freezes everything it does not own
        p.requires_grad_(True)
    live = [op for op in ops if op.enable_qgfd]
    # The Stage 2 LoRA-only arm deliberately has no gate at all, so "no gate
    # parameters" is only an error when there is nothing else to train either.
    if not gate_params and not lora_params:
        raise RuntimeError("nothing is trainable: no gate and no LoRA parameters")

    groups = []
    if gate_params:
        groups.append({"params": gate_params, "lr": cfg.gate_lr})
    if lora_params:
        groups.append({"params": lora_params, "lr": cfg.lora_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    use_scaler = (device == "cuda" and cfg.dtype == "float16")
    scaler = _grad_scaler(use_scaler)
    if cfg.grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    lo, hi = cfg.curriculum
    rng = random.Random(seed)
    fn = CORRUPTIONS[cfg.train_corruption]
    clean = _blocks(tok, texts, cfg.seq_len)
    noisy_lo = _blocks(tok, [fn(t, lo, rng) for t in texts], cfg.seq_len)
    noisy_hi = _blocks(tok, [fn(t, hi, rng) for t in texts], cfg.seq_len)

    ramp = max(1, int(cfg.lambda_ramp_frac * cfg.max_steps))
    model.train()
    history = []
    t0 = time.time()
    for step in range(cfg.max_steps):
        lam = cfg.lambda_noisy_max * min(1.0, step / ramp)
        noisy = noisy_lo if step < cfg.max_steps // 2 else noisy_hi
        opt.zero_grad(set_to_none=True)
        acc = {"clean": 0.0, "noisy": 0.0, "l1": 0.0}
        for micro in range(cfg.grad_accum):
            i = (step * cfg.grad_accum + micro)
            cb = clean[i % clean.size(0)].unsqueeze(0).to(device)
            nb = noisy[i % noisy.size(0)].unsqueeze(0).to(device)
            ce_clean = model(cb, labels=cb).loss.float()
            ce_noisy = model(nb, labels=nb).loss.float()
            l1s = [op.gate_l1 for op in live if op.gate_l1 is not None]
            l1 = (torch.stack(l1s).mean() if l1s
                  else torch.zeros((), device=ce_clean.device))
            loss = (ce_clean + lam * ce_noisy + cfg.gamma_l1 * l1) / cfg.grad_accum
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            acc["clean"] += ce_clean.item() / cfg.grad_accum
            acc["noisy"] += ce_noisy.item() / cfg.grad_accum
            acc["l1"] += l1.item() / cfg.grad_accum
        if use_scaler:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(gate_params + lora_params, 1.0)
        if use_scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if verbose and (step % max(1, cfg.max_steps // 10) == 0
                        or step == cfg.max_steps - 1):
            print(f"    step {step:4d}  ce_clean {acc['clean']:.4f}  "
                  f"ce_noisy {acc['noisy']:.4f}  |g| {acc['l1']:.5f}  "
                  f"lambda_n {lam:.2f}")
        history.append({"step": step, "lambda_noisy": lam, **acc})
    model.eval()
    return {"history": history, "train_s": time.time() - t0,
            "n_gate_params": sum(p.numel() for p in gate_params),
            "n_lora_params": sum(p.numel() for p in lora_params),
            "model": model}


# --------------------------------------------------------------------------- #
# Evaluating a trained gate
# --------------------------------------------------------------------------- #
def _toggle(ops: Sequence[GatedQGFDOperator], live: Sequence[GatedQGFDOperator],
            on: bool) -> None:
    for op in live:
        op.enable_qgfd = bool(on)


def _track(live: Sequence[GatedQGFDOperator], start: bool) -> Optional[Dict]:
    if start:
        for op in live:
            op.start_tracking()
        return None
    logs = [op.stop_tracking() for op in live]
    per_forward = [sum(vals) / len(logs)
                   for vals in zip(*[l["per_forward_absmean"] for l in logs])] \
        if all(l["per_forward_absmean"] for l in logs) else []
    return {"per_layer_per_head": [l["per_head"] for l in logs],
            "frac_inert_by_layer": [l.get("frac_head_inert") for l in logs],
            "per_window_absmean": per_forward,
            "mean_absmean": (sum(per_forward) / len(per_forward)
                             if per_forward else None)}


def evaluate_gate(cfg: GatedConfig, model, tok, ops: Sequence[GatedQGFDOperator],
                  device: str, texts: Sequence[str],
                  controls: bool = True) -> Dict:
    """Constrained-Pareto evaluation, plus the diffusion-utilisation table.

    The softmax reference is the SAME weights with the gate switched off, which
    for Stage 1 (frozen base) is literally the pretrained model. That makes the
    clean-CE constraint and the noisy-CE claim exactly paired: same weights, same
    windows, same bytes, one difference.
    """
    live = [op for op in ops if op.enable_qgfd]
    rate = float(cfg.eval_rate)
    corrupted = precompute_corruptions(
        texts, [rate], cfg.eval_corruption,
        cfg.eval_config().robustness_seed)[rate]
    W = cfg.eval_max_windows

    _track(live, True)
    on_clean = window_ce(model, tok, texts, device, cfg.seq_len, W)
    gate_clean = _track(live, False)
    _track(live, True)
    on_noisy = window_ce(model, tok, corrupted, device, cfg.seq_len, W)
    gate_noisy = _track(live, False)

    _toggle(ops, live, False)
    off_clean = window_ce(model, tok, texts, device, cfg.seq_len, W)
    off_noisy = window_ce(model, tok, corrupted, device, cfg.seq_len, W)
    _toggle(ops, live, True)

    d_clean = paired_delta(on_clean, off_clean)
    d_noisy = paired_delta(on_noisy, off_noisy)
    eps_ok = (d_clean["mean"] is not None
              and d_clean["mean"] <= cfg.epsilon_clean_nats)
    noisy_win = bool(d_noisy["mean"] is not None and d_noisy["mean"] < 0
                     and d_noisy["sig"])

    out = {
        "corruption": cfg.eval_corruption, "rate": rate,
        "delta_ce_clean": d_clean, "delta_ce_noisy": d_noisy,
        "epsilon_clean_nats": cfg.epsilon_clean_nats,
        "epsilon_satisfied": bool(eps_ok), "noisy_win": noisy_win,
        "pareto": bool(eps_ok and noisy_win),
        "gate_clean": gate_clean, "gate_noisy": gate_noisy,
        "gate_auc_noisy_vs_clean": _auc(
            gate_noisy.get("per_window_absmean", []) if gate_noisy else [],
            gate_clean.get("per_window_absmean", []) if gate_clean else []),
    }
    if controls:
        out["controls"] = _p_structure_controls(cfg, model, tok, live, device,
                                                corrupted, off_noisy, W)
    return out


def _p_structure_controls(cfg: GatedConfig, model, tok, live, device: str,
                          corrupted: Sequence[str],
                          off_noisy: Sequence[Tuple[float, int]],
                          max_windows: int) -> Dict:
    """The E4 falsifiers, re-run against the TRAINED gate.

    A gate that still beats softmax with `uniform` or `shuffled` P has not learned
    to route along the key graph. It has learned "spread mass when the row is
    uncertain", which is an adaptive temperature schedule -- free, and foldable
    into W_Q. `shuffled` is the sharper of the two: it holds each row's entire
    multiset of probabilities (entropy, max, sparsity) and changes only which key
    receives which mass.
    """
    out = {}
    original = [op.p_structure for op in live]
    try:
        for structure in ("uniform", "shuffled"):
            for op in live:
                op.p_structure = structure
            w = window_ce(model, tok, corrupted, device, cfg.seq_len, max_windows)
            d = paired_delta(w, off_noisy)
            out[structure] = {"delta_ce_noisy": d,
                              "beats_softmax": bool(d["mean"] is not None
                                                    and d["mean"] < 0 and d["sig"])}
    finally:
        for op, s in zip(live, original):
            op.p_structure = s
    real_needed = not (out.get("shuffled", {}).get("beats_softmax")
                       or out.get("uniform", {}).get("beats_softmax"))
    out["verdict"] = ("graph_matters" if real_needed else
                      "temperature_in_disguise")
    out["note"] = ("If a structure-free P reproduces the gain, the result is an "
                   "adaptive temperature schedule, not graph routing, and it "
                   "should be reported as such.")
    return out


# --------------------------------------------------------------------------- #
# Stage 1 — gate only
# --------------------------------------------------------------------------- #
def stage1(cfg: GatedConfig, layers: Optional[Sequence[int]] = None) -> Dict:
    """Smallest model, base frozen, gate parameters only (~3 per head).

    ~1k trainable parameters. There is nothing here that can overfit in an
    interesting way, which is the point: if the gate cannot win with the base
    model untouched, a win after LoRA is a win by the adapters, not by diffusion.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    tr, ev = train_texts(cfg), eval_texts(cfg)
    runs = []
    for seed in cfg.seeds:
        print(f"\n[stage1] seed {seed}")
        tok, model, device = load_arm(cfg)
        ops = make_gated_ops(cfg, model, device, tok=tok, layers=layers,
                             label=f"stage1/s{seed}")
        info = train_gate(cfg, model, tok, ops, device, tr, seed, use_lora=False)
        model = info.pop("model")
        res = evaluate_gate(cfg, model, tok, ops, device, ev)
        res.update({"seed": seed, "train_s": info["train_s"],
                    "n_gate_params": info["n_gate_params"],
                    "final_train": info["history"][-1] if info["history"] else None})
        runs.append(res)
        print(f"  clean  {fmt_delta(res['delta_ce_clean'])}")
        print(f"  noisy  {fmt_delta(res['delta_ce_noisy'])}")
        print(f"  pareto {res['pareto']}  controls "
              f"{res.get('controls', {}).get('verdict')}")
        free(model)
    out = {"experiment": "S1_gate_only",
           "meta": {"model_id": cfg.model_id, "seq_len": cfg.seq_len,
                    "g_max": cfg.g_max, "signed": cfg.signed,
                    "max_steps": cfg.max_steps, "seeds": list(cfg.seeds),
                    "layers": None if layers is None else list(layers)},
           "runs": runs, "summary": _summarise_runs(runs)}
    _save(out, cfg, "stage1.json")
    return out


def _summarise_runs(runs: Sequence[Dict]) -> Dict:
    def col(key, sub="mean"):
        return [r[key][sub] for r in runs
                if r.get(key) and r[key].get(sub) is not None]
    clean, noisy = col("delta_ce_clean"), col("delta_ce_noisy")
    n_pareto = sum(1 for r in runs if r.get("pareto"))
    # Stage 2 pairs two trained arms and has no gate-off control to falsify, so
    # an absent `controls` block means "not applicable", never "failed".
    ctrl = [r["controls"]["verdict"] for r in runs if r.get("controls")]
    graph_ok = all(v == "graph_matters" for v in ctrl) if ctrl else None
    return {
        "n_seeds": len(runs),
        "mean_delta_ce_clean": sum(clean) / len(clean) if clean else None,
        "mean_delta_ce_noisy": sum(noisy) / len(noisy) if noisy else None,
        "n_pareto": n_pareto,
        "all_seeds_pareto": bool(runs) and n_pareto == len(runs),
        "controls": ctrl or None,
        "graph_matters_all_seeds": graph_ok,
        "verdict": ("temperature" if any(v == "temperature_in_disguise"
                                        for v in ctrl)
                    else "success" if runs and n_pareto == len(runs)
                    and graph_ok is not False
                    else "fail"),
    }


# --------------------------------------------------------------------------- #
# Stage 2 — trust-aware sparse graph + Q/K LoRA
# --------------------------------------------------------------------------- #
def stage2(cfg: GatedConfig, layers: Optional[Sequence[int]] = None,
           seeds: Optional[Sequence[int]] = None) -> Dict:
    """Two arms, identical seed / data / steps / LR: LoRA-only vs gate+trust+LoRA.

    Once LoRA moves the weights, "gate off" is no longer the right control -- the
    adapters are in both arms, so the comparison has to be across two separately
    trained models, paired at the window level.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    seeds = tuple(seeds or cfg.seeds[:1])
    tr, ev = train_texts(cfg), eval_texts(cfg)
    rate = float(cfg.eval_rate)
    corrupted = precompute_corruptions(
        ev, [rate], cfg.eval_corruption,
        cfg.eval_config().robustness_seed)[rate]
    runs = []
    for seed in seeds:
        arm_windows: Dict[str, Dict[str, List[Tuple[float, int]]]] = {}
        extra: Dict[str, Dict] = {}
        for arm in ("lora", "trust_gate_lora"):
            print(f"\n[stage2] seed {seed} arm {arm}")
            tok, model, device = load_arm(cfg)
            ops = make_gated_ops(
                cfg, model, device, tok=tok, trust=(arm != "lora"),
                # the LoRA-only arm installs the same operator type with the
                # diffusion inert, so the two arms differ only in the gate
                layers=(layers if arm != "lora" else []),
                label=f"stage2/{arm}/s{seed}")
            info = train_gate(cfg, model, tok, ops, device, tr, seed,
                              use_lora=True)
            model = info.pop("model")
            live = [op for op in ops if op.enable_qgfd]
            if live:
                _track(live, True)
            arm_windows[arm] = {
                "clean": window_ce(model, tok, ev, device, cfg.seq_len,
                                   cfg.eval_max_windows),
                "noisy": window_ce(model, tok, corrupted, device, cfg.seq_len,
                                   cfg.eval_max_windows),
            }
            extra[arm] = {"train_s": info["train_s"],
                          "n_lora_params": info["n_lora_params"],
                          "gate": _track(live, False) if live else None}
            free(model)
        g, b = arm_windows["trust_gate_lora"], arm_windows["lora"]
        d_clean = paired_delta(g["clean"], b["clean"])
        d_noisy = paired_delta(g["noisy"], b["noisy"])
        eps_ok = (d_clean["mean"] is not None
                  and d_clean["mean"] <= cfg.epsilon_clean_nats)
        win = bool(d_noisy["mean"] is not None and d_noisy["mean"] < 0
                   and d_noisy["sig"])
        runs.append({"seed": seed, "delta_ce_clean": d_clean,
                     "delta_ce_noisy": d_noisy, "epsilon_satisfied": eps_ok,
                     "noisy_win": win, "pareto": bool(eps_ok and win),
                     "arms": extra})
        print(f"  clean  {fmt_delta(d_clean)}")
        print(f"  noisy  {fmt_delta(d_noisy)}")
    out = {"experiment": "S2_trust_graph_lora",
           "meta": {"model_id": cfg.model_id, "top_k": cfg.top_k,
                    "self_loop": cfg.self_loop, "reliability": cfg.reliability,
                    "lora_targets": list(cfg.lora_targets), "seeds": list(seeds),
                    "max_steps": cfg.max_steps, "seq_len": cfg.seq_len},
           "runs": runs, "summary": _summarise_runs(runs)}
    _save(out, cfg, "stage2.json")
    return out


# --------------------------------------------------------------------------- #
# Stage 3 — recipe transfer
# --------------------------------------------------------------------------- #
LARGER_MODELS = ("Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")


def stage3(cfg: GatedConfig, models: Sequence[str] = LARGER_MODELS,
           with_lora: bool = False) -> Dict:
    """Same recipe, one seed, no per-model tuning.

    Deliberately un-tuned. If the effect is a mechanism it survives a transfer;
    if it needs its hyper-parameters re-fitted per checkpoint it is a fit, and the
    honest thing is to report it as one.
    """
    out = {"experiment": "S3_scale_transfer", "by_model": {},
           "meta": {"source_model": cfg.model_id, "with_lora": with_lora}}
    for mid in models:
        print(f"\n[stage3] {mid}")
        sub = replace(cfg, model_id=mid, seeds=(cfg.seeds[0],),
                      out_dir=os.path.join(cfg.out_dir, "stage3",
                                           mid.split("/")[-1]))
        try:
            out["by_model"][mid] = (stage2(sub, seeds=(cfg.seeds[0],))
                                    if with_lora else stage1(sub))
        except Exception as exc:                      # one model must not abort
            out["by_model"][mid] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    verdicts = [v.get("summary", {}).get("verdict")
                for v in out["by_model"].values() if "summary" in v]
    out["summary"] = {
        "verdicts": verdicts,
        "transfers": bool(verdicts) and all(v == "success" for v in verdicts),
    }
    _save(out, cfg, "stage3.json")
    return out


# --------------------------------------------------------------------------- #
# I/O and CLI
# --------------------------------------------------------------------------- #
def _save(payload: Dict, cfg: GatedConfig, name: str) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, name)
    with open(path, "w") as fh:
        json.dump({"config": asdict(cfg), **payload}, fh, indent=2, default=str)
    print(f"  wrote {path}")
    return path


def apply_quick(cfg: GatedConfig) -> GatedConfig:
    """A few minutes on CPU, exercising every code path. Run this first."""
    return replace(
        cfg, model_id="JackFram/llama-160m", dtype="float32", seq_len=64,
        n_train_texts=24, n_eval_texts=12, eval_max_windows=6, probe_windows=2,
        max_steps=4, grad_accum=1, seeds=(0,), lambda_ramp_frac=0.5,
        probe_families=("word_drop", "char"), quick=True,
    )


def run_stages(cfg: GatedConfig, stages: Sequence[str],
               honour_stop_rule: bool = True) -> Dict:
    """Run the requested stages, obeying Stage 0's stop rule by default.

    `honour_stop_rule=False` exists so the notebook can be forced past a "stop"
    for a rehearsal, but the default is to stop -- a stop rule that is routinely
    overridden is decoration.
    """
    out: Dict[str, Dict] = {}
    layers: Optional[List[int]] = None
    if "0" in stages:
        out["stage0"] = stage0(cfg)
        decision = out["stage0"]["decision"]["decision"]
        print(f"\n[stage0] decision: {decision}\n  "
              f"{out['stage0']['decision']['why']}")
        helpful = out["stage0"]["s0b"]["summary"]["helpful_layers"]
        if helpful:
            layers = helpful
            print(f"  restricting the Stage 1 gate to layers {layers}")
        if decision != "proceed" and honour_stop_rule:
            out["stopped_after"] = "stage0"
            _save(out, cfg, "summary.json")
            return out
    if "1" in stages:
        out["stage1"] = stage1(cfg, layers=layers)
        if (out["stage1"]["summary"]["verdict"] != "success"
                and honour_stop_rule):
            out["stopped_after"] = "stage1"
            _save(out, cfg, "summary.json")
            return out
    if "2" in stages:
        out["stage2"] = stage2(cfg, layers=layers)
        if (out["stage2"]["summary"]["verdict"] != "success"
                and honour_stop_rule):
            out["stopped_after"] = "stage2"
            _save(out, cfg, "summary.json")
            return out
    if "3" in stages:
        out["stage3"] = stage3(cfg)
    _save(out, cfg, "summary.json")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stages", default="0",
                    help="comma-separated subset of 0,1,2,3 (default: 0)")
    ap.add_argument("--model_id", default=GatedConfig.model_id)
    ap.add_argument("--dtype", default=GatedConfig.dtype,
                    choices=("float16", "bfloat16", "float32"))
    ap.add_argument("--seq_len", type=int, default=GatedConfig.seq_len)
    ap.add_argument("--g_max", type=float, default=GatedConfig.g_max)
    ap.add_argument("--unsigned_gate", action="store_true",
                    help="sigmoid gate in [0, g_max]; not exactly 0 at init")
    ap.add_argument("--top_k", type=int, default=GatedConfig.top_k)
    ap.add_argument("--max_steps", type=int, default=GatedConfig.max_steps)
    ap.add_argument("--train_corruption", default=GatedConfig.train_corruption)
    ap.add_argument("--eval_corruption", default=GatedConfig.eval_corruption)
    ap.add_argument("--eval_rate", type=float, default=GatedConfig.eval_rate)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out_dir", default=GatedConfig.out_dir)
    ap.add_argument("--force", action="store_true",
                    help="ignore Stage 0's stop rule")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)

    cfg = GatedConfig(
        model_id=a.model_id, dtype=a.dtype, seq_len=a.seq_len, g_max=a.g_max,
        signed=not a.unsigned_gate, top_k=a.top_k, max_steps=a.max_steps,
        train_corruption=a.train_corruption, eval_corruption=a.eval_corruption,
        eval_rate=a.eval_rate, out_dir=a.out_dir,
        seeds=tuple(int(s) for s in a.seeds.split(",") if s.strip()),
    )
    if a.quick:
        cfg = apply_quick(cfg)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    run_stages(cfg, stages, honour_stop_rule=not a.force)


if __name__ == "__main__":
    main()
