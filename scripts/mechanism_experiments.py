"""
QGFD mechanism experiments — the falsifiers.
===========================================
The main harness (`scripts/review_experiments.py`) answers "does QGFD help?".
This module answers the three questions a reviewer asks next, each of which can
invalidate the contribution on its own:

  E2  Does the theory describe the mechanism?
      Clean-PPL cost should grow as alpha^2, not alpha. Near a converged
      optimum the gradient vanishes, so the leading loss term is quadratic.
      Fitting the exponent is a falsifiable test of that argument.

  E3  Does QGFD beat the FREE alternative?
      QGFD's dominant first-order effect is a change in attention entropy.
      Temperature scaling -- softmax(scores / tau) -- changes entropy too, at
      zero compute and zero memory cost, foldable into W_Q. If an
      ENTROPY-MATCHED temperature arm recovers the same robustness gain, QGFD's
      robustness contribution is nil. This is the experiment most likely to
      kill the paper, so it is cheap and it runs early.

  E4  Does key SIMILARITY do any work?
      Replace P = softmax(KK^T/sqrt(d)) with (a) uniform over the causal past
      and (b) each row's values randomly permuted within the causal prefix.
      Both preserve row-stochasticity and the alpha displacement bound; only
      the real P carries key-graph structure. If QGFD ~= uniform-P then the
      "graph" in Query-Graph Flow Diffusion is decoration.

  E6  Is the gain about noise, or about BPE fragmentation?
      Character noise changes the tokenisation, so "diffuse onto similar keys"
      may be recovering from subword fragmentation specifically. Word deletion
      and local word swaps corrupt the input without shredding the tokeniser.

  E8  Is 1.6-1.9x compute better spent on a bigger model?
      The reviewer's question. Compare QGFD on the small model against plain
      softmax on the next size up, at matched wall-clock, under noise.

Design notes that matter for the statistics
-------------------------------------------
* Every sweep here holds the text sample FIXED and pre-computes each corrupted
  variant ONCE, then reuses it across alphas / operators. That makes the
  comparison paired at the level of the individual token, which is far tighter
  than pairing at the level of the seed.
* alpha is mutated on a SINGLE model instance rather than rebuilt per value.
  `QGFDOperator.get_alpha()` in eval mode returns `target_alpha` clamped to
  `max_alpha`, so mutating the attribute is sufficient -- and it removes model
  re-loading from the inner loop entirely.
* alpha=0 is used as the softmax baseline inside the sweeps. That is legitimate
  because the equivalence is verified bit-exact (max |dlogit| = 0.0), not
  assumed.

Nothing here writes to `paper/REPORT.md`; these are mechanism results reported
in their own section. See docs/interpreting-results.md.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from torchdire.nn.attention_operators import (
    AttentionProbabilityOperator,
    QGFDOperator,
)
from torchdire import wrap_model_with_qgfd_operator

from scripts.review_experiments import (
    ExperimentConfig,
    compute_perplexity,
    load_model_and_tokenizer,
    load_wikitext,
    perturb_text,
    verify_patch,
    free,
)
# --------------------------------------------------------------------------- #
# E3 — the free alternative: temperature scaling
# --------------------------------------------------------------------------- #
class TemperatureOperator(AttentionProbabilityOperator):
    """
    softmax(scores / tau). Deliberately the cheapest possible way to change
    attention entropy: no extra matmul, no extra memory, and it is *foldable*
    into W_Q (scaling W_Q by 1/tau produces identical logits), so a pretrained
    model already had both the capacity and the gradient signal to pick any tau
    it wanted. That makes it the right null hypothesis for QGFD's robustness
    claim rather than a strawman.

    tau > 1 flattens the distribution; tau < 1 sharpens it. Masking mirrors
    SoftmaxOperator exactly so the only difference between the arms is the
    divisor.
    """

    # Anything this far below a row's maximum contributes exactly 0.0 after
    # exp() for any tau in the search range, so clamping here is lossless.
    _FLOOR = -1e4

    def __init__(self, tau: float = 1.0, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert tau > 0, f"tau must be positive, got {tau}"
        self.tau = float(tau)
        self.dtype = dtype

    def forward(self, scores, key_states=None, attention_mask=None, head_mask=None):
        x = scores.to(self.dtype)
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                x = x.masked_fill(~attention_mask, torch.finfo(x.dtype).min)
            else:
                x = x + attention_mask.to(x.dtype)

        # Masked entries arrive as `torch.finfo(dtype).min` sentinels, and
        # dividing one of those by tau < 1 overflows to -inf; a row that is
        # entirely masked then softmaxes to NaN, where SoftmaxOperator would
        # have returned the uniform vector. Softmax is shift-invariant, so
        # subtracting the row max and flooring at -1e4 removes the overflow
        # without changing any output value: exp(-1e4/tau) is 0.0 either way.
        # This keeps tau=1.0 bit-identical to SoftmaxOperator.
        x = (x - x.amax(dim=-1, keepdim=True)).clamp_min(self._FLOOR)
        probs = F.softmax(x / self.tau, dim=-1)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            probs = probs * head_mask
        return probs.to(scores.dtype)
# --------------------------------------------------------------------------- #
# E4 — is the key-similarity GRAPH doing the work?
# --------------------------------------------------------------------------- #
class StructuredPOperator(QGFDOperator):
    """
    QGFD with the key-similarity graph replaced by a structure-free control.

    `p_structure`:
      "real"     — P = softmax(KK^T/temp), i.e. unmodified QGFD.
      "uniform"  — every row uniform over its causal prefix. Row-stochastic,
                   causal, and it moves exactly as much mass as the real P
                   (the alpha bound is identical), but it encodes no similarity.
      "shuffled" — the real P's row values randomly permuted WITHIN the causal
                   prefix. Preserves each row's entire multiset of probabilities
                   — its entropy, its max, its sparsity — and destroys only
                   *which key* receives which mass.

    "shuffled" is the sharper control of the two: it holds every scalar summary
    of P fixed and varies only the assignment. If QGFD's gain survives it, the
    gain is not coming from the key graph.
    """

    def __init__(self, *args, p_structure: str = "real", shuffle_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        assert p_structure in ("real", "uniform", "shuffled"), p_structure
        self.p_structure = p_structure
        self.shuffle_seed = int(shuffle_seed)

    def build_transition_from_keys(self, K, target_heads=None, is_causal=None, dtype=None):
        if self.p_structure == "real":
            return super().build_transition_from_keys(K, target_heads, is_causal, dtype)
        if is_causal is None:
            is_causal = getattr(self, "is_causal", False)
        if self.p_structure == "uniform":
            return self._uniform_P(K, target_heads, is_causal, dtype)
        P = super().build_transition_from_keys(K, target_heads, is_causal, dtype)
        return self._shuffle_rows(P, is_causal)
    def _uniform_P(self, K, target_heads, is_causal, dtype):
        """Row-uniform over the causal prefix, with QGFD's own post-processing."""
        out_dtype = dtype if dtype is not None else K.dtype
        B, H_k, Lk, _ = K.shape
        if is_causal:
            allow = torch.tril(torch.ones((Lk, Lk), device=K.device, dtype=torch.bool))
        else:
            allow = torch.ones((Lk, Lk), device=K.device, dtype=torch.bool)
        P = allow.to(torch.float32)
        P = P / P.sum(-1, keepdim=True)
        P = P.to(out_dtype).expand(B, H_k, Lk, Lk).clone()

        # Same position-0 convention and same jitter as the real P, so that any
        # difference between the arms cannot be attributed to either of them.
        if Lk > 1:
            P[:, :, 0, :] = 0.0
            P[:, :, 0, 0] = 1.0
        jitter = self._eps(P)
        P = P * (1.0 - jitter) + (jitter / P.size(-1))

        if target_heads is not None and H_k != target_heads:
            P = torch.repeat_interleave(P, target_heads // H_k, dim=1)
        return P.detach() if self.detach_P else P

    def _shuffle_rows(self, P, is_causal):
        """
        Permute each row's values within its causal prefix.

        Trick: draw uniform noise, push the disallowed (future) columns to the
        end of the sort order, then argsort. For row i the first i+1 entries of
        the resulting index vector are exactly a random permutation of
        {0..i}, so gathering with it permutes the causal block in place and
        leaves the near-zero future entries where they are. Row sums are
        preserved exactly — this is a permutation, not a reweighting.
        """
        Lk = P.size(-1)
        gen = torch.Generator(device="cpu").manual_seed(self.shuffle_seed)
        noise = torch.rand(P.shape, generator=gen).to(P.device)
        if is_causal and Lk > 1:
            allow = torch.tril(torch.ones((Lk, Lk), device=P.device, dtype=torch.bool))
            noise = noise.masked_fill(~allow[None, None, :, :], 2.0)
        idx = noise.argsort(dim=-1)
        return torch.gather(P, -1, idx)
# --------------------------------------------------------------------------- #
# E6 — corruption types: input noise vs BPE fragmentation
# --------------------------------------------------------------------------- #
# `perturb_text` (imported) edits CHARACTERS, which re-tokenises the word:
# "attention" (1 token) becomes "attentiln" (3-4 subword fragments). So a QGFD
# win under character noise is consistent with two different stories —
# "diffusion recovers from corrupted keys" and "diffusion re-assembles
# fragmented subwords" — and only the first is the paper's claim. The two
# corruptions below leave every surviving word's tokenisation intact.
def perturb_word_drop(text: str, rate: float, rng: random.Random) -> str:
    """Delete whole whitespace words at `rate`. Surviving words tokenise as before."""
    if rate <= 0:
        return text
    words = text.split(" ")
    kept = [w for w in words if not (w.strip() and rng.random() < rate)]
    return " ".join(kept) if kept else text


def perturb_word_swap(text: str, rate: float, rng: random.Random) -> str:
    """
    Swap adjacent word pairs at `rate`. Destroys local order — the thing
    positional information encodes — while preserving the exact token multiset.
    """
    if rate <= 0:
        return text
    words = text.split(" ")
    i = 0
    while i < len(words) - 1:
        if words[i].strip() and words[i + 1].strip() and rng.random() < rate:
            words[i], words[i + 1] = words[i + 1], words[i]
            i += 2                       # don't re-swap a word we just moved
        else:
            i += 1
    return " ".join(words)


CORRUPTIONS = {
    "char": perturb_text,               # the headline corruption; changes tokenisation
    "word_drop": perturb_word_drop,     # information removed, tokenisation intact
    "word_swap": perturb_word_swap,     # order destroyed, token multiset intact
}
# --------------------------------------------------------------------------- #
# Shared machinery
# --------------------------------------------------------------------------- #
def fixed_texts(cfg: ExperimentConfig, n: Optional[int] = None) -> List[str]:
    """
    A FIXED WikiText sample, identical for every arm and every alpha in this
    module. The mechanism experiments are not multi-seed corpus estimates; they
    are within-sample comparisons, and freezing the corpus removes the largest
    variance component from the comparison entirely.
    """
    return load_wikitext(n if n is not None else cfg.robustness_num_texts)


def precompute_corruptions(texts: List[str], rates: Sequence[float],
                           corruption: str, seed: int) -> Dict[float, List[str]]:
    """
    Corrupt ONCE per rate, then reuse across arms/alphas.

    This is the pairing that matters: every arm scores the byte-identical
    corrupted string, so the comparison has no corruption-draw variance at all.
    """
    fn = CORRUPTIONS[corruption]
    out = {}
    for rate in rates:
        rng = random.Random(seed)          # same draw for every rate's own arm set
        out[float(rate)] = [fn(t, float(rate), rng) for t in texts]
    return out


@torch.no_grad()
def operator_entropy(model, tok, texts: List[str], device: str,
                     cfg: ExperimentConfig) -> Dict[str, float]:
    """
    Mean attention entropy (nats) and position-0 mass, hooking ANY
    `AttentionProbabilityOperator`.

    `review_experiments.attention_stats` hooks QGFDOperator and SoftmaxOperator
    by type, so it silently measures nothing for the temperature and
    structured-P arms. This version dispatches on the base class instead, which
    is what makes the entropy MATCHING in E3 possible.
    """
    captured: List[torch.Tensor] = []
    handles = []
    seen = set()
    for m in model.modules():
        if isinstance(m, AttentionProbabilityOperator) and id(m) not in seen:
            seen.add(id(m))
            handles.append(m.register_forward_hook(
                lambda _mod, _i, out: captured.append(out.detach().float())))
    if not handles:
        raise RuntimeError("no AttentionProbabilityOperator found — model is unpatched")

    ent_sum, sink_sum, count = 0.0, 0.0, 0
    try:
        for t in texts:
            captured.clear()
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=cfg.attn_seq_len).input_ids.to(device)
            if ids.size(1) < 4:
                continue
            model(ids)
            for p in captured:                        # (B, H, Lq, Lk)
                p = p.clamp_min(1e-12)
                ent_sum += (-(p * p.log()).sum(-1)).mean().item()
                sink_sum += p[..., 0].mean().item()
                count += 1
    finally:
        for h in handles:
            h.remove()
    if count == 0:
        raise RuntimeError("no text was long enough to measure attention entropy")
    return {"mean_attention_entropy_nats": ent_sum / count,
            "mean_sink_mass_pos0": sink_sum / count}
def build_patched(operator, cfg: ExperimentConfig, label: str):
    """Load a fresh checkpoint, install `operator`, verify it is really live."""
    tok, model, device = load_model_and_tokenizer(cfg)
    model = wrap_model_with_qgfd_operator(model, operator, verbose=False)
    model.eval()
    verify_patch(model, tok, device, label)
    return tok, model, device


def _ppl(model, tok, texts: List[str], device: str, cfg: ExperimentConfig) -> float:
    return compute_perplexity(model, tok, texts, device,
                              max_length=cfg.ppl_max_length, stride=cfg.ppl_stride)


def _pct_delta(noisy: float, clean: float) -> float:
    return 100.0 * (noisy - clean) / clean


def fit_power_law(alphas: Sequence[float], deltas: Sequence[float]) -> Dict:
    """
    Least-squares fit of log|delta| = log c + k log alpha over alpha > 0.

    Returns the exponent k with its standard error and R^2. The curvature
    argument predicts k ~= 2: at a stationary point the gradient term vanishes
    and the leading cost is (1/2) d^T H d with ||d|| = O(alpha). k ~= 1 would
    mean the checkpoint is far enough from a WikiText-2 optimum that the linear
    term dominates — a real finding, and one that weakens the "safe drop-in"
    framing because the cost would then fall off much more slowly in alpha.
    """
    pts = [(math.log(a), math.log(abs(d))) for a, d in zip(alphas, deltas)
           if a > 0 and abs(d) > 1e-12]
    if len(pts) < 3:
        return {"exponent_k": None, "n_points": len(pts),
                "note": "need >= 3 non-zero alphas to fit an exponent"}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    n = len(pts)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    c = my - k * mx
    resid = [y - (c + k * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in resid)
    sst = sum((y - my) ** 2 for y in ys)
    se_k = math.sqrt(sse / (n - 2) / sxx) if n > 2 and sse > 0 else 0.0
    return {"exponent_k": k, "exponent_se": se_k, "log_c": c,
            "r2": (1.0 - sse / sst) if sst > 0 else 1.0, "n_points": n,
            "predicted_k": 2.0,
            "consistent_with_quadratic": bool(se_k > 0 and abs(k - 2.0) <= 2.0 * se_k)}
# --------------------------------------------------------------------------- #
# E2 — alpha x noise sweep, and the curvature exponent
# --------------------------------------------------------------------------- #
DEFAULT_ALPHAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10)


def alpha_noise_sweep(cfg: ExperimentConfig,
                      alphas: Sequence[float] = DEFAULT_ALPHAS,
                      noise_rates: Sequence[float] = (0.0, 0.15),
                      corruption: str = "char",
                      diffusion_steps: Optional[Sequence[int]] = None) -> Dict:
    """
    The money figure: perplexity as a joint function of alpha and noise.

    One model instance is loaded and `target_alpha` is mutated between
    evaluations. In eval mode `get_alpha()` returns `target_alpha` clamped to
    `max_alpha`, so mutation is sufficient and no reload is needed -- but the
    default clamp is 0.10, which would silently flatten the top of the sweep, so
    `max_alpha` is raised to cover whatever alphas are requested.

    alpha = 0.0 is the softmax baseline. That substitution is safe because the
    equivalence is verified bit-exact elsewhere, not assumed.
    """
    steps_list = list(diffusion_steps) if diffusion_steps else [cfg.diffusion_steps]
    alphas = [float(a) for a in alphas]
    if 0.0 not in alphas:
        raise ValueError("alpha=0.0 must be in `alphas`: it is the softmax baseline "
                         "every delta in this sweep is measured against")
    alphas = [0.0] + [a for a in alphas if a != 0.0]     # baseline evaluated first
    rates = [float(r) for r in noise_rates]

    texts = fixed_texts(cfg)
    corrupted = precompute_corruptions(texts, rates, corruption, cfg.robustness_seed)

    op = QGFDOperator(
        diffusion_steps=steps_list[0], target_alpha=alphas[0], mode="full",
        max_full_seq_len=cfg.max_full_seq_len, full_fallback_mode=cfg.full_fallback_mode,
        detach_P=cfg.detach_P, is_causal=True,
        max_alpha=max(1.0, max(abs(a) for a in alphas) * 1.5),
    )
    tok, model, device = build_patched(op, cfg, "qgfd/alpha-sweep")

    grid, curves = [], {}
    try:
        for T in steps_list:
            op.diffusion_steps = int(T)
            for rate in rates:
                base = None
                for a in alphas:
                    op.target_alpha = float(a)
                    ppl = _ppl(model, tok, corrupted[rate], device, cfg)
                    if a == 0.0:
                        base = ppl
                    grid.append({"T": int(T), "noise_rate": rate, "alpha": a,
                                 "ppl": ppl,
                                 "delta_pct_vs_alpha0": (_pct_delta(ppl, base)
                                                         if base else None)})
                    print(f"    T={T} noise={rate:.0%} alpha={a:<6.3f} ppl={ppl:9.4f}"
                          + (f"  ({_pct_delta(ppl, base):+.3f}% vs alpha=0)" if base else ""))
    finally:
        free(model)
    # Curvature fit on the CLEAN curve — that is where the argument applies.
    clean_rate = 0.0 if 0.0 in rates else rates[0]
    for T in steps_list:
        rows = [g for g in grid if g["T"] == T and g["noise_rate"] == clean_rate]
        rows.sort(key=lambda g: g["alpha"])
        curves[f"T{T}"] = {
            "alphas": [g["alpha"] for g in rows],
            "delta_pct": [g["delta_pct_vs_alpha0"] for g in rows],
            "fit": fit_power_law([g["alpha"] for g in rows],
                                 [g["delta_pct_vs_alpha0"] or 0.0 for g in rows]),
        }

    # The robustness read: at each alpha, how much SMALLER is the degradation
    # under noise than the alpha=0 arm's degradation? Positive = QGFD wins.
    noisy_rate = max(rates)
    gaps = {}
    if noisy_rate > 0:
        for T in steps_list:
            g0 = {g["alpha"]: g["ppl"] for g in grid
                  if g["T"] == T and g["noise_rate"] == 0.0}
            gn = {g["alpha"]: g["ppl"] for g in grid
                  if g["T"] == T and g["noise_rate"] == noisy_rate}
            if 0.0 in g0 and 0.0 in gn:
                base_deg = _pct_delta(gn[0.0], g0[0.0])
                gaps[f"T{T}"] = {
                    "noise_rate": noisy_rate,
                    "baseline_degradation_pct": base_deg,
                    "by_alpha": {f"{a:.4f}": {
                        "degradation_pct": _pct_delta(gn[a], g0[a]),
                        "robustness_gap_pp": base_deg - _pct_delta(gn[a], g0[a]),
                    } for a in sorted(g0) if a in gn},
                }

    return {
        "experiment": "E2_alpha_noise_sweep",
        "meta": {"model_id": cfg.model_id, "corruption": corruption,
                 "n_texts": len(texts), "alphas": alphas, "noise_rates": rates,
                 "diffusion_steps": steps_list, "detach_P": cfg.detach_P,
                 "robustness_seed": cfg.robustness_seed, "device": device,
                 "paired": "identical fixed text sample and identical corrupted "
                           "strings across every alpha; no seed variance",
                 "note": "alpha=0 is the softmax baseline (bit-exact equivalence "
                         "verified separately in E1)"},
        "grid": grid,
        "clean_curvature": curves,
        "robustness_by_alpha": gaps,
    }
# --------------------------------------------------------------------------- #
# E3 — the experiment most likely to kill the paper
# --------------------------------------------------------------------------- #
def match_temperature(model, tok, texts: List[str], device: str, cfg: ExperimentConfig,
                      operator: TemperatureOperator, target_entropy: float,
                      lo: float = 0.2, hi: float = 10.0,
                      tol: float = 1e-3, max_iter: int = 18) -> Dict:
    """
    Bisect tau so that mean attention entropy matches `target_entropy`.

    Entropy is monotone increasing in tau, so bisection is valid. It is run on
    log(tau) because the useful range is multiplicative. The search MEASURES
    entropy rather than assuming a direction, which matters: diffusion does not
    always raise entropy. A peaked or sink-dominated P concentrates mass, and in
    that regime the matched tau lands BELOW 1. Encoding "flatter" as an
    assumption here would have hidden that case.
    """
    llo, lhi = math.log(lo), math.log(hi)
    trace = []
    best = None
    for _ in range(max_iter):
        lmid = 0.5 * (llo + lhi)
        operator.tau = math.exp(lmid)
        ent = operator_entropy(model, tok, texts, device, cfg)["mean_attention_entropy_nats"]
        err = ent - target_entropy
        trace.append({"tau": operator.tau, "entropy": ent, "err": err})
        if best is None or abs(err) < abs(best["err"]):
            best = trace[-1]
        if abs(err) <= tol:
            break
        if err < 0:
            llo = lmid          # too sharp -> need a larger tau
        else:
            lhi = lmid
    operator.tau = best["tau"]
    bracketed = min(t["entropy"] for t in trace) <= target_entropy <= max(t["entropy"] for t in trace)
    return {"tau": best["tau"], "achieved_entropy": best["entropy"],
            "target_entropy": target_entropy, "abs_err": abs(best["err"]),
            "converged": abs(best["err"]) <= tol, "bracketed": bool(bracketed),
            "n_iter": len(trace), "search_range": [lo, hi]}
def entropy_matched_control(cfg: ExperimentConfig,
                            alphas: Sequence[float] = (0.02, 0.05),
                            noise_rate: float = 0.15,
                            corruption: str = "char") -> Dict:
    """
    Three arms on byte-identical text: softmax (tau=1), QGFD at each alpha, and
    temperature scaling tuned to QGFD's OWN measured attention entropy.

    How to read the result:
      qgfd_gap_pp > temp_gap_pp   ->  QGFD buys robustness that entropy change
                                      alone does not. The contribution survives.
      qgfd_gap_pp ~= temp_gap_pp  ->  a one-line, zero-cost change reproduces the
                                      effect. The mechanism claim is dead even if
                                      the robustness number is real.
      temp_gap_pp > qgfd_gap_pp   ->  worse than dead: the free control wins.

    Only tau=1.0 and the matched tau differ between the softmax and temperature
    arms, so nothing else can explain a difference between them.
    """
    rates = [0.0, float(noise_rate)]
    texts = fixed_texts(cfg)
    corrupted = precompute_corruptions(texts, rates, corruption, cfg.robustness_seed)
    ent_texts = texts[:cfg.attn_num_texts]

    # --- QGFD arms: one load, alpha mutated ------------------------------- #
    qgfd_arms = {}
    op = QGFDOperator(diffusion_steps=cfg.diffusion_steps, target_alpha=float(alphas[0]),
                      mode="full", max_full_seq_len=cfg.max_full_seq_len,
                      full_fallback_mode=cfg.full_fallback_mode, detach_P=cfg.detach_P,
                      is_causal=True, max_alpha=max(1.0, max(alphas) * 1.5))
    tok, model, device = build_patched(op, cfg, "qgfd/entropy-match")
    try:
        for a in alphas:
            op.target_alpha = float(a)
            stats = operator_entropy(model, tok, ent_texts, device, cfg)
            clean = _ppl(model, tok, corrupted[0.0], device, cfg)
            noisy = _ppl(model, tok, corrupted[rates[1]], device, cfg)
            qgfd_arms[f"{a:.4f}"] = {
                "alpha": float(a), "clean_ppl": clean, "noisy_ppl": noisy,
                "degradation_pct": _pct_delta(noisy, clean), **stats}
            print(f"    qgfd  alpha={a:<6.3f} H={stats['mean_attention_entropy_nats']:.4f} "
                  f"clean={clean:8.4f} noisy={noisy:8.4f} "
                  f"deg={_pct_delta(noisy, clean):+.3f}%")
    finally:
        free(model)
    # --- softmax + matched-temperature arms: one load, tau mutated -------- #
    top = TemperatureOperator(tau=1.0)
    tok, model, device = build_patched(top, cfg, "temperature/entropy-match")
    temp_arms, paired = {}, {}
    try:
        top.tau = 1.0
        base_stats = operator_entropy(model, tok, ent_texts, device, cfg)
        base_clean = _ppl(model, tok, corrupted[0.0], device, cfg)
        base_noisy = _ppl(model, tok, corrupted[rates[1]], device, cfg)
        base_deg = _pct_delta(base_noisy, base_clean)
        softmax_arm = {"tau": 1.0, "clean_ppl": base_clean, "noisy_ppl": base_noisy,
                       "degradation_pct": base_deg, **base_stats}
        print(f"    softmax     tau=1.000 H={base_stats['mean_attention_entropy_nats']:.4f} "
              f"clean={base_clean:8.4f} noisy={base_noisy:8.4f} deg={base_deg:+.3f}%")

        for key, arm in qgfd_arms.items():
            fit = match_temperature(model, tok, ent_texts, device, cfg, top,
                                    arm["mean_attention_entropy_nats"])
            clean = _ppl(model, tok, corrupted[0.0], device, cfg)
            noisy = _ppl(model, tok, corrupted[rates[1]], device, cfg)
            deg = _pct_delta(noisy, clean)
            temp_arms[key] = {"clean_ppl": clean, "noisy_ppl": noisy,
                              "degradation_pct": deg, "match": fit}
            qgap = base_deg - arm["degradation_pct"]
            tgap = base_deg - deg
            paired[key] = {
                "alpha": arm["alpha"], "matched_tau": fit["tau"],
                "entropy_match_abs_err": fit["abs_err"],
                "qgfd_gap_pp": qgap, "temp_gap_pp": tgap,
                "qgfd_minus_temp_pp": qgap - tgap,
                "qgfd_clean_cost_pct": _pct_delta(arm["clean_ppl"], base_clean),
                "temp_clean_cost_pct": _pct_delta(clean, base_clean),
                "verdict": ("qgfd_beats_free_control" if qgap - tgap > 0.0
                            else "free_control_matches_or_wins"),
            }
            print(f"    temp  tau={fit['tau']:<6.3f} (H err {fit['abs_err']:.1e}) "
                  f"clean={clean:8.4f} noisy={noisy:8.4f} deg={deg:+.3f}%  "
                  f"| qgfd_gap={qgap:+.3f}pp temp_gap={tgap:+.3f}pp "
                  f"-> {paired[key]['verdict']}")
    finally:
        free(model)

    return {"experiment": "E3_entropy_matched_temperature",
            "meta": {"model_id": cfg.model_id, "corruption": corruption,
                     "noise_rate": float(noise_rate), "n_texts": len(texts),
                     "n_entropy_texts": len(ent_texts), "device": device,
                     "diffusion_steps": cfg.diffusion_steps,
                     "note": "temperature is foldable into W_Q, so it is free at "
                             "inference; if it matches QGFD the mechanism claim fails"},
            "softmax": softmax_arm, "qgfd": qgfd_arms, "temperature": temp_arms,
            "paired": paired}
# --------------------------------------------------------------------------- #
# E4 — is the key-similarity graph load-bearing?
# --------------------------------------------------------------------------- #
def p_structure_ablation(cfg: ExperimentConfig,
                         alpha: Optional[float] = None,
                         noise_rate: float = 0.15,
                         corruption: str = "char",
                         structures: Sequence[str] = ("real", "uniform", "shuffled"),
                         shuffle_seed: int = 0) -> Dict:
    """
    Same alpha, same T, same amount of mass moved — only the destination changes.

    The baseline is alpha=0 on this very operator, so all four arms share one
    loaded checkpoint and one set of corrupted strings. `p_structure` is read
    inside `build_transition_from_keys` on every forward pass, so switching
    arms is an attribute assignment.

    If "uniform" or "shuffled" reproduces "real", the paper's name is wrong:
    the effect is generic probability-mass smoothing, not graph diffusion, and
    it should be reported as such.
    """
    a = float(cfg.target_alpha if alpha is None else alpha)
    rates = [0.0, float(noise_rate)]
    texts = fixed_texts(cfg)
    corrupted = precompute_corruptions(texts, rates, corruption, cfg.robustness_seed)

    op = StructuredPOperator(
        diffusion_steps=cfg.diffusion_steps, target_alpha=a, mode="full",
        max_full_seq_len=cfg.max_full_seq_len, full_fallback_mode=cfg.full_fallback_mode,
        detach_P=cfg.detach_P, is_causal=True, max_alpha=max(1.0, a * 1.5),
        p_structure="real", shuffle_seed=shuffle_seed,
    )
    tok, model, device = build_patched(op, cfg, "qgfd/P-structure")

    arms = {}
    try:
        op.target_alpha = 0.0
        clean0 = _ppl(model, tok, corrupted[0.0], device, cfg)
        noisy0 = _ppl(model, tok, corrupted[rates[1]], device, cfg)
        base_deg = _pct_delta(noisy0, clean0)
        arms["softmax_alpha0"] = {"clean_ppl": clean0, "noisy_ppl": noisy0,
                                  "degradation_pct": base_deg}
        print(f"    alpha=0 (softmax)   clean={clean0:8.4f} noisy={noisy0:8.4f} "
              f"deg={base_deg:+.3f}%")

        op.target_alpha = a
        for s in structures:
            op.p_structure = s
            stats = operator_entropy(model, tok, texts[:cfg.attn_num_texts], device, cfg)
            clean = _ppl(model, tok, corrupted[0.0], device, cfg)
            noisy = _ppl(model, tok, corrupted[rates[1]], device, cfg)
            deg = _pct_delta(noisy, clean)
            arms[s] = {"clean_ppl": clean, "noisy_ppl": noisy, "degradation_pct": deg,
                       "robustness_gap_pp": base_deg - deg,
                       "clean_cost_pct": _pct_delta(clean, clean0), **stats}
            print(f"    P={s:<9} clean={clean:8.4f} noisy={noisy:8.4f} "
                  f"deg={deg:+.3f}%  gap={base_deg - deg:+.3f}pp  "
                  f"H={stats['mean_attention_entropy_nats']:.4f}")
    finally:
        free(model)
    verdict = None
    if "real" in arms:
        real_gap = arms["real"]["robustness_gap_pp"]
        controls = {s: arms[s]["robustness_gap_pp"] for s in structures if s != "real"}
        if controls:
            best_ctrl = max(controls, key=controls.get)
            margin = real_gap - controls[best_ctrl]
            verdict = {
                "real_gap_pp": real_gap,
                "best_control": best_ctrl,
                "best_control_gap_pp": controls[best_ctrl],
                "margin_pp": margin,
                "reading": ("key-similarity structure is load-bearing" if margin > 0
                            else "structure-free P matches or beats the real P — the "
                                 "effect is generic smoothing, not graph diffusion"),
            }

    return {"experiment": "E4_p_structure_ablation",
            "meta": {"model_id": cfg.model_id, "alpha": a,
                     "diffusion_steps": cfg.diffusion_steps, "corruption": corruption,
                     "noise_rate": float(noise_rate), "n_texts": len(texts),
                     "shuffle_seed": shuffle_seed, "device": device,
                     "note": "all arms share one checkpoint, one alpha, one T and "
                             "byte-identical corrupted text; only P's structure varies"},
            "arms": arms, "verdict": verdict}
# --------------------------------------------------------------------------- #
# E6 — noise robustness, or subword-fragmentation repair?
# --------------------------------------------------------------------------- #
def corruption_type_sweep(cfg: ExperimentConfig,
                          alpha: Optional[float] = None,
                          rate: float = 0.15,
                          corruptions: Sequence[str] = ("char", "word_drop", "word_swap")
                          ) -> Dict:
    """
    Run the same alpha against corruptions that do and do not disturb tokenisation.

    Character noise re-tokenises: one word becomes several subword fragments, and
    their key vectors are near-neighbours of each other. Diffusion could be
    re-assembling those fragments rather than doing anything about noise as such.
    Word deletion and adjacent-word swaps leave every surviving word's token ids
    untouched, so they separate the two explanations.

    A gap that appears ONLY under "char" is a narrower, more specific claim than
    the paper currently makes -- still publishable, but it must be stated as
    robustness to sub-word fragmentation, not to input noise.

    Also reports token counts: if word_drop shortens the text far more than char
    noise lengthens it, part of any perplexity difference is sequence-length
    effect rather than corruption, and the counts make that visible.
    """
    a = float(cfg.target_alpha if alpha is None else alpha)
    texts = fixed_texts(cfg)

    op = QGFDOperator(diffusion_steps=cfg.diffusion_steps, target_alpha=a, mode="full",
                      max_full_seq_len=cfg.max_full_seq_len,
                      full_fallback_mode=cfg.full_fallback_mode, detach_P=cfg.detach_P,
                      is_causal=True, max_alpha=max(1.0, a * 1.5))
    tok, model, device = build_patched(op, cfg, "qgfd/corruption-types")

    out = {}
    try:
        for name in corruptions:
            variants = precompute_corruptions(texts, [0.0, float(rate)], name,
                                              cfg.robustness_seed)
            n_clean = len(tok("\n\n".join(variants[0.0])).input_ids)
            n_noisy = len(tok("\n\n".join(variants[float(rate)])).input_ids)
            row = {}
            for arm_alpha, label in ((0.0, "softmax"), (a, "qgfd")):
                op.target_alpha = arm_alpha
                clean = _ppl(model, tok, variants[0.0], device, cfg)
                noisy = _ppl(model, tok, variants[float(rate)], device, cfg)
                row[label] = {"clean_ppl": clean, "noisy_ppl": noisy,
                              "degradation_pct": _pct_delta(noisy, clean)}
            gap = row["softmax"]["degradation_pct"] - row["qgfd"]["degradation_pct"]
            out[name] = {**row, "robustness_gap_pp": gap,
                         "n_tokens_clean": n_clean, "n_tokens_noisy": n_noisy,
                         "token_count_ratio": n_noisy / max(1, n_clean),
                         "changes_tokenisation": name == "char"}
            print(f"    {name:<10} softmax deg={row['softmax']['degradation_pct']:+8.3f}%  "
                  f"qgfd deg={row['qgfd']['degradation_pct']:+8.3f}%  "
                  f"gap={gap:+.3f}pp  tokens {n_clean}->{n_noisy}")
    finally:
        free(model)
    tokenising = [n for n, r in out.items() if r["changes_tokenisation"]]
    preserving = [n for n, r in out.items() if not r["changes_tokenisation"]]
    verdict = None
    if tokenising and preserving:
        gap_tok = max(out[n]["robustness_gap_pp"] for n in tokenising)
        gap_pres = max(out[n]["robustness_gap_pp"] for n in preserving)
        verdict = {
            "best_gap_tokenisation_changing_pp": gap_tok,
            "best_gap_tokenisation_preserving_pp": gap_pres,
            "reading": ("generalises beyond tokenisation damage — the broad "
                        "'input noise' claim is supported" if gap_pres > 0
                        else "gap appears only where tokenisation is disturbed — "
                             "claim must be narrowed to sub-word fragmentation"),
        }

    return {"experiment": "E6_corruption_types",
            "meta": {"model_id": cfg.model_id, "alpha": a, "rate": float(rate),
                     "diffusion_steps": cfg.diffusion_steps, "n_texts": len(texts),
                     "device": device,
                     "note": "alpha=0 is the softmax arm; both arms see byte-identical "
                             "corrupted text within each corruption type"},
            "by_corruption": out, "verdict": verdict}
# --------------------------------------------------------------------------- #
# E8 — spend the 1.6-1.9x on diffusion, or on parameters?
# --------------------------------------------------------------------------- #
def iso_compute_compare(cfg: ExperimentConfig,
                        small_model: str,
                        large_model: str,
                        noise_rate: float = 0.15,
                        corruption: str = "char",
                        alpha: Optional[float] = None) -> Dict:
    """
    QGFD on the small model vs plain softmax on the next size up.

    This is the question a reviewer will ask out loud, and the honest answer is
    probably "the bigger model". Reporting it anyway is what makes the rest of
    the paper credible: the contribution is a mechanism with a characterised
    cost, not a claim to a better accuracy-per-FLOP frontier.

    Latency is measured, not estimated, and it is measured against EAGER softmax
    on both models -- QGFD needs the explicit probability matrix, so it cannot
    use a fused kernel, and the fused-baseline gap is strictly larger than what
    this reports. Two arms are run on the small model so the small model's own
    QGFD overhead is visible next to the parameter-count comparison.
    """
    from scripts.review_experiments import benchmark_latency

    a = float(cfg.target_alpha if alpha is None else alpha)
    rates = [0.0, float(noise_rate)]
    texts = fixed_texts(cfg)
    corrupted = precompute_corruptions(texts, rates, corruption, cfg.robustness_seed)

    def measure(model_id: str, arm_alpha: float, label: str) -> Dict:
        sub = replace(cfg, model_id=model_id)
        op = QGFDOperator(diffusion_steps=cfg.diffusion_steps, target_alpha=arm_alpha,
                          mode="full", max_full_seq_len=cfg.max_full_seq_len,
                          full_fallback_mode=cfg.full_fallback_mode,
                          detach_P=cfg.detach_P, is_causal=True,
                          max_alpha=max(1.0, a * 1.5))
        tok, model, device = build_patched(op, sub, label)
        try:
            n_params = sum(p.numel() for p in model.parameters())
            clean = _ppl(model, tok, corrupted[0.0], device, sub)
            noisy = _ppl(model, tok, corrupted[rates[1]], device, sub)
            lat = benchmark_latency(model, tok, device, sub)
            row = {"model_id": model_id, "alpha": arm_alpha, "n_params": n_params,
                   "clean_ppl": clean, "noisy_ppl": noisy,
                   "degradation_pct": _pct_delta(noisy, clean), **lat}
        finally:
            free(model)
        print(f"    {label:<26} params={row['n_params']/1e6:7.1f}M "
              f"clean={clean:8.4f} noisy={noisy:8.4f} "
              f"deg={row['degradation_pct']:+.3f}% "
              f"prefill={lat['prefill_ms']:.1f}ms")
        return row
    arms = {
        "small_softmax": measure(small_model, 0.0, "small + softmax"),
        "small_qgfd": measure(small_model, a, "small + QGFD"),
        "large_softmax": measure(large_model, 0.0, "large + softmax"),
    }

    sq, ls = arms["small_qgfd"], arms["large_softmax"]
    ss = arms["small_softmax"]
    comparison = {
        "qgfd_overhead_x": sq["prefill_ms"] / ss["prefill_ms"],
        "large_vs_small_qgfd_latency_x": ls["prefill_ms"] / sq["prefill_ms"],
        "param_ratio_x": ls["n_params"] / ss["n_params"],
        "noisy_ppl_small_qgfd": sq["noisy_ppl"],
        "noisy_ppl_large_softmax": ls["noisy_ppl"],
        "winner_under_noise": ("small+QGFD" if sq["noisy_ppl"] < ls["noisy_ppl"]
                               else "large+softmax"),
        "latency_cheaper": ("small+QGFD" if sq["prefill_ms"] < ls["prefill_ms"]
                            else "large+softmax"),
    }
    comparison["reading"] = (
        "small+QGFD is on the useful side of the trade-off"
        if (comparison["winner_under_noise"] == "small+QGFD"
            and comparison["latency_cheaper"] == "small+QGFD")
        else "extra parameters buy more than diffusion does at this budget — report "
             "QGFD as a mechanism with a stated cost, not as a compute-efficiency win")
    print(f"    -> under noise: {comparison['winner_under_noise']}; "
          f"cheaper: {comparison['latency_cheaper']}")

    return {"experiment": "E8_iso_compute",
            "meta": {"small_model": small_model, "large_model": large_model,
                     "alpha": a, "diffusion_steps": cfg.diffusion_steps,
                     "corruption": corruption, "noise_rate": float(noise_rate),
                     "n_texts": len(texts), "latency_seq_len": cfg.latency_seq_len,
                     "latency_baseline": "eager materialised softmax, NOT FlashAttention; "
                                         "the fused-kernel gap is larger",
                     "note": "all three arms score byte-identical corrupted text"},
            "arms": arms, "comparison": comparison}
# --------------------------------------------------------------------------- #
# E7 — how far does the walk usefully reach?
# --------------------------------------------------------------------------- #
def diffusion_depth_sweep(cfg: ExperimentConfig,
                          steps: Sequence[int] = (1, 2, 3, 4),
                          alpha: Optional[float] = None,
                          quick: bool = False) -> Dict:
    """
    Sweep T on the retrieval probes at fixed alpha.

    What this tests, precisely. Theorem 3 expands p^(T) as
    (1-alpha) * sum_{k<T} alpha^k p0 P^k + alpha^T p0 P^T, so raising T adds
    longer paths through the key graph and nothing else; Theorem 5 says that far
    enough along, p^(T) approaches P's stationary distribution and query
    representations collapse. Together they predict a NON-MONOTONE curve: some
    optimal T > 1, then decay. Finding a monotone or flat curve instead
    falsifies the useful-reach story even though the algebra stays true.

    What this does NOT test: whether a single QGFD layer performs genuine
    multi-hop composition. That would need control over the key vectors
    themselves -- on a pretrained checkpoint the key graph is whatever the
    checkpoint learned, so an in-layer k-hop claim is not identifiable here.
    Establishing it needs a from-scratch synthetic-graph model, which is out of
    scope for this budget. The paper should say so rather than let a T-sweep
    stand in for it.

    Note also that alpha^T shrinks geometrically: at alpha=0.05 the T=4 term is
    6e-6 of the mass. Any observable T effect at small alpha therefore comes
    from the *sum*, and a flat curve past T=2 is the expected outcome, not a bug.
    """
    from scripts.eval_synthetic import (
        SyntheticConfig, apply_quick, eval_induction, eval_passkey,
    )

    a = float(cfg.target_alpha if alpha is None else alpha)
    scfg = SyntheticConfig(model_id=cfg.model_id, dtype=cfg.dtype, device=cfg.device,
                           target_alpha=a, max_full_seq_len=cfg.max_full_seq_len,
                           full_fallback_mode=cfg.full_fallback_mode,
                           detach_P=cfg.detach_P, seed=cfg.seed)
    if quick:
        scfg = apply_quick(scfg)

    op = QGFDOperator(diffusion_steps=1, target_alpha=a, mode="full",
                      max_full_seq_len=cfg.max_full_seq_len,
                      full_fallback_mode=cfg.full_fallback_mode, detach_P=cfg.detach_P,
                      is_causal=True, max_alpha=max(1.0, a * 1.5))
    tok, model, device = build_patched(op, cfg, "qgfd/depth-sweep")

    rows = {}
    try:
        for T, arm_alpha, label in ([(0, 0.0, "softmax")]
                                    + [(int(t), a, f"T{int(t)}") for t in steps]):
            op.diffusion_steps = max(1, T)
            op.target_alpha = arm_alpha
            ind = eval_induction(model, tok, device, scfg)
            psk = eval_passkey(model, tok, device, scfg)
            rows[label] = {"T": T, "alpha": arm_alpha,
                           "induction_acc": ind["acc"],
                           "induction_control_acc": ind["control_acc"],
                           "induction_by_noise": {k: v["acc"] for k, v in ind["by_noise"].items()},
                           "passkey_acc": psk["acc"],
                           "passkey_by_depth": {k: v["acc"] for k, v in psk["by_depth"].items()}}
            print(f"    {label:<8} induction={ind['acc']:.4f} "
                  f"(control {ind['control_acc']:.4f})  passkey={psk['acc']:.4f}")
    finally:
        free(model)
    qgfd_rows = {k: v for k, v in rows.items() if k != "softmax"}
    verdict = None
    if qgfd_rows and "softmax" in rows:
        best = max(qgfd_rows, key=lambda k: qgfd_rows[k]["induction_acc"])
        accs = [qgfd_rows[f"T{int(t)}"]["induction_acc"] for t in steps
                if f"T{int(t)}" in qgfd_rows]
        monotone_up = all(y >= x for x, y in zip(accs, accs[1:]))
        flat = (max(accs) - min(accs)) < 1e-9
        verdict = {
            "best_T": qgfd_rows[best]["T"],
            "best_induction_acc": qgfd_rows[best]["induction_acc"],
            "softmax_induction_acc": rows["softmax"]["induction_acc"],
            "control_acc_floor": rows["softmax"]["induction_control_acc"],
            "shape": "flat" if flat else ("monotone_increasing" if monotone_up
                                          else "non_monotone_or_decreasing"),
            "reading": ("identical across T — at this alpha the alpha^k terms are too "
                        "small to move argmax; not evidence against the theory, but "
                        "not evidence for useful reach either" if flat
                        else "T has a measurable effect; read best_T as the useful "
                             "reach of the walk at this alpha"),
        }
        if rows["softmax"]["induction_acc"] <= rows["softmax"]["induction_control_acc"] + 0.05:
            verdict["warning"] = ("induction accuracy is at the control floor — this "
                                  "model does not do induction on this probe, so no "
                                  "row here says anything about routing")

    return {"experiment": "E7_diffusion_depth",
            "meta": {"model_id": cfg.model_id, "alpha": a, "steps": list(steps),
                     "device": device, "quick": quick,
                     "alpha_pow_T": {f"T{int(t)}": a ** int(t) for t in steps},
                     "scope": "tests useful REACH of the walk, not in-layer k-hop "
                              "composition; the latter is not identifiable on a "
                              "pretrained checkpoint",
                     "coarse_metric_note": "exact-match accuracy rarely flips at small "
                                           "alpha; identical scores across T are expected"},
            "rows": rows, "verdict": verdict}
# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
QUICK_OVERRIDES = dict(
    model_id="JackFram/llama-160m", dtype="float32", device="cpu",
    ppl_num_texts=4, ppl_max_length=128, ppl_stride=128,
    robustness_num_texts=4, attn_num_texts=2, attn_seq_len=64,
    latency_seq_len=64, latency_iters=2, latency_warmup=1,
    max_full_seq_len=128,
)


def apply_quick(cfg: ExperimentConfig) -> ExperimentConfig:
    """CPU smoke sizing. Numbers from a quick run are plumbing checks, not results."""
    return replace(cfg, **QUICK_OVERRIDES)


# Experiment id -> (title, what a negative result kills)
MECHANISM_EXPERIMENTS = {
    "E2": ("alpha x noise sweep + curvature exponent",
           "the theory's account of the clean-PPL cost"),
    "E3": ("entropy-matched temperature control",
           "the whole robustness contribution"),
    "E4": ("P-structure ablation (uniform / shuffled)",
           "the word 'graph' in the method name"),
    "E6": ("corruption types (char / word_drop / word_swap)",
           "the breadth of the 'input noise' claim"),
    "E7": ("diffusion depth sweep on retrieval probes",
           "the useful-reach story for T > 1"),
    "E8": ("iso-compute vs a larger model",
           "any efficiency framing"),
}


def run_mechanism_suite(cfg: ExperimentConfig,
                        which: Sequence[str] = ("E2", "E3", "E4", "E6", "E7", "E8"),
                        alphas: Sequence[float] = DEFAULT_ALPHAS,
                        noise_rate: float = 0.15,
                        large_model: str = "HuggingFaceTB/SmolLM2-360M",
                        depth_steps: Sequence[int] = (1, 2, 3, 4),
                        quick: bool = False,
                        out_dir: Optional[str] = None) -> Dict:
    """
    Run the requested mechanism experiments and write one JSON per experiment
    plus a combined `mechanism_results.json`.

    Failures are recorded, not swallowed: an experiment that raises stores its
    exception under `errors` and the suite continues, so one missing checkpoint
    does not cost you the other five results. `errors` being non-empty is
    reported in the summary rather than left for the reader to notice.
    """
    out_dir = out_dir or os.path.join(cfg.out_dir, "mechanism")
    os.makedirs(out_dir, exist_ok=True)
    results, errors = {}, {}
    runners = {
        "E2": lambda: alpha_noise_sweep(cfg, alphas=alphas,
                                        noise_rates=(0.0, noise_rate)),
        "E3": lambda: entropy_matched_control(cfg, alphas=[a for a in alphas if a > 0][-2:],
                                              noise_rate=noise_rate),
        "E4": lambda: p_structure_ablation(cfg, noise_rate=noise_rate),
        "E6": lambda: corruption_type_sweep(cfg, rate=noise_rate),
        "E7": lambda: diffusion_depth_sweep(cfg, steps=depth_steps, quick=quick),
        "E8": lambda: iso_compute_compare(cfg, cfg.model_id, large_model,
                                          noise_rate=noise_rate),
    }

    for key in which:
        if key not in runners:
            raise ValueError(f"unknown experiment '{key}'; known: {sorted(runners)}")
        title, kills = MECHANISM_EXPERIMENTS[key]
        print(f"\n[{key}] {title}\n     a negative result invalidates: {kills}")
        try:
            res = runners[key]()
        except Exception as exc:                       # noqa: BLE001 — recorded below
            errors[key] = f"{type(exc).__name__}: {exc}"
            print(f"  !! {key} FAILED: {errors[key]}")
            continue
        results[key] = res
        with open(os.path.join(out_dir, f"{key}.json"), "w") as fh:
            json.dump(res, fh, indent=2)

    combined = {"config": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in cfg.__dict__.items()},
                "quick": quick, "requested": list(which),
                "results": results, "errors": errors}
    with open(os.path.join(out_dir, "mechanism_results.json"), "w") as fh:
        json.dump(combined, fh, indent=2)

    print(f"\n{'=' * 72}")
    print(f"mechanism suite: {len(results)}/{len(which)} completed -> {out_dir}")
    for key in which:
        if key in errors:
            print(f"  {key}  FAILED   {errors[key]}")
        else:
            print(f"  {key}  ok       {MECHANISM_EXPERIMENTS[key][0]}")
    if quick:
        print("  NOTE: --quick sizing. These are plumbing checks, not results.")
    print("=" * 72)
    return combined
def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_id", default=ExperimentConfig.model_id)
    ap.add_argument("--large_model", default="HuggingFaceTB/SmolLM2-360M",
                    help="E8 comparison model (the next size up)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--experiments", default="E2,E3,E4,E6,E7,E8",
                    help="comma-separated subset of " + ",".join(MECHANISM_EXPERIMENTS))
    ap.add_argument("--alphas", default=",".join(str(a) for a in DEFAULT_ALPHAS))
    ap.add_argument("--noise_rate", type=float, default=0.15)
    ap.add_argument("--diffusion_steps", type=int, default=1)
    ap.add_argument("--depth_steps", default="1,2,3,4")
    ap.add_argument("--robustness_num_texts", type=int, default=60)
    ap.add_argument("--out_dir", default="./qgfd_review_results")
    ap.add_argument("--quick", action="store_true",
                    help="CPU smoke sizing on a 160M model — plumbing only")
    args = ap.parse_args(argv)

    cfg = ExperimentConfig(model_id=args.model_id, dtype=args.dtype, device=args.device,
                           diffusion_steps=args.diffusion_steps,
                           robustness_num_texts=args.robustness_num_texts,
                           out_dir=args.out_dir)
    if args.quick:
        cfg = apply_quick(cfg)
        # E8 would otherwise pull a 360M checkpoint onto CPU for a plumbing check.
        if args.large_model == "HuggingFaceTB/SmolLM2-360M":
            args.large_model = "HuggingFaceTB/SmolLM2-135M"

    run_mechanism_suite(
        cfg,
        which=[s.strip() for s in args.experiments.split(",") if s.strip()],
        alphas=[float(s) for s in args.alphas.split(",") if s.strip()],
        noise_rate=args.noise_rate,
        large_model=args.large_model,
        depth_steps=[int(s) for s in args.depth_steps.split(",") if s.strip()],
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
