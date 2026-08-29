"""
QGFD Review Experiment Suite
============================
Self-contained, GPU-optional experiment harness for the Milestone-1 review.
Produces the numbers/plots for: perplexity, noise-robustness, attention
entropy / sink concentration, compute overhead, and qualitative generation.

NO TRAINING REQUIRED — every experiment here is zero-shot evaluation, so a full
sweep runs in minutes on a single Colab/Kaggle GPU (and on CPU with a tiny model).

Correctness notes (verified on CPU, llama-160m):
  * QGFD MUST be built with is_causal=True for a causal LM. With is_causal=False
    the key-graph diffusion p0 @ P spreads attention mass onto FUTURE keys,
    leaking future tokens into earlier positions (~4e-3 logit shift) and
    artificially deflating teacher-forced perplexity. build_operator() below
    forces is_causal=True for QGFD.
  * The baseline is wrapped with SoftmaxOperator (numerically identical to the
    stock model) so both arms expose an operator we can hook for attention stats.

Usage (see also QGFD_Review_Experiments.ipynb):
    from scripts.review_experiments import run_all, ExperimentConfig
    run_all(ExperimentConfig(model_id="meta-llama/Llama-3.2-1B"))
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict, replace
from typing import List, Dict, Optional, Sequence, Tuple

import torch

from torchdire import (
    QGFDOperator,
    SoftmaxOperator,
    wrap_model_with_qgfd_operator,
    collect_qgfd_operators,
)
from torchdire.nn.attention_adapters import (
    AttentionOperatorAdapter,
    GenericAttentionAdapter,
)


# --------------------------------------------------------------------------- #
# Supported models
# --------------------------------------------------------------------------- #
# QGFD is only genuinely wired up for the Llama-family adapters
# (LlamaAttentionAdapter and its Qwen2/Mistral subclasses). Verified on CPU:
#   llama-160m        -> LlamaAttentionAdapter on 12 layers, logits change. OK
#   gpt2              -> patched forward raises TypeError (adapter is a stub)
#   facebook/opt-125m -> patched forward raises TypeError (adapter is a stub)
#   gpt-neo-125m      -> 0 modules patched, QGFD silently has NO effect
# So pick a Llama/Mistral/Qwen2 model. These are all ungated (no HF token):
UNGATED_MODELS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "llama, 1.1B, 32 heads / 4 KV (real GQA) — headline choice",
    "HuggingFaceTB/SmolLM2-135M": "llama, 135M, fast",
    "JackFram/llama-160m": "llama, 160M, fastest (good for smoke tests)",
    "Qwen/Qwen2.5-0.5B": "qwen2, 0.5B, 14 heads / 2 KV",
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dtype: str = "bfloat16"           # "bfloat16" | "float16" | "float32"
    device: str = "auto"              # "auto" | "cuda" | "cpu"
    # QGFD hyper-parameters (validated defaults)
    diffusion_steps: int = 1
    target_alpha: float = 0.05
    max_full_seq_len: int = 512
    full_fallback_mode: str = "conv"
    # Whether the key-similarity transition matrix P carries gradient. True (the
    # default) treats P as a fixed routing structure; exposed here so the ablation
    # can vary it rather than pretending to.
    detach_P: bool = True
    # Perplexity eval
    ppl_num_texts: int = 200          # number of wikitext docs to concatenate
    ppl_max_length: int = 512
    ppl_stride: int = 512
    # Robustness sweep
    noise_rates: tuple = (0.0, 0.05, 0.10, 0.15)
    robustness_num_texts: int = 60
    robustness_seed: int = 42
    # Attention statistics
    attn_num_texts: int = 16
    attn_seq_len: int = 128
    # Latency benchmark
    latency_seq_len: int = 512
    latency_iters: int = 20
    latency_warmup: int = 3
    # Generation samples
    gen_prompts: tuple = (
        "Artificial Intelligence is transforming",
        "The most important discovery in physics was",
        "In the future, education will",
    )
    gen_max_new_tokens: int = 40
    # Output
    out_dir: str = "./qgfd_review_results"
    seed: int = 42
    # Multi-seed protocol (set by run_all_seeds; None = legacy deterministic slice).
    # When set, the WikiText pool is shuffled with this seed before taking the
    # first N paragraphs, so clean perplexity varies across seeds instead of
    # being a single deterministic point estimate.
    text_sample_seed: Optional[int] = None
    text_pool_size: int = 1200
# __APPEND_MARKER__


# --------------------------------------------------------------------------- #
# Model / operator setup
# --------------------------------------------------------------------------- #
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model_and_tokenizer(cfg: ExperimentConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(cfg.device)
    dtype = _DTYPES[cfg.dtype]
    if device == "cpu" and dtype is not torch.float32:
        dtype = torch.float32  # bf16/fp16 matmuls are unstable/slow on CPU

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tok, model, device


def build_operator(kind: str, cfg: ExperimentConfig):
    """kind='softmax' -> baseline; kind='qgfd' -> diffusion (is_causal forced True)."""
    if kind == "softmax":
        return SoftmaxOperator()
    if kind == "qgfd":
        return QGFDOperator(
            diffusion_steps=cfg.diffusion_steps,
            target_alpha=cfg.target_alpha,
            mode="full",
            max_full_seq_len=cfg.max_full_seq_len,
            full_fallback_mode=cfg.full_fallback_mode,
            detach_P=cfg.detach_P,
            is_causal=True,        # REQUIRED for causal LMs — see module docstring
        )
    raise ValueError(f"unknown operator kind: {kind}")


def make_model(kind: str, cfg: ExperimentConfig):
    """Fresh model patched with the requested operator, in eval mode."""
    tok, model, device = load_model_and_tokenizer(cfg)
    operator = build_operator(kind, cfg)
    model = wrap_model_with_qgfd_operator(model, operator, verbose=False)
    model.eval()
    verify_patch(model, tok, device, kind)
    return tok, model, device


@torch.no_grad()
def verify_patch(model, tok, device: str, kind: str) -> int:
    """
    Fail loudly if the operator was not actually installed and executed.

    Without this check, unsupported architectures (GPT-2, OPT, GPT-Neo) either
    crash or patch ZERO modules — the run then silently reports plain-softmax
    numbers under the QGFD label. Better to error than to publish a no-op.
    """
    adapters = [m for m in model.modules() if isinstance(m, AttentionOperatorAdapter)]
    if not adapters:
        raise RuntimeError(
            f"[{kind}] No attention module was patched for '{model.config.model_type}'. "
            f"QGFD would have no effect. Use a Llama/Mistral/Qwen2 model — e.g. "
            f"{', '.join(list(UNGATED_MODELS)[:2])}."
        )
    n_generic = sum(1 for m in adapters if isinstance(m, GenericAttentionAdapter))
    if n_generic:
        raise RuntimeError(
            f"[{kind}] {n_generic}/{len(adapters)} layers fell back to "
            f"GenericAttentionAdapter, which does not reliably intercept softmax. "
            f"Use a Llama/Mistral/Qwen2 model instead."
        )

    # Confirm the operator is actually reached during a forward pass.
    # All layers share one operator instance, so hook each unique object once.
    calls = []
    seen, handles = set(), []
    for m in adapters:
        op = getattr(m, "prob_operator", None)
        if op is not None and id(op) not in seen:
            seen.add(id(op))
            handles.append(op.register_forward_hook(lambda *_: calls.append(1)))
    try:
        ids = tok("verification probe sentence", return_tensors="pt").input_ids.to(device)
        model(ids)
    finally:
        for h in handles:
            h.remove()
    if not calls:
        raise RuntimeError(f"[{kind}] Operator installed on {len(adapters)} layers but never "
                           f"invoked during forward — attention is bypassing it.")
    print(f"  [{kind}] patch verified: {len(adapters)} x {type(adapters[0]).__name__}, "
          f"operator invoked {len(calls)}x")
    return len(adapters)


def free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# __APPEND2__


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
# WikiText source candidates, tried in order.
#   * "Salesforce/wikitext" is the canonical parquet-backed mirror and is the only
#     one that works on datasets>=4.0, which dropped loading-script support.
#   * "wikitext" is the legacy script-based repo (older datasets only).
#   * "mikasenghaas/wikitext-2" is a plain-parquet backup if the first two fail.
WIKITEXT_SOURCES = (
    ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    ("wikitext", "wikitext-2-raw-v1"),
    ("mikasenghaas/wikitext-2", None),
)


def load_wikitext(num_texts: int, sources=WIKITEXT_SOURCES, split: str = "test") -> List[str]:
    """
    Return up to num_texts non-empty WikiText-2 paragraphs.

    Tries each (repo, config) candidate in order so the suite survives both
    datasets>=4 (no loading scripts) and Hub repo renames.
    """
    from datasets import load_dataset

    errors = []
    for repo, config in sources:
        try:
            ds = (load_dataset(repo, config, split=split) if config
                  else load_dataset(repo, split=split))
        except Exception as exc:                       # noqa: BLE001 - report all attempts
            errors.append(f"  {repo}" + (f" [{config}]" if config else "") + f" -> {type(exc).__name__}: {exc}")
            continue
        column = "text" if "text" in ds.column_names else ds.column_names[0]
        texts = [t for t in ds[column] if t and len(t.strip()) > 64]
        if texts:
            print(f"  (corpus: {repo}" + (f" / {config}" if config else "") + f", {len(texts)} usable paragraphs)")
            return texts[:num_texts]
        errors.append(f"  {repo}: loaded but no paragraph longer than 64 chars")

    raise RuntimeError(
        "Could not load WikiText-2 from any known source. Attempts:\n"
        + "\n".join(errors)
        + "\n\nFixes: (1) `pip install -U datasets`, (2) check network/HF access, "
          "or (3) pass your own corpus via load_wikitext(..., sources=[('<repo>', '<config>')])."
    )



# --------------------------------------------------------------------------- #
# Experiment 1 — Perplexity (sliding window, teacher-forced)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_perplexity(model, tok, texts: List[str], device: str,
                       max_length: int = 512, stride: int = 512) -> float:
    """Standard sliding-window perplexity over concatenated text."""
    enc = tok("\n\n".join(texts), return_tensors="pt").input_ids.to(device)
    n = enc.size(1)
    nll_sum, n_tokens = 0.0, 0
    prev_end = 0
    for begin in range(0, n, stride):
        end = min(begin + max_length, n)
        trg_len = end - prev_end
        ids = enc[:, begin:end]
        targets = ids.clone()
        targets[:, :-trg_len] = -100
        out = model(ids, labels=targets)
        # out.loss is mean over (trg_len) counted tokens
        num = (targets != -100).sum().item()
        nll_sum += out.loss.float().item() * num
        n_tokens += num
        prev_end = end
        if end == n:
            break
    return math.exp(nll_sum / max(1, n_tokens))
# __APPEND3__


# --------------------------------------------------------------------------- #
# Experiment 2 — Noise robustness
# --------------------------------------------------------------------------- #
_QWERTY = "qwertyuiopasdfghjklzxcvbnm"


def perturb_text(text: str, rate: float, rng: random.Random) -> str:
    """Character-level OCR/typo noise: swap, drop, or substitute chars at `rate`."""
    if rate <= 0:
        return text
    out = []
    for ch in text:
        if ch.isalpha() and rng.random() < rate:
            op = rng.choice(("sub", "drop", "dup"))
            if op == "sub":
                out.append(rng.choice(_QWERTY))
            elif op == "dup":
                out.append(ch); out.append(ch)
            # "drop": append nothing
        else:
            out.append(ch)
    return "".join(out)


def robustness_sweep(model, tok, texts: List[str], device: str, cfg: ExperimentConfig) -> Dict[float, float]:
    """Perplexity at each noise rate. Δ% vs clean is the robustness signal."""
    results = {}
    for rate in cfg.noise_rates:
        rng = random.Random(cfg.robustness_seed)
        noisy = [perturb_text(t, rate, rng) for t in texts]
        results[float(rate)] = compute_perplexity(
            model, tok, noisy, device,
            max_length=cfg.ppl_max_length, stride=cfg.ppl_stride,
        )
    return results
# __APPEND4__


# --------------------------------------------------------------------------- #
# Experiment 3 — Attention entropy & sink concentration (via operator hooks)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def attention_stats(model, tok, texts: List[str], device: str, cfg: ExperimentConfig) -> Dict[str, float]:
    """
    Hook every attention operator, capture its output probability tensor, and
    measure (a) mean attention entropy (nats) and (b) sink mass on position 0.
    Higher entropy + lower sink mass = less concentrated / less collapsed.
    """
    captured = []

    def hook(_module, _inp, out):
        captured.append(out.detach().float())

    handles = []
    for op in collect_qgfd_operators(model):
        handles.append(op.register_forward_hook(hook))
    # SoftmaxOperator is not a QGFDOperator; hook it explicitly if present.
    from torchdire import SoftmaxOperator as _Softmax
    for m in model.modules():
        if isinstance(m, _Softmax):
            handles.append(m.register_forward_hook(hook))

    ent_sum, sink_sum, count = 0.0, 0.0, 0
    try:
        for t in texts:
            captured.clear()
            ids = tok(t, return_tensors="pt", truncation=True,
                      max_length=cfg.attn_seq_len).input_ids.to(device)
            if ids.size(1) < 4:
                continue
            model(ids)
            for p in captured:                       # (B, H, Lq, Lk)
                p = p.clamp_min(1e-12)
                ent = -(p * p.log()).sum(-1)          # (B, H, Lq)
                ent_sum += ent.mean().item()
                sink_sum += p[..., 0].mean().item()
                count += 1
    finally:
        for h in handles:
            h.remove()

    count = max(1, count)
    return {"mean_attention_entropy_nats": ent_sum / count,
            "mean_sink_mass_pos0": sink_sum / count}
# __APPEND5__


# --------------------------------------------------------------------------- #
# Experiment 4 — Compute overhead (prefill latency)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def benchmark_latency(model, tok, device: str, cfg: ExperimentConfig) -> Dict[str, float]:
    vocab = getattr(model.config, "vocab_size", 32000)
    ids = torch.randint(0, vocab, (1, cfg.latency_seq_len), device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(cfg.latency_warmup):
        model(ids)
    sync()
    t0 = time.perf_counter()
    for _ in range(cfg.latency_iters):
        model(ids)
    sync()
    total = time.perf_counter() - t0
    per_pass = total / cfg.latency_iters
    return {"prefill_ms": per_pass * 1e3,
            "tokens_per_s": cfg.latency_seq_len / per_pass}


# --------------------------------------------------------------------------- #
# Experiment 5 — Qualitative generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_samples(model, tok, device: str, cfg: ExperimentConfig) -> List[Dict[str, str]]:
    out = []
    for prompt in cfg.gen_prompts:
        ids = tok(prompt, return_tensors="pt").to(device)
        gen = model.generate(**ids, max_new_tokens=cfg.gen_max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        out.append({"prompt": prompt,
                    "completion": tok.decode(gen[0], skip_special_tokens=True)})
    return out
# __APPEND6__


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _run_arm(kind: str, cfg: ExperimentConfig, texts: List[str]) -> Dict:
    from transformers import set_seed
    set_seed(cfg.seed)
    tok, model, device = make_model(kind, cfg)
    arm = {"kind": kind}
    print(f"  [{kind}] perplexity ...", flush=True)
    arm["clean_ppl"] = compute_perplexity(
        model, tok, texts[:cfg.ppl_num_texts], device,
        max_length=cfg.ppl_max_length, stride=cfg.ppl_stride)
    print(f"  [{kind}] robustness sweep ...", flush=True)
    arm["robustness"] = robustness_sweep(
        model, tok, texts[:cfg.robustness_num_texts], device, cfg)
    print(f"  [{kind}] attention stats ...", flush=True)
    arm["attention"] = attention_stats(
        model, tok, texts[:cfg.attn_num_texts], device, cfg)
    print(f"  [{kind}] latency ...", flush=True)
    arm["latency"] = benchmark_latency(model, tok, device, cfg)
    print(f"  [{kind}] generation ...", flush=True)
    arm["generation"] = generate_samples(model, tok, device, cfg)
    free(model)
    return arm


def select_texts(cfg: ExperimentConfig) -> List[str]:
    """
    WikiText paragraphs for this run.

    With cfg.text_sample_seed=None the behaviour is the legacy deterministic
    "first N paragraphs". With a seed set (the multi-seed protocol) a larger
    pool is loaded and shuffled first, so each seed evaluates a different
    subsample — that is what gives clean perplexity a non-degenerate variance.
    """
    n_texts = max(cfg.ppl_num_texts, cfg.robustness_num_texts, cfg.attn_num_texts)
    if cfg.text_sample_seed is None:
        print(f"Loading WikiText-2 ({n_texts} paragraphs) ...", flush=True)
        return load_wikitext(n_texts)

    pool_size = max(n_texts, cfg.text_pool_size)
    print(f"Loading WikiText-2 (pool {pool_size}, subsampling {n_texts} "
          f"@ text_sample_seed={cfg.text_sample_seed}) ...", flush=True)
    pool = list(load_wikitext(pool_size))
    random.Random(cfg.text_sample_seed).shuffle(pool)
    return pool[:n_texts]


def run_all(cfg: ExperimentConfig) -> Dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    texts = select_texts(cfg)

    results = {"config": asdict(cfg), "device": resolve_device(cfg.device), "arms": {}}
    for kind in ("softmax", "qgfd"):
        print(f"\n=== Arm: {kind} ===", flush=True)
        results["arms"][kind] = _run_arm(kind, cfg, texts)

    _save(results, cfg)
    _print_summary(results)
    _plot(results, cfg)
    return results
# __APPEND7__


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _save(results: Dict, cfg: ExperimentConfig) -> None:
    path = os.path.join(cfg.out_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw results -> {path}")


def _print_summary(results: Dict) -> None:
    sm, qg = results["arms"]["softmax"], results["arms"]["qgfd"]
    print("\n" + "=" * 62)
    print("QGFD REVIEW — SUMMARY  (softmax baseline vs QGFD)")
    print("=" * 62)
    print(f"{'Metric':<34}{'softmax':>13}{'qgfd':>13}")
    print("-" * 62)
    print(f"{'Clean perplexity':<34}{sm['clean_ppl']:>13.3f}{qg['clean_ppl']:>13.3f}")
    print(f"{'Attn entropy (nats)':<34}"
          f"{sm['attention']['mean_attention_entropy_nats']:>13.3f}"
          f"{qg['attention']['mean_attention_entropy_nats']:>13.3f}")
    print(f"{'Sink mass @ pos0':<34}"
          f"{sm['attention']['mean_sink_mass_pos0']:>13.4f}"
          f"{qg['attention']['mean_sink_mass_pos0']:>13.4f}")
    print(f"{'Prefill latency (ms)':<34}"
          f"{sm['latency']['prefill_ms']:>13.2f}{qg['latency']['prefill_ms']:>13.2f}")
    print(f"{'Tokens / s':<34}"
          f"{sm['latency']['tokens_per_s']:>13.1f}{qg['latency']['tokens_per_s']:>13.1f}")
    overhead = qg['latency']['prefill_ms'] / max(1e-9, sm['latency']['prefill_ms'])
    print("-" * 62)
    print(f"QGFD compute overhead: {overhead:.3f}x baseline prefill")
    print("\nRobustness (perplexity vs noise rate; Δ% vs clean):")
    print(f"{'rate':>8}{'softmax_ppl':>14}{'sm_Δ%':>9}{'qgfd_ppl':>12}{'qg_Δ%':>9}")
    sm_clean = sm['robustness'][0.0]; qg_clean = qg['robustness'][0.0]
    for rate in sorted(sm['robustness'], key=float):
        smp, qgp = sm['robustness'][rate], qg['robustness'][rate]
        sd = 100 * (smp - sm_clean) / sm_clean
        qd = 100 * (qgp - qg_clean) / qg_clean
        print(f"{float(rate):>8.2f}{smp:>14.3f}{sd:>9.1f}{qgp:>12.3f}{qd:>9.1f}")
# __APPEND8__


def _plot(results: Dict, cfg: ExperimentConfig) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available — skipping plots; JSON still saved)")
        return

    sm, qg = results["arms"]["softmax"], results["arms"]["qgfd"]

    # Robustness curve
    rates = sorted(sm["robustness"], key=float)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([float(r) * 100 for r in rates], [sm["robustness"][r] for r in rates],
            "o-", label="Softmax")
    ax.plot([float(r) * 100 for r in rates], [qg["robustness"][r] for r in rates],
            "s-", label="QGFD")
    ax.set_xlabel("Noise rate (%)"); ax.set_ylabel("Perplexity")
    ax.set_title("Noise robustness: perplexity vs input corruption")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(cfg.out_dir, "robustness_curve.png")
    fig.savefig(p1, dpi=150); plt.close(fig)

    # Attention entropy / sink bars
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].bar(["Softmax", "QGFD"],
                [sm["attention"]["mean_attention_entropy_nats"],
                 qg["attention"]["mean_attention_entropy_nats"]],
                color=["#888", "#3b7"])
    axes[0].set_title("Mean attention entropy (nats)")
    axes[1].bar(["Softmax", "QGFD"],
                [sm["attention"]["mean_sink_mass_pos0"],
                 qg["attention"]["mean_sink_mass_pos0"]],
                color=["#888", "#3b7"])
    axes[1].set_title("Sink mass @ pos 0")
    fig.tight_layout()
    p2 = os.path.join(cfg.out_dir, "attention_stats.png")
    fig.savefig(p2, dpi=150); plt.close(fig)
    print(f"Saved plots -> {p1}, {p2}")


# --------------------------------------------------------------------------- #
# Multi-seed protocol — mean +/- std with confidence intervals
# --------------------------------------------------------------------------- #
# Single-seed point estimates cannot support a claim like "QGFD degrades less
# under noise" when the effect is ~1% of the metric. run_all_seeds re-runs the
# whole suite per seed (varying both the evaluated text subsample and the noise
# realisation) and reports mean +/- std, t-based 95% CIs, and — the statistic
# that actually matters — the PAIRED per-seed QGFD-minus-softmax difference.

# Two-sided t critical values at 95%, indexed by degrees of freedom (n-1).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def _stat(values: Sequence[float]) -> Dict:
    """mean / std / sem / t-based 95% CI half-width for a small sample."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": 0.0, "sem": 0.0,
                "ci95": 0.0, "n": 0, "values": []}
    mean = sum(vals) / n
    if n == 1:
        std = sem = ci = 0.0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
        sem = std / math.sqrt(n)
        ci = _T95.get(n - 1, 1.96) * sem
    return {"mean": mean, "std": std, "sem": sem, "ci95": ci,
            "n": n, "values": vals}


def fmt_stat(s: Dict, prec: int = 3, ci: bool = False) -> str:
    """'11.234 +/- 0.021' (std by default, CI half-width if ci=True)."""
    spread = s["ci95"] if ci else s["std"]
    return f"{s['mean']:.{prec}f} +/- {spread:.{prec}f}"


def _rate_key(rate) -> str:
    """Stable string key for a noise rate (JSON turns float keys into strings)."""
    return f"{float(rate):.4f}"


def _rk(d: Dict, rate) -> float:
    """Read a robustness dict that may be float- or string-keyed."""
    for k in (float(rate), _rate_key(rate), str(float(rate)), str(rate)):
        if k in d:
            return float(d[k])
    raise KeyError(f"noise rate {rate!r} not in {list(d)}")


def aggregate_runs(runs: Sequence[Dict], seeds: Optional[Sequence[int]] = None) -> Dict:
    """
    Collapse a list of run_all() result dicts into mean +/- std per metric.

    Also emits `paired`: per-seed QGFD-minus-softmax differences. Because both
    arms see the identical text subsample and identical noise realisation within
    a seed, the paired difference removes the (large) between-seed corpus
    variance and is the only statistic that can honestly support a small effect.
    """
    if not runs:
        raise ValueError("aggregate_runs() got no runs")
    seeds = list(seeds) if seeds is not None else [
        r["config"].get("seed") for r in runs]
    rates = sorted({float(r) for run in runs
                    for r in run["arms"]["softmax"]["robustness"]})

    agg: Dict = {
        "meta": {
            "seeds": seeds,
            "n_seeds": len(runs),
            "model_id": runs[0]["config"]["model_id"],
            "device": runs[0].get("device"),
            "dtype": runs[0]["config"]["dtype"],
            "diffusion_steps": runs[0]["config"]["diffusion_steps"],
            "target_alpha": runs[0]["config"]["target_alpha"],
            "noise_rates": [_rate_key(r) for r in rates],
            "ci_method": "two-sided t, 95%",
            "baseline_note": ("baseline is eager materialised softmax "
                              "(SoftmaxOperator), NOT SDPA/FlashAttention — "
                              "QGFD needs the explicit probability matrix"),
        },
        "arms": {},
        "paired": {},
    }

    for kind in ("softmax", "qgfd"):
        arms = [r["arms"][kind] for r in runs]
        clean = [a["clean_ppl"] for a in arms]
        entry: Dict = {
            "clean_ppl": _stat(clean),
            "robustness": {},
            "robustness_delta_pct": {},
        }
        # attention / latency are absent for the fine-tuning track, which reuses
        # this aggregator for its post-training clean-PPL + robustness numbers.
        if all("attention" in a for a in arms):
            entry["attention"] = {
                k: _stat([a["attention"][k] for a in arms])
                for k in ("mean_attention_entropy_nats", "mean_sink_mass_pos0")
            }
        if all("latency" in a for a in arms):
            entry["latency"] = {
                k: _stat([a["latency"][k] for a in arms])
                for k in ("prefill_ms", "tokens_per_s")
            }
        for rate in rates:
            ppls = [_rk(a["robustness"], rate) for a in arms]
            entry["robustness"][_rate_key(rate)] = _stat(ppls)
            # Degradation relative to THIS seed's own clean run.
            base = [_rk(a["robustness"], 0.0) for a in arms]
            entry["robustness_delta_pct"][_rate_key(rate)] = _stat(
                [100.0 * (p - b) / b for p, b in zip(ppls, base)])
        agg["arms"][kind] = entry

    sm = [r["arms"]["softmax"] for r in runs]
    qg = [r["arms"]["qgfd"] for r in runs]
    agg["paired"]["clean_ppl_qgfd_minus_softmax"] = _stat(
        [q["clean_ppl"] - s["clean_ppl"] for s, q in zip(sm, qg)])
    if all("latency" in a for a in sm + qg):
        agg["paired"]["latency_overhead_x"] = _stat(
            [q["latency"]["prefill_ms"] / max(1e-9, s["latency"]["prefill_ms"])
             for s, q in zip(sm, qg)])
    # Robustness gap: softmax degradation minus QGFD degradation, per seed.
    # POSITIVE => QGFD degraded LESS => QGFD is more robust at that noise rate.
    gap: Dict = {}
    for rate in rates:
        if rate == 0.0:
            continue
        per_seed = []
        for s, q in zip(sm, qg):
            sd = 100.0 * (_rk(s["robustness"], rate) - _rk(s["robustness"], 0.0)) / _rk(s["robustness"], 0.0)
            qd = 100.0 * (_rk(q["robustness"], rate) - _rk(q["robustness"], 0.0)) / _rk(q["robustness"], 0.0)
            per_seed.append(sd - qd)
        gap[_rate_key(rate)] = _stat(per_seed)
    agg["paired"]["robustness_gap_pct"] = gap
    return agg


def _signif(s: Dict) -> str:
    """'*' when the paired 95% CI excludes zero, 'ns' otherwise."""
    if s["n"] < 2:
        return "?"
    return "*" if abs(s["mean"]) > s["ci95"] > 0 else "ns"


def _print_aggregate(agg: Dict) -> None:
    m = agg["meta"]
    sm, qg = agg["arms"]["softmax"], agg["arms"]["qgfd"]
    w = 74
    print("\n" + "=" * w)
    print(f"QGFD MULTI-SEED SUMMARY — {m['model_id']}  "
          f"(n={m['n_seeds']} seeds: {m['seeds']})")
    print(f"T={m['diffusion_steps']}, alpha={m['target_alpha']}, "
          f"dtype={m['dtype']}, device={m['device']}")
    print("=" * w)
    print(f"{'Metric':<34}{'softmax':>19}{'qgfd':>19}")
    print("-" * w)
    rows = [("Clean perplexity", sm["clean_ppl"], qg["clean_ppl"], 3)]
    if "attention" in sm and "attention" in qg:
        rows += [
            ("Attn entropy (nats)",
             sm["attention"]["mean_attention_entropy_nats"],
             qg["attention"]["mean_attention_entropy_nats"], 3),
            ("Sink mass @ pos0",
             sm["attention"]["mean_sink_mass_pos0"],
             qg["attention"]["mean_sink_mass_pos0"], 4),
        ]
    if "latency" in sm and "latency" in qg:
        rows += [
            ("Prefill latency (ms)",
             sm["latency"]["prefill_ms"], qg["latency"]["prefill_ms"], 2),
            ("Tokens / s",
             sm["latency"]["tokens_per_s"], qg["latency"]["tokens_per_s"], 1),
        ]
    for name, a, b, p in rows:
        print(f"{name:<34}{fmt_stat(a, p):>19}{fmt_stat(b, p):>19}")
    print("-" * w)
    print("Robustness — perplexity degradation vs clean (mean +/- std over seeds):")
    print(f"{'noise':>7}{'softmax d%':>20}{'qgfd d%':>20}{'gap (sm-qg)':>18}{'':>4}")
    for rate in m["noise_rates"]:
        if float(rate) == 0.0:
            continue
        s_d = sm["robustness_delta_pct"][rate]
        q_d = qg["robustness_delta_pct"][rate]
        g = agg["paired"]["robustness_gap_pct"][rate]
        print(f"{float(rate) * 100:>6.0f}%{fmt_stat(s_d, 2):>20}"
              f"{fmt_stat(q_d, 2):>20}{fmt_stat(g, 2):>18}{_signif(g):>4}")
    print("-" * w)
    cp = agg["paired"]["clean_ppl_qgfd_minus_softmax"]
    print(f"Paired clean-PPL delta (qgfd - softmax): {fmt_stat(cp, 4)}  "
          f"[95% CI +/-{cp['ci95']:.4f}] {_signif(cp)}")
    if "latency_overhead_x" in agg["paired"]:
        lo = agg["paired"]["latency_overhead_x"]
        print(f"Paired prefill overhead: {fmt_stat(lo, 3)}x  "
              f"(vs EAGER softmax, not SDPA/FlashAttention)")
    print("'*' = paired 95% CI excludes 0 · 'ns' = not distinguishable from 0")
    print("=" * w)


def _plot_aggregate(agg: Dict, out_dir: str) -> List[str]:
    """Robustness curve with error bars + the paired robustness gap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available — skipping aggregate plots)")
        return []

    m = agg["meta"]
    rates = [float(r) for r in m["noise_rates"]]
    sm, qg = agg["arms"]["softmax"], agg["arms"]["qgfd"]
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for arm, label, marker in ((sm, "Softmax", "o"), (qg, "QGFD", "s")):
        ys = [arm["robustness"][_rate_key(r)]["mean"] for r in rates]
        es = [arm["robustness"][_rate_key(r)]["std"] for r in rates]
        ax.errorbar([r * 100 for r in rates], ys, yerr=es, fmt=marker + "-",
                    capsize=4, label=label)
    ax.set_xlabel("Character noise rate (%)")
    ax.set_ylabel("Perplexity")
    ax.set_title(f"Noise robustness (n={m['n_seeds']} seeds, mean +/- std)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    nz = [r for r in rates if r != 0.0]
    gaps = [agg["paired"]["robustness_gap_pct"][_rate_key(r)]["mean"] for r in nz]
    cis = [agg["paired"]["robustness_gap_pct"][_rate_key(r)]["ci95"] for r in nz]
    ax.bar([f"{r * 100:.0f}%" for r in nz], gaps, yerr=cis, capsize=5,
           color=["#3b7" if g > 0 else "#c55" for g in gaps])
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Character noise rate")
    ax.set_ylabel("softmax d% - qgfd d%   (>0 = QGFD more robust)")
    ax.set_title("Paired robustness gap (95% CI)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(m["model_id"], fontsize=10)
    fig.tight_layout()
    p = os.path.join(out_dir, "robustness_aggregated.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    paths.append(p)
    print(f"Saved aggregate plot -> {p}")
    return paths


def run_all_seeds(cfg: ExperimentConfig,
                  seeds: Sequence[int] = (0, 1, 2),
                  vary_texts: bool = True) -> Dict:
    """
    Run the full suite once per seed and aggregate into mean +/- std.

    Each seed varies (a) the evaluated WikiText subsample and (b) the character
    noise realisation, while keeping both arms perfectly paired within a seed.
    Per-seed artefacts land in `<out_dir>/seed<k>/`; the aggregate goes to
    `<out_dir>/results_aggregated.json`.

    Setting vary_texts=False keeps the deterministic corpus slice — then clean
    perplexity is identical across seeds by construction (std == 0), which is
    honest but uninformative, so the default is True.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)
    runs, seeds = [], list(seeds)
    for i, s in enumerate(seeds):
        print(f"\n{'#' * 74}\n# SEED {s}  ({i + 1}/{len(seeds)})\n{'#' * 74}", flush=True)
        sub = replace(
            cfg,
            seed=s,
            robustness_seed=s,
            text_sample_seed=(s if vary_texts else None),
            out_dir=os.path.join(cfg.out_dir, f"seed{s}"),
        )
        runs.append(run_all(sub))

    agg = aggregate_runs(runs, seeds)
    path = os.path.join(cfg.out_dir, "results_aggregated.json")
    with open(path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved aggregated results -> {path}")
    _print_aggregate(agg)
    _plot_aggregate(agg, cfg.out_dir)
    return agg


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="QGFD Milestone-1 review experiments")
    ap.add_argument("--model_id", default=ExperimentConfig.model_id)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out_dir", default="./qgfd_review_results")
    ap.add_argument("--ppl_num_texts", type=int, default=ExperimentConfig.ppl_num_texts)
    ap.add_argument("--diffusion_steps", type=int, default=ExperimentConfig.diffusion_steps)
    ap.add_argument("--target_alpha", type=float, default=ExperimentConfig.target_alpha)
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated seeds, e.g. '0,1,2'. Enables the "
                         "multi-seed protocol (mean +/- std + paired CIs).")
    ap.add_argument("--quick", action="store_true",
                    help="Tiny budget for a fast smoke run")
    a = ap.parse_args()
    cfg = ExperimentConfig(model_id=a.model_id, device=a.device, dtype=a.dtype,
                           out_dir=a.out_dir, ppl_num_texts=a.ppl_num_texts,
                           diffusion_steps=a.diffusion_steps,
                           target_alpha=a.target_alpha)
    if a.quick:
        cfg.ppl_num_texts = 8; cfg.robustness_num_texts = 8
        cfg.attn_num_texts = 4; cfg.latency_iters = 3; cfg.latency_warmup = 1
        cfg.text_pool_size = 64
    if a.seeds:
        run_all_seeds(cfg, seeds=[int(s) for s in a.seeds.split(",") if s.strip()])
    else:
        run_all(cfg)
















