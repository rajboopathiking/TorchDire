import time
import gc
import random
import string
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers  # for version info

from .qgfd_attention import MultiHeadQGFDLayer
from .universal_qgfd_replacer import wrap_model_with_qgfd

# ============================================================
# CONFIG
# ============================================================

MODELS = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "gpt2",
    "gpt2-medium",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global defaults (used as fallback)
DEFAULT_TARGET_ALPHA = 0.05
DIFFUSION_STEPS = 1
WARMUP_STEPS = 0
KERNEL_SIZE = 5

DEFAULT_NOISE_PROBABILITY = 0.15
NOISE_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20]

# ============================================================
# ENV CHECK: ARE CAUSAL LM CLASSES AVAILABLE?
# ============================================================

try:
    from transformers import GPT2LMHeadModel, OPTForCausalLM  # noqa
    CAUSAL_LM_AVAILABLE = True
except Exception:
    CAUSAL_LM_AVAILABLE = False

if not CAUSAL_LM_AVAILABLE:
    print("WARNING: Your transformers install is missing GPT2LMHeadModel / OPTForCausalLM.")
    print("         No AutoModelForCausalLM causal models will load.")
    print("         Version:", transformers.__version__)
    print("         To fix, in a shell or notebook cell do:")
    print('           pip uninstall -y transformers')
    print('           pip install -U "transformers[torch]==4.44.2" accelerate sentencepiece')
    print("         Then restart the runtime and re-run this script.\n")

# ============================================================
# BENCHMARK DATASET (BUILT-IN)
# ============================================================

BENCH_TEXTS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming many industries around the world.",
    "Python is a widely used programming language for machine learning and data science.",
    "The attention mechanism revolutionized natural language processing and sequence modeling.",
    "Scaling laws suggest that larger models tend to perform better given sufficient data and compute.",
    "Robustness to noisy inputs is a key property for deploying models in real-world applications.",
]

SENTIMENT_SAMPLES = [
    ("I absolutely loved this movie!", "Positive"),
    ("The food was terrible and cold.", "Negative"),
    ("What a fantastic day, I feel so happy.", "Positive"),
    ("I am very disappointed with the service.", "Negative"),
    ("This product works wonderfully and exceeded my expectations.", "Positive"),
    ("The experience was awful and I will not return.", "Negative"),
]


# ============================================================
# PER-MODEL DYNAMIC TARGET_ALPHA
# ============================================================

def compute_target_alpha(model_name: str) -> float:
    """
    Choose target_alpha per model to avoid oversmoothing on small GPT-2
    while keeping strong effect on larger OPT models.

    Heuristic:
      - OPT 125M / 350M:      0.05   (works very well in your results)
      - GPT-2 (small, 117M):  0.01   (weaker smoothing to avoid degradation)
      - GPT-2-medium:         0.02   (moderate smoothing)
      - Fallback:             DEFAULT_TARGET_ALPHA
    """
    name = model_name.lower()

    if "opt-125m" in name or "opt-350m" in name:
        return 0.05
    if name == "gpt2":
        return 0.01
    if "gpt2-medium" in name:
        return 0.02

    # default fallback
    return DEFAULT_TARGET_ALPHA


# ============================================================
# HELPER: NOISE INJECTION
# ============================================================

def inject_noise(text: str, prob: float = 0.1) -> str:
    chars = list(text)
    for i in range(len(chars)):
        if random.random() < prob and chars[i] not in string.whitespace:
            if i < len(chars) - 1 and random.random() > 0.5:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            else:
                chars[i] = random.choice(string.ascii_letters)
    return "".join(chars)


# ============================================================
# METRIC 1 & 2: PERPLEXITY
# ============================================================

def get_perplexity(model, tokenizer, texts: List[str]) -> float:
    model.eval()
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

    max_length = getattr(model.config, "max_position_embeddings", 1024)
    stride = 512
    seq_len = encodings["input_ids"].size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings["input_ids"][:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()
    return ppl


# ============================================================
# METRIC 3: LATENCY
# ============================================================

def measure_latency(model, tokenizer, prompt: str = "The future of AI is", n_tokens: int = 50) -> float:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    start_time = time.time()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False)
    end_time = time.time()

    duration = max(end_time - start_time, 1e-6)
    tokens_per_sec = n_tokens / duration
    return tokens_per_sec


# ============================================================
# METRIC 4: PEAK MEMORY
# ============================================================

def measure_memory() -> float:
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


# ============================================================
# METRIC 5: ZERO-SHOT SENTIMENT
# ============================================================

def evaluate_sentiment(model, tokenizer) -> float:
    model.eval()
    correct = 0

    candidates = [" Positive", " Negative"]
    candidate_ids = []
    for c in candidates:
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) == 0:
            candidate_ids.append(tokenizer.eos_token_id)
        else:
            candidate_ids.append(ids[0])

    for text, label in SENTIMENT_SAMPLES:
        prompt = f"Review: {text}\nSentiment:"
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]

        pos_score = logits[candidate_ids[0]].item()
        neg_score = logits[candidate_ids[1]].item()

        pred = "Positive" if pos_score > neg_score else "Negative"
        if pred == label:
            correct += 1

    return correct / len(SENTIMENT_SAMPLES)


# ============================================================
# NOISE SWEEP
# ============================================================

def noise_sweep(model, tokenizer, texts: List[str], noise_levels) -> Dict[float, float]:
    results = {}
    for p in noise_levels:
        noisy_texts = [inject_noise(t, p) for t in texts]
        ppl = get_perplexity(model, tokenizer, noisy_texts)
        results[p] = ppl
    return results


# ============================================================
# MODEL LOAD / CLEANUP
# ============================================================

def load_model_and_tokenizer(model_name: str):
    print(f"\n--- Loading {model_name} ---")
    if not CAUSAL_LM_AVAILABLE:
        print(f"!! Skipping {model_name} because causal LM classes are not available.")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    except ValueError as e:
        print(f"!! Skipping {model_name} due to load error:")
        print(f"   {e}")
        print(f"   transformers version: {transformers.__version__}")
        return None, None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = tokenizer.eos_token_id

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    return model, tokenizer


def cleanup_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# BENCHMARK RUNNER
# ============================================================

def run_benchmark(
    model,
    tokenizer,
    name: str = "Model",
    noise_prob: float = DEFAULT_NOISE_PROBABILITY,
    do_noise_sweep: bool = False,
):
    print(f"\n⚡ Running Benchmarks for: {name}")

    ppl_clean = get_perplexity(model, tokenizer, BENCH_TEXTS)
    print(f"  -> Clean PPL:      {ppl_clean:.2f}")

    noisy_texts = [inject_noise(t, noise_prob) for t in BENCH_TEXTS]
    ppl_noisy = get_perplexity(model, tokenizer, noisy_texts)
    print(f"  -> Noisy PPL:      {ppl_noisy:.2f} (Noise Lvl: {noise_prob})")

    acc = evaluate_sentiment(model, tokenizer)
    print(f"  -> Sentiment Acc:  {acc:.0%}")

    tps = measure_latency(model, tokenizer)
    print(f"  -> Speed:          {tps:.2f} tokens/sec")

    mem = measure_memory()
    print(f"  -> Peak VRAM:      {mem:.1f} MB")

    noise_curve = None
    if do_noise_sweep:
        print("  -> Running noise sweep...")
        noise_curve = noise_sweep(model, tokenizer, BENCH_TEXTS, NOISE_SWEEP)
        for p, ppl in noise_curve.items():
            print(f"     - Noise {p:.2f}: PPL={ppl:.2f}")

    return {
        "ppl_clean": ppl_clean,
        "ppl_noisy": ppl_noisy,
        "acc": acc,
        "tps": tps,
        "mem": mem,
        "noise_curve": noise_curve,
        "noise_prob": noise_prob,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    all_results = []

    for MODEL_NAME in MODELS:
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
        if model is None:
            continue

        # -------- baseline (standard attention) --------
        base_stats = run_benchmark(
            model,
            tokenizer,
            name=f"{MODEL_NAME} - Baseline (Standard Attention)",
            noise_prob=DEFAULT_NOISE_PROBABILITY,
            do_noise_sweep=True,
        )
        base_vram = measure_memory()

        # -------- QGFD injection with per-model alpha --------
        target_alpha = compute_target_alpha(MODEL_NAME)
        print(f"\n... Injecting QGFD Layers for {MODEL_NAME} with target_alpha={target_alpha} ...")

        model = wrap_model_with_qgfd(
            model,
            MultiHeadQGFDLayer,
            diffusion_steps=DIFFUSION_STEPS,
            target_alpha=target_alpha,
            warmup_steps=WARMUP_STEPS,
            kernel_size=KERNEL_SIZE,
            early_stop_eps=0.0,
            verbose=False,
        )

        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()

        qgfd_stats = run_benchmark(
            model,
            tokenizer,
            name=f"{MODEL_NAME} - QGFD (Wrapped, alpha={target_alpha})",
            noise_prob=DEFAULT_NOISE_PROBABILITY,
            do_noise_sweep=True,
        )
        qgfd_vram = measure_memory()

        # -------- summary --------
        print("\n" + "=" * 60)
        print(f"MODEL: {MODEL_NAME}")
        print(f"{'METRIC':<20} | {'BASELINE':<12} | {'QGFD':<12} | {'CHANGE':<12}")
        print("-" * 60)

        diff = qgfd_stats["ppl_clean"] - base_stats["ppl_clean"]
        print(f"{'PPL (Clean)':<20} | {base_stats['ppl_clean']:<12.2f} | {qgfd_stats['ppl_clean']:<12.2f} | {diff:+12.2f}")

        diff = qgfd_stats["ppl_noisy"] - base_stats["ppl_noisy"]
        print(f"{'PPL (Noisy)':<20} | {base_stats['ppl_noisy']:<12.2f} | {qgfd_stats['ppl_noisy']:<12.2f} | {diff:+12.2f}")

        base_ratio = base_stats["ppl_noisy"] / base_stats["ppl_clean"]
        qgfd_ratio = qgfd_stats["ppl_noisy"] / qgfd_stats["ppl_clean"]
        print(f"{'Noise Sensitivity':<20} | {base_ratio:<12.2f} | {qgfd_ratio:<12.2f} | {(qgfd_ratio - base_ratio):+12.2f}")

        diff = qgfd_stats["tps"] - base_stats["tps"]
        print(f"{'Speed (tok/s)':<20} | {base_stats['tps']:<12.2f} | {qgfd_stats['tps']:<12.2f} | {diff:+12.2f}")

        print(f"{'Sentiment Acc':<20} | {base_stats['acc']:<12.0%} | {qgfd_stats['acc']:<12.0%} | {'---':<12}")

        print(f"{'Peak VRAM (MB)':<20} | {base_vram:<12.1f} | {qgfd_vram:<12.1f} | {(qgfd_vram - base_vram):+12.1f}")

        print("=" * 60)
        print("INTERPRETATION:")
        if qgfd_stats["ppl_noisy"] < base_stats["ppl_noisy"]:
            print("  ✅ QGFD is MORE ROBUST to noise (better noisy PPL).")
        else:
            print("  ❌ QGFD did not improve noisy PPL in this configuration.")

        if qgfd_stats["tps"] < base_stats["tps"] * 0.8:
            print("  ⚠️ QGFD introduced >20% latency overhead.")
        else:
            print("  ✅ QGFD latency overhead is acceptable.")

        all_results.append({
            "model": MODEL_NAME,
            "baseline": base_stats,
            "qgfd": qgfd_stats,
            "baseline_vram": base_vram,
            "qgfd_vram": qgfd_vram,
            "target_alpha": target_alpha,
        })

        cleanup_model(model, tokenizer)

    print("\n\n=== GLOBAL SUMMARY (RAW) ===")
    if not all_results:
        print("No models were successfully loaded. See warnings above about transformers install.")
    else:
        for res in all_results:
            print(
                res["model"], "(alpha=", res["target_alpha"], ")->",
                "PPL_clean:", f"{res['baseline']['ppl_clean']:.2f}", "->", f"{res['qgfd']['ppl_clean']:.2f}",
                "| PPL_noisy:", f"{res['baseline']['ppl_noisy']:.2f}", "->", f"{res['qgfd']['ppl_noisy']:.2f}",
            )
