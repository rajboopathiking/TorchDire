import random
import time
import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Union
from torchdire.nn.llama_qgfd import patch_llama_with_qgfd
from torchdire.utils.replacer import wrap_model_with_qgfd


DEFAULT_BENCH_TEXTS = [
    "The rapid development of artificial intelligence has revolutionized modern technology and computational research.",
    "Quantum graph flow diffusion provides a robust mathematical framework for regularizing attention distributions.",
    "Natural language processing models require robust contextual representations to handle noisy transcription input.",
    "Efficient transformer decoding with key-value caching reduces computational complexity from quadratic to linear time.",
]

DEFAULT_NOISE_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20]


def inject_noise(text: str, noise_prob: float = 0.15) -> str:
    """Inject realistic character-level noise (swaps, drops, typos) into text."""
    if noise_prob <= 0.0 or not text:
        return text

    chars = list(text)
    n = len(chars)
    num_corrupt = max(1, int(n * noise_prob))
    indices = random.sample(range(n), min(n, num_corrupt))

    qwerty_neighbors = {
        'a': 's', 'b': 'v', 'c': 'x', 'd': 'f', 'e': 'w', 'f': 'g', 'g': 'h',
        'h': 'j', 'i': 'o', 'j': 'k', 'k': 'l', 'l': 'k', 'm': 'n', 'n': 'm',
        'o': 'p', 'p': 'o', 'q': 'w', 'r': 'e', 's': 'd', 't': 'r', 'u': 'i',
        'v': 'b', 'w': 'q', 'x': 'c', 'y': 't', 'z': 'x'
    }

    for idx in indices:
        ch = chars[idx].lower()
        if ch in qwerty_neighbors:
            chars[idx] = qwerty_neighbors[ch] if chars[idx].islower() else qwerty_neighbors[ch].upper()
        elif ch.isalnum():
            chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")

    return "".join(chars)


def compute_perplexity(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: torch.device,
) -> float:
    """Compute perplexity over a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in encoded.items()}
            input_ids = inputs["input_ids"]
            if input_ids.size(1) < 2:
                continue

            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    if total_tokens == 0:
        return float("nan")

    avg_nll = total_loss / total_tokens
    return float(math.exp(min(avg_nll, 100.0)))


def measure_generation_performance(
    model: nn.Module,
    tokenizer: Any,
    prompt: str = "Hello, I'm a language model",
    max_new_tokens: int = 50,
    num_runs: int = 3,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Measure generation latency (ms), tokens per second (TPS), and peak VRAM memory (MB)."""
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in encoded.items()}

    # Warmup
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True)

    latencies = []
    tps_list = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()

            gen_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            latency_ms = (t1 - t0) * 1000.0
            tps = gen_tokens / max(1e-5, (t1 - t0))

            latencies.append(latency_ms)
            tps_list.append(tps)

    avg_latency = sum(latencies) / len(latencies)
    avg_tps = sum(tps_list) / len(tps_list)

    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "latency_ms": round(avg_latency, 2),
        "tokens_per_sec": round(avg_tps, 2),
        "peak_vram_mb": round(peak_mem_mb, 2),
    }


def run_single_benchmark(
    model: nn.Module,
    tokenizer: Any,
    name: str = "Model",
    bench_texts: Optional[List[str]] = None,
    noise_prob: float = 0.15,
    do_noise_sweep: bool = False,
    max_new_tokens: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run full evaluation suite on a single model instance."""
    if bench_texts is None:
        bench_texts = DEFAULT_BENCH_TEXTS

    if device is None:
        param = next(model.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")

    # 1. Perplexity on Clean Input
    ppl_clean = compute_perplexity(model, tokenizer, bench_texts, device)

    # 2. Perplexity on Noisy Input
    noisy_texts = [inject_noise(t, noise_prob) for t in bench_texts]
    ppl_noisy = compute_perplexity(model, tokenizer, noisy_texts, device)

    # 3. Generation Speed & Latency
    perf = measure_generation_performance(
        model, tokenizer, max_new_tokens=max_new_tokens, device=device
    )

    # 4. Optional Noise Sweep
    noise_curve = None
    if do_noise_sweep:
        noise_curve = {}
        for p in DEFAULT_NOISE_SWEEP:
            cur_noisy = [inject_noise(t, p) for t in bench_texts] if p > 0 else bench_texts
            noise_curve[p] = compute_perplexity(model, tokenizer, cur_noisy, device)

    return {
        "name": name,
        "ppl_clean": round(ppl_clean, 2),
        "ppl_noisy": round(ppl_noisy, 2),
        "latency_ms": perf["latency_ms"],
        "tokens_per_sec": perf["tokens_per_sec"],
        "peak_vram_mb": perf["peak_vram_mb"],
        "noise_prob": noise_prob,
        "noise_curve": noise_curve,
    }


def compare_qgfd_vs_softmax(
    model: nn.Module,
    tokenizer: Any,
    diffusion_steps: int = 2,
    target_alpha: float = 0.02,
    warmup_steps: int = 0,
    mode: str = "full",
    noise_prob: float = 0.15,
    do_noise_sweep: bool = False,
    max_new_tokens: int = 50,
    bench_texts: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    High-Level Endpoint Function to calculate Resource & Performance trade-offs
    between Softmax Attention (Baseline) and QGFD Attention (Treatment).

    Returns detailed performance metrics, resource metrics, and relative trade-off ratios.
    """
    param = next(model.parameters(), None)
    device = param.device if param is not None else torch.device("cpu")

    if verbose:
        print("==========================================================================")
        print("⚡ QGFD vs Softmax Attention Trade-off & Performance Benchmark")
        print("==========================================================================")
        print(f"Device: {device} | Dtype: {param.dtype if param is not None else 'N/A'}")
        print(f"QGFD Config: mode='{mode}', steps={diffusion_steps}, alpha={target_alpha}")
        print("--------------------------------------------------------------------------")

    # Step 1: Benchmark Baseline (Softmax Attention)
    if verbose:
        print("[1/2] Benchmarking Baseline (Softmax Attention)...")
    base_res = run_single_benchmark(
        model=model,
        tokenizer=tokenizer,
        name="Baseline (Softmax)",
        bench_texts=bench_texts,
        noise_prob=noise_prob,
        do_noise_sweep=do_noise_sweep,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    # Step 2: Patch Model with QGFD
    if verbose:
        print("[2/2] Patching Model with QGFD Attention and Benchmarking...")

    is_llama = any("LlamaAttention" in m.__class__.__name__ for _, m in model.named_modules())
    if is_llama:
        qgfd_model = patch_llama_with_qgfd(
            model,
            diffusion_steps=diffusion_steps,
            target_alpha=target_alpha,
            warmup_steps=warmup_steps,
            mode=mode,
            verbose=False,
        )
    else:
        qgfd_model = wrap_model_with_qgfd(
            model,
            diffusion_steps=diffusion_steps,
            target_alpha=target_alpha,
            warmup_steps=warmup_steps,
            mode=mode,
            verbose=False,
        )

    qgfd_res = run_single_benchmark(
        model=qgfd_model,
        tokenizer=tokenizer,
        name=f"QGFD (steps={diffusion_steps}, alpha={target_alpha})",
        bench_texts=bench_texts,
        noise_prob=noise_prob,
        do_noise_sweep=do_noise_sweep,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    # Step 3: Calculate Trade-Off Ratios & Deltas
    clean_ppl_diff = round(qgfd_res["ppl_clean"] - base_res["ppl_clean"], 2)
    noisy_ppl_diff = round(qgfd_res["ppl_noisy"] - base_res["ppl_noisy"], 2)
    
    robustness_improvement = round(
        ((base_res["ppl_noisy"] - qgfd_res["ppl_noisy"]) / max(1e-5, base_res["ppl_noisy"])) * 100.0, 2
    )
    
    latency_overhead = round(
        ((qgfd_res["latency_ms"] - base_res["latency_ms"]) / max(1e-5, base_res["latency_ms"])) * 100.0, 2
    )
    
    tps_ratio = round(qgfd_res["tokens_per_sec"] / max(1e-5, base_res["tokens_per_sec"]), 3)

    summary = {
        "baseline": base_res,
        "qgfd": qgfd_res,
        "tradeoff": {
            "clean_ppl_delta": clean_ppl_diff,
            "noisy_ppl_delta": noisy_ppl_diff,
            "robustness_improvement_percent": robustness_improvement,
            "latency_overhead_percent": latency_overhead,
            "tps_speed_ratio": tps_ratio,
        },
    }

    if verbose:
        print("\n==========================================================================")
        print("📊 QGFD vs Softmax Attention Evaluation & Trade-off Summary")
        print("==========================================================================")
        print(f"{'Metric':<32} | {'Baseline (Softmax)':<20} | {'QGFD Attention':<20}")
        print("-" * 76)
        print(f"{'Clean Perplexity (PPL)':<32} | {base_res['ppl_clean']:<20.2f} | {qgfd_res['ppl_clean']:<20.2f}")
        print(f"{'Noisy Perplexity (PPL@' + str(noise_prob) + ')':<32} | {base_res['ppl_noisy']:<20.2f} | {qgfd_res['ppl_noisy']:<20.2f}")
        print(f"{'Generation Speed (tokens/sec)':<32} | {base_res['tokens_per_sec']:<20.2f} | {qgfd_res['tokens_per_sec']:<20.2f}")
        print(f"{'Generation Latency (ms)':<32} | {base_res['latency_ms']:<20.2f} | {qgfd_res['latency_ms']:<20.2f}")
        if device.type == "cuda":
            print(f"{'Peak VRAM (MB)':<32} | {base_res['peak_vram_mb']:<20.2f} | {qgfd_res['peak_vram_mb']:<20.2f}")
        print("-" * 76)
        print(f"💡 Robustness Gain (Noisy PPL reduction): {robustness_improvement}%")
        print(f"💡 Computational Overhead (Latency diff): {latency_overhead}%")
        print("==========================================================================\n")

    return summary
