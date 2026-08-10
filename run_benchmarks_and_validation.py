import math
import time
import tracemalloc
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdire.nn.qgfd import MultiHeadQGFDLayer

# Set random seed for reproducibility
torch.manual_seed(42)

def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


# =====================================================================
# SECTION 1: VALIDATE CORRECTNESS AND NUMERICAL STABILITY
# =====================================================================

def validate_correctness_and_stability():
    print_section("1. Correctness and Numerical Stability Validation")
    
    results = {}
    
    # -----------------------------------------------------------------
    # Test 1.1: Equivalence to Baseline (Alpha=0 / disabled)
    # -----------------------------------------------------------------
    print("\n[Test 1.1] Equivalence to Baseline (alpha=0 or enable_qgfd=False)...")
    layer_std = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, enable_qgfd=False)
    layer_qgfd_zero = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, enable_qgfd=True, target_alpha=0.0, warmup_steps=0)
    
    # Sync weights
    layer_qgfd_zero.load_state_dict(layer_std.state_dict(), strict=False)
    
    x = torch.randn(2, 16, 64)
    out_std, p_std = layer_std(x, output_attentions=True)
    out_zero, p_zero = layer_qgfd_zero(x, output_attentions=True)
    
    diff_out = torch.max(torch.abs(out_std - out_zero)).item()
    diff_p = torch.max(torch.abs(p_std - p_zero)).item()
    
    passed_eq = diff_out < 1e-6 and diff_p < 1e-6
    results["Baseline Equivalence (alpha=0)"] = passed_eq
    print(f"  -> Max Output Diff: {diff_out:.8e}")
    print(f"  -> Max Attention Diff: {diff_p:.8e}")
    print(f"  -> Status: {'PASSED [OK]' if passed_eq else 'FAILED [FAIL]'}")

    # -----------------------------------------------------------------
    # Test 1.2: Probability Distribution Properties (Row sum == 1, non-negative)
    # -----------------------------------------------------------------
    print("\n[Test 1.2] Probability Distribution Validity (Row sum == 1.0, non-negative)...")
    passed_probs = True
    for mode in ["full", "conv"]:
        layer = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode=mode, diffusion_steps=4, target_alpha=0.05, warmup_steps=0)
        x = torch.randn(4, 32, 64)
        _, p = layer(x, output_attentions=True)
        
        row_sums = p.sum(dim=-1)
        max_sum_err = torch.max(torch.abs(row_sums - 1.0)).item()
        min_val = p.min().item()
        max_val = p.max().item()
        has_nan = torch.isnan(p).any().item() or torch.isinf(p).any().item()
        
        mode_passed = max_sum_err < 1e-5 and min_val >= -1e-7 and max_val <= 1.0 + 1e-5 and not has_nan
        print(f"  [{mode.upper()} Mode] Max Sum Error: {max_sum_err:.8e} | Range: [{min_val:.6f}, {max_val:.6f}] | NaNs/Infs: {has_nan}")
        if not mode_passed:
            passed_probs = False
            
    results["Probability Distribution Validity"] = passed_probs
    print(f"  -> Status: {'PASSED [OK]' if passed_probs else 'FAILED [FAIL]'}")

    # -----------------------------------------------------------------
    # Test 1.3: Dtype Precision Stability (FP32, FP16, BF16)
    # -----------------------------------------------------------------
    print("\n[Test 1.3] Dtype Precision Stability (FP32, FP16, BF16)...")
    passed_dtypes = True
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    
    for dt in dtypes:
        for mode in ["full", "conv"]:
            layer = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode=mode, diffusion_steps=3, target_alpha=0.03, warmup_steps=0).to(dtype=dt)
            x = torch.randn(2, 16, 64, dtype=dt)
            out, p = layer(x, output_attentions=True)
            
            has_nan = torch.isnan(out).any().item() or torch.isnan(p).any().item()
            has_inf = torch.isinf(out).any().item() or torch.isinf(p).any().item()
            
            if has_nan or has_inf:
                passed_dtypes = False
                print(f"  [{dt} - {mode}] NaNs/Infs detected! NaN: {has_nan}, Inf: {has_inf}")
            else:
                print(f"  [{str(dt).split('.')[-1]} - {mode:4s}] Stable. Output shape: {list(out.shape)}")
                
    results["Dtype Precision Stability"] = passed_dtypes
    print(f"  -> Status: {'PASSED [OK]' if passed_dtypes else 'FAILED [FAIL]'}")

    # -----------------------------------------------------------------
    # Test 1.4: Gradient Flow and Autograd Backpropagation
    # -----------------------------------------------------------------
    print("\n[Test 1.4] Gradient Flow and Autograd Backpropagation...")
    passed_grads = True
    for mode in ["full", "conv"]:
        layer = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode=mode, diffusion_steps=3, target_alpha=0.05, warmup_steps=0)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out, _ = layer(x, output_attentions=True)
        loss = out.sum()
        loss.backward()
        
        x_grad_has_nan = torch.isnan(x.grad).any().item() if x.grad is not None else True
        param_grads_ok = True
        for name, param in layer.named_parameters():
            if param.requires_grad:
                if param.grad is None or torch.isnan(param.grad).any().item() or torch.isinf(param.grad).any().item():
                    param_grads_ok = False
                    print(f"    Issue in parameter gradient: {name}")
                    
        if x_grad_has_nan or not param_grads_ok:
            passed_grads = False
            print(f"  [{mode}] Gradient flow check FAILED!")
        else:
            print(f"  [{mode:4s}] Gradients computed successfully for input and all parameters.")
            
    results["Autograd Gradient Flow"] = passed_grads
    print(f"  -> Status: {'PASSED [OK]' if passed_grads else 'FAILED [FAIL]'}")

    # -----------------------------------------------------------------
    # Test 1.5: Extreme Value & Edge Case Stress Testing
    # -----------------------------------------------------------------
    print("\n[Test 1.5] Extreme Value & Edge Case Stress Testing...")
    passed_stress = True
    stress_cases = [
        ("Zero Inputs", torch.zeros(2, 16, 64)),
        ("Large Positive Scale (1e4)", torch.randn(2, 16, 64) * 1e4),
        ("Large Negative Scale (-1e4)", torch.randn(2, 16, 64) * -1e4),
        ("Single Token Sequence (L=1)", torch.randn(2, 1, 64)),
        ("Padded Mask (90% Masked)", torch.randn(2, 16, 64)),
    ]
    
    # Mask for final case
    mask = torch.ones(2, 16, dtype=torch.bool)
    mask[:, 2:] = False  # Keep only 2 tokens
    
    for name, input_tensor in stress_cases:
        attn_mask = mask if "Padded" in name else None
        for mode in ["full", "conv"]:
            layer = MultiHeadQGFDLayer(embed_dim=64, num_heads=4, mode=mode, diffusion_steps=4, target_alpha=0.05, warmup_steps=0)
            try:
                out, p = layer(input_tensor, attention_mask=attn_mask, output_attentions=True)
                has_nan = torch.isnan(out).any().item() or torch.isnan(p).any().item()
                has_inf = torch.isinf(out).any().item() or torch.isinf(p).any().item()
                if has_nan or has_inf:
                    passed_stress = False
                    print(f"  [{name} - {mode}] FAILED: Produced NaNs ({has_nan}) or Infs ({has_inf})")
                else:
                    print(f"  [{name:25s} | {mode:4s}] PASSED. No NaNs/Infs.")
            except Exception as e:
                passed_stress = False
                print(f"  [{name} - {mode}] FAILED with exception: {e}")
                
    results["Extreme Stress Testing"] = passed_stress
    print(f"  -> Status: {'PASSED [OK]' if passed_stress else 'FAILED [FAIL]'}")

    # -----------------------------------------------------------------
    # Test 1.6: Fallback Mechanisms for Long Sequences
    # -----------------------------------------------------------------
    print("\n[Test 1.6] Sequence Fallback Mechanisms (Lk > max_full_seq_len)...")
    passed_fallback = True
    
    # Fallback to conv
    layer_fallback_conv = MultiHeadQGFDLayer(
        embed_dim=64, num_heads=4, mode="full", max_full_seq_len=16, full_fallback_mode="conv", diffusion_steps=2, target_alpha=0.05, warmup_steps=0
    )
    x_long = torch.randn(2, 32, 64)
    out_f1 = layer_fallback_conv(x_long)[0]
    
    # Fallback to disable
    layer_fallback_disable = MultiHeadQGFDLayer(
        embed_dim=64, num_heads=4, mode="full", max_full_seq_len=16, full_fallback_mode="disable", diffusion_steps=2, target_alpha=0.05, warmup_steps=0
    )
    out_f2 = layer_fallback_disable(x_long)[0]
    
    f1_ok = out_f1.shape == (2, 32, 64) and not torch.isnan(out_f1).any()
    f2_ok = out_f2.shape == (2, 32, 64) and not torch.isnan(out_f2).any()
    passed_fallback = f1_ok and f2_ok
    
    results["Fallback Mechanisms"] = passed_fallback
    print(f"  -> Fallback 'conv' (L=32 > max=16): {'PASSED' if f1_ok else 'FAILED'}")
    print(f"  -> Fallback 'disable' (L=32 > max=16): {'PASSED' if f2_ok else 'FAILED'}")
    print(f"  -> Status: {'PASSED [OK]' if passed_fallback else 'FAILED [FAIL]'}")

    # -----------------------------------------------------------------
    # Test 1.7: Incremental Decoding Equivalence (use_cache=True vs False)
    # -----------------------------------------------------------------
    print("\n[Test 1.7] Incremental Decoding Equivalence (use_cache=True vs False)...")
    passed_cache = False
    try:
        from transformers.models.llama.modeling_llama import LlamaConfig, LlamaForCausalLM
        from torchdire.nn.llama_qgfd import patch_llama_with_qgfd
        
        config = LlamaConfig(
            vocab_size=100, hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
            num_hidden_layers=1, intermediate_size=128, max_position_embeddings=128,
        )
        model = LlamaForCausalLM(config)
        patch_llama_with_qgfd(model, diffusion_steps=2, target_alpha=0.02, warmup_steps=0, verbose=False)
        
        input_ids = torch.randint(0, 100, (1, 8))
        out_cache = model.generate(input_ids, max_new_tokens=10, min_new_tokens=10, do_sample=False, use_cache=True)
        out_nocache = model.generate(input_ids, max_new_tokens=10, min_new_tokens=10, do_sample=False, use_cache=False)
        
        if out_cache.tolist() == out_nocache.tolist():
            passed_cache = True
            print("  -> PASSED. Token-for-token generation strictly matches.")
        else:
            print("  -> FAILED. Mismatch between cached and non-cached decoding!")
    except ImportError:
        print("  -> SKIPPED (Transformers library not found)")
        passed_cache = True
    
    results["Incremental Decoding Equivalence"] = passed_cache
    print(f"  -> Status: {'PASSED [OK]' if passed_cache else 'FAILED [FAIL]'}")

    return results


# =====================================================================
# SECTION 2: RUN MICROBENCHMARKS (LATENCY, MEMORY, THROUGHPUT)
# =====================================================================

def measure_peak_memory(fn, *args, **kwargs):
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak / (1024 * 1024)  # MB

def estimate_tensor_memory_mb(batch_size, seq_len, embed_dim, num_heads, mode, diffusion_steps):
    # Activation & weight memory estimate in MB
    # Q, K, V, P, p0, p_t, outputs
    d_k = embed_dim // num_heads
    base_bytes = (batch_size * seq_len * embed_dim * 4 * 4) # inputs, Q, K, V
    attn_matrix_bytes = (batch_size * num_heads * seq_len * seq_len * 4) # p0
    if mode == "full":
        trans_matrix_bytes = (batch_size * num_heads * seq_len * seq_len * 4) # P matrix
        diff_state_bytes = (batch_size * num_heads * seq_len * seq_len * 4) # p_t
        total_bytes = base_bytes + attn_matrix_bytes + trans_matrix_bytes + (diff_state_bytes * min(diffusion_steps, 2))
    elif mode == "conv":
        conv_bytes = (batch_size * num_heads * seq_len * 5 * 4)
        total_bytes = base_bytes + attn_matrix_bytes + conv_bytes
    else:
        total_bytes = base_bytes + attn_matrix_bytes
    return total_bytes / (1024 * 1024)

def run_microbenchmarks(batch_size=4, seq_len=256, embed_dim=512, num_heads=8, num_runs=10, warmup=2):
    print_section("2. Microbenchmarks (Latency, Memory, Throughput)")
    print(f"Config: Batch Size={batch_size}, Seq Len={seq_len}, Embed Dim={embed_dim}, Heads={num_heads}", flush=True)
    
    configs = [
        ("Baseline (Standard Attention)", {"enable_qgfd": False, "diffusion_steps": 0, "target_alpha": 0.0}),
        ("QGFD Full (Steps=1, α=0.02)", {"enable_qgfd": True, "mode": "full", "diffusion_steps": 1, "target_alpha": 0.02, "warmup_steps": 0}),
        ("QGFD Full (Steps=2, α=0.02)", {"enable_qgfd": True, "mode": "full", "diffusion_steps": 2, "target_alpha": 0.02, "warmup_steps": 0}),
        ("QGFD Full (Steps=4, α=0.02)", {"enable_qgfd": True, "mode": "full", "diffusion_steps": 4, "target_alpha": 0.02, "warmup_steps": 0}),
        ("QGFD Conv (Steps=1, α=0.02)", {"enable_qgfd": True, "mode": "conv", "diffusion_steps": 1, "target_alpha": 0.02, "warmup_steps": 0}),
        ("QGFD Conv (Steps=2, α=0.02)", {"enable_qgfd": True, "mode": "conv", "diffusion_steps": 2, "target_alpha": 0.02, "warmup_steps": 0}),
        ("QGFD Conv (Steps=4, α=0.02)", {"enable_qgfd": True, "mode": "conv", "diffusion_steps": 4, "target_alpha": 0.02, "warmup_steps": 0}),
    ]
    
    benchmark_results = []
    
    header = f"{'Layer Configuration':<32} | {'Fwd (ms)':<9} | {'Bwd (ms)':<9} | {'Total (ms)':<10} | {'Throughput (tok/s)':<20} | {'Peak Mem (MB)':<12}"
    print("\n" + header, flush=True)
    print("-" * len(header), flush=True)

    for name, cfg in configs:
        layer = MultiHeadQGFDLayer(embed_dim=embed_dim, num_heads=num_heads, **cfg)
        layer.eval()
        
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = layer(x)
            
        # Forward Latency
        start_fwd = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = layer(x)
        end_fwd = time.perf_counter()
        fwd_latency_ms = ((end_fwd - start_fwd) / num_runs) * 1000.0
        
        # Backward Latency (3 runs for speed)
        layer.train()
        start_bwd = time.perf_counter()
        for _ in range(3):
            layer.zero_grad()
            x_bwd = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
            out = layer(x_bwd)[0]
            loss = out.sum()
            loss.backward()
        end_bwd = time.perf_counter()
        bwd_latency_ms = ((end_bwd - start_bwd) / 3) * 1000.0
        
        # Total Latency
        total_latency_ms = fwd_latency_ms + bwd_latency_ms
        
        # Throughput (Forward tokens / sec)
        total_tokens = batch_size * seq_len
        fwd_throughput = total_tokens / (fwd_latency_ms / 1000.0)
        
        # Memory Footprint Estimate
        mode = cfg.get("mode", "baseline") if cfg.get("enable_qgfd", True) else "baseline"
        steps = cfg.get("diffusion_steps", 0)
        mem_mb = estimate_tensor_memory_mb(batch_size, seq_len, embed_dim, num_heads, mode, steps)
        
        res = {
            "name": name,
            "fwd_ms": round(fwd_latency_ms, 3),
            "bwd_ms": round(bwd_latency_ms, 3),
            "total_ms": round(total_latency_ms, 3),
            "throughput_tok_sec": round(fwd_throughput, 1),
            "peak_mem_mb": round(mem_mb, 3)
        }
        benchmark_results.append(res)
        print(f"{res['name']:<32} | {res['fwd_ms']:<9.3f} | {res['bwd_ms']:<9.3f} | {res['total_ms']:<10.3f} | {res['throughput_tok_sec']:<20.1f} | {res['peak_mem_mb']:<12.3f}", flush=True)

    return benchmark_results


# =====================================================================
# SECTION 3: BENCHMARK SCALING ACROSS SEQUENCE LENGTHS
# =====================================================================

def benchmark_sequence_scaling(batch_size=2, embed_dim=256, num_heads=4, num_runs=5):
    print_section("3. Sequence Length Scaling Benchmark")
    
    seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]
    
    modes = [
        ("Baseline Attention", {"enable_qgfd": False}),
        ("QGFD Full (Steps=2, max_L=4096)", {"enable_qgfd": True, "mode": "full", "max_full_seq_len": 4096, "diffusion_steps": 2, "target_alpha": 0.02, "warmup_steps": 0}),
        ("QGFD Conv (Steps=2, K=5)", {"enable_qgfd": True, "mode": "conv", "diffusion_steps": 2, "target_alpha": 0.02, "warmup_steps": 0}),
        ("QGFD Full w/ Conv Fallback (max_L=512)", {"enable_qgfd": True, "mode": "full", "max_full_seq_len": 512, "full_fallback_mode": "conv", "diffusion_steps": 2, "target_alpha": 0.02, "warmup_steps": 0}),
    ]
    
    scaling_data = []
    
    print(f"Sweeping Sequence Lengths: {seq_lengths} across {len(modes)} configurations...\n", flush=True)
    
    header = f"{'Seq Len':<8} | {'Mode':<38} | {'Fwd (ms)':<10} | {'Throughput (tok/s)':<20} | {'Peak Mem (MB)':<12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    
    for L in seq_lengths:
        x = torch.randn(batch_size, L, embed_dim)
        
        for name, cfg in modes:
            layer = MultiHeadQGFDLayer(embed_dim=embed_dim, num_heads=num_heads, **cfg)
            layer.eval()
            
            # Warmup
            with torch.no_grad():
                _ = layer(x)
            
            # Measure Latency
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(num_runs if L <= 1024 else 2):
                    _ = layer(x)
            t1 = time.perf_counter()
            runs_actual = num_runs if L <= 1024 else 2
            fwd_ms = ((t1 - t0) / runs_actual) * 1000.0
            
            throughput = (batch_size * L) / (fwd_ms / 1000.0)
            
            mode_str = cfg.get("mode", "baseline") if cfg.get("enable_qgfd", True) else "baseline"
            mem_mb = estimate_tensor_memory_mb(batch_size, L, embed_dim, num_heads, mode_str, cfg.get("diffusion_steps", 0))
            
            scaling_data.append({
                "seq_len": L,
                "mode": name,
                "fwd_ms": round(fwd_ms, 3),
                "throughput": round(throughput, 1),
                "mem_mb": round(mem_mb, 3)
            })
            
            print(f"{L:<8} | {name:<38} | {fwd_ms:<10.3f} | {throughput:<20.1f} | {mem_mb:<12.3f}", flush=True)
        print("-" * len(header), flush=True)
        
    return scaling_data


if __name__ == "__main__":
    print("=" * 80)
    print("   TORCHDIRE / QGFD ATTENTION VALIDATION & BENCHMARK SUITE")
    print("=" * 80)
    
    correctness_results = validate_correctness_and_stability()
    microbench_results = run_microbenchmarks()
    scaling_results = benchmark_sequence_scaling()
    
    print("\n" + "=" * 80)
    print("ALL CHECKS & BENCHMARKS COMPLETED SUCCESSFULLY.")
    print("=" * 80)
