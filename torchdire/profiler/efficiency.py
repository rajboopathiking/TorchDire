import time
import torch
import torch.nn as nn
from torchdire.nn.qgfd import MultiHeadQGFDLayer


class QGFDProfiler:
    """
    Profiles inference latency (ms), peak VRAM memory footprint (MB), and FLOPs overhead.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def profile_layer(
        self,
        batch_size: int = 4,
        seq_len: int = 512,
        embed_dim: int = 768,
        num_heads: int = 12,
        diffusion_steps: int = 2,
        target_alpha: float = 0.02,
        mode: str = "full",
        num_runs: int = 20,
    ) -> dict[str, float]:

        x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)

        # Baseline Standard Attention
        baseline_layer = MultiHeadQGFDLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            diffusion_steps=0,
            target_alpha=0.0,
            enable_qgfd=False,
        ).to(self.device)

        # QGFD Layer
        qgfd_layer = MultiHeadQGFDLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            diffusion_steps=diffusion_steps,
            target_alpha=target_alpha,
            mode=mode,
            enable_qgfd=True,
        ).to(self.device)

        # Warmup
        for _ in range(5):
            _ = baseline_layer(x)
            _ = qgfd_layer(x)

        # Measure baseline latency
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = baseline_layer(x)
        baseline_ms = ((time.perf_counter() - start_time) / num_runs) * 1000.0

        # Measure QGFD latency
        start_time = time.perf_counter()
        for _ in range(num_runs):
            _ = qgfd_layer(x)
        qgfd_ms = ((time.perf_counter() - start_time) / num_runs) * 1000.0

        # Estimate FLOPs per forward pass
        # Standard: 2 * B * H * L * L * d_k
        d_k = embed_dim // num_heads
        base_flops = 2 * batch_size * num_heads * seq_len * seq_len * d_k

        if mode == "full":
            # QGFD adds: P matrix construction (2 * B * H * L * L * d_k) + T steps of (2 * B * H * L * L * L)
            qgfd_extra_flops = (2 * batch_size * num_heads * seq_len * seq_len * d_k) + (diffusion_steps * 2 * batch_size * num_heads * seq_len * seq_len)
        else:
            # Conv mode adds T steps of kernel conv: T * 2 * B * H * L * K
            qgfd_extra_flops = diffusion_steps * 2 * batch_size * num_heads * seq_len * 5

        total_qgfd_flops = base_flops + qgfd_extra_flops

        # VRAM memory estimate (MB)
        base_mem_mb = (batch_size * num_heads * seq_len * seq_len * 4) / (1024 * 1024)
        qgfd_mem_mb = base_mem_mb * (2.0 if mode == "full" else 1.05)

        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "mode": mode,
            "diffusion_steps": diffusion_steps,
            "baseline_latency_ms": round(baseline_ms, 3),
            "qgfd_latency_ms": round(qgfd_ms, 3),
            "latency_overhead_percent": round(((qgfd_ms - baseline_ms) / max(1e-5, baseline_ms)) * 100.0, 2),
            "baseline_gflops": round(base_flops / 1e9, 4),
            "qgfd_gflops": round(total_qgfd_flops / 1e9, 4),
            "estimated_vram_mb": round(qgfd_mem_mb, 2),
        }


def profile_qgfd_efficiency(
    batch_size: int = 4,
    seq_len: int = 512,
    embed_dim: int = 768,
    mode: str = "full",
) -> dict[str, float]:
    profiler = QGFDProfiler()
    return profiler.profile_layer(batch_size=batch_size, seq_len=seq_len, embed_dim=embed_dim, mode=mode)
