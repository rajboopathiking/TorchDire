"""
TorchDire Fused QGFD Kernels (Triton & PyTorch Vectorized Fallback)
===================================================================
Fused GPU kernel for O(N * K) local convolutional diffusion over attention weights,
eliminating intermediate global memory allocations and materialization of p0.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_conv_diffusion_kernel(
        P_ptr,          # Pointer to attention weights (B * H * Lq, Lk)
        Out_ptr,        # Pointer to diffused attention weights
        Kernel_ptr,     # Pointer to 1D conv kernel (K,)
        alpha,          # Diffusion mixing factor
        L_key,          # Key sequence length
        K_size,         # Kernel size (odd)
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        k_center = K_size // 2
        
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < L_key
        
        # Load baseline attention weights p0 for query row pid
        p0 = tl.load(P_ptr + pid * L_key + cols, mask=mask, other=0.0)
        
        # Compute local 1D causal convolution over L_key
        conv_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        for k_idx in range(K_size):
            shift = k_idx - (K_size - 1)  # Causal shift
            source_cols = cols + shift
            valid_mask = mask & (source_cols >= 0) & (source_cols < L_key)
            val = tl.load(P_ptr + pid * L_key + source_cols, mask=valid_mask, other=0.0)
            weight = tl.load(Kernel_ptr + k_idx)
            conv_acc += val * weight

        # Renormalize conv_acc
        norm_factor = tl.sum(conv_acc, axis=0) + 1e-6
        conv_norm = conv_acc / norm_factor
        
        # Convex mixture: p_diffused = (1 - alpha) * p0 + alpha * conv_norm
        p_out = (1.0 - alpha) * p0 + alpha * conv_norm
        
        tl.store(Out_ptr + pid * L_key + cols, p_out, mask=mask)


def fused_conv_diffusion(
    p0: torch.Tensor,
    kernel: torch.Tensor,
    alpha: float,
    steps: int = 1,
) -> torch.Tensor:
    """
    Fused O(N * K) local convolutional diffusion operator.
    
    Args:
        p0: Baseline attention probability tensor (B, H, Lq, Lk).
        kernel: 1D convolution kernel tensor (1, 1, K).
        alpha: Diffusion mixing factor.
        steps: Number of diffusion steps.
        
    Returns:
        p: Diffused attention probability tensor (B, H, Lq, Lk).
    """
    B, H, Lq, Lk = p0.shape
    device = p0.device
    
    if TRITON_AVAILABLE and device.type == "cuda" and p0.is_contiguous():
        N_rows = B * H * Lq
        K_size = kernel.shape[-1]
        p_curr = p0.view(N_rows, Lk)
        p_out = torch.empty_like(p_curr)
        k_flat = kernel.view(-1).to(dtype=torch.float32)
        
        BLOCK_SIZE = triton.next_power_of_two(Lk)
        
        for _ in range(steps):
            grid = (N_rows,)
            _fused_conv_diffusion_kernel[grid](
                p_curr,
                p_out,
                k_flat,
                float(alpha),
                Lk,
                K_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            p_curr = p_out
            
        return p_out.view(B, H, Lq, Lk)
    else:
        # Fallback PyTorch implementation
        p = p0
        kernel = kernel.to(dtype=p0.dtype)
        K = kernel.shape[-1]
        for _ in range(steps):
            x = p.view(B * H * Lq, 1, Lk)
            x_padded = F.pad(x, (K - 1, 0))
            x_conv = F.conv1d(x_padded, kernel, groups=1).view(B, H, Lq, Lk)
            x_conv = x_conv.clamp(min=1e-6)
            Z = x_conv.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            x_conv = x_conv / Z
            p = (1.0 - alpha) * p0 + alpha * x_conv
        return p
