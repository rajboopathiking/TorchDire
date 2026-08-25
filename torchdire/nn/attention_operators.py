"""
Attention Probability Operators and Architecture-Specific Adapters.

This module implements the operator-based approach for replacing only the
attention probability computation (softmax) while preserving all other
attention mechanics (projections, RoPE, KV cache, GQA, causal masking).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union


class AttentionProbabilityOperator(nn.Module, ABC):
    """
    Base class for attention probability operators.

    Replaces the standard softmax(attention_scores) computation with a
    custom operator while preserving the rest of the attention mechanism.
    """

    @abstractmethod
    def forward(
        self,
        scores: torch.Tensor,
        key_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute attention probabilities from raw scores.

        Args:
            scores: (B, H, Lq, Lk) raw attention scores (QK^T / sqrt(d))
            key_states: (B, H_k, Lk, head_dim) key projections, needed for
                operators that use key-key interactions (e.g., QGFD)
            attention_mask: Optional mask for padding/causal attention
            head_mask: Optional per-head mask

        Returns:
            attention_probs: (B, H, Lq, Lk) attention probabilities
        """
        pass


class SoftmaxOperator(AttentionProbabilityOperator):
    """Standard softmax attention probability operator."""

    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.dtype = dtype

    def forward(
        self,
        scores: torch.Tensor,
        key_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attention_mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + attention_mask.to(scores.dtype)

        probs = F.softmax(scores.to(self.dtype), dim=-1)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            probs = probs * head_mask

        return probs.to(scores.dtype)


class QGFDOperator(AttentionProbabilityOperator):
    """
    Query-Graph Flow Diffusion (QGFD) attention probability operator.

    Implements the diffusion-refined attention operator:
    
    p_0 = softmax(S)
    P = softmax(KK^T / sqrt(d))
    p_{t+1} = (1 - alpha) * p_0 + alpha * (p_t @ P)
    
    This REPLACES the final attention-weight transformation with a 
    diffusion-refined attention operator. Note that it still uses softmax
    internally for both p_0 and P - it does not "eliminate softmax entirely".
    """

    def __init__(
        self,
        diffusion_steps: int = 1,
        target_alpha: float = 0.05,
        warmup_steps: int = 0,
        early_stop_eps: float = 0.0,
        detach_P: bool = True,
        temp: float = 1.0,
        mode: str = "full",
        kernel_size: int = 5,
        enable_qgfd: bool = True,
        max_alpha: float = 0.10,
        max_full_seq_len: int = 512,
        full_fallback_mode: str = "conv",
        mask_threshold: float = -1e4,
        debug: bool = False,
        learnable_alpha: bool = False,
        num_heads: Optional[int] = None,
        is_causal: bool = False,
    ):
        super().__init__()
        assert mode in ("full", "conv"), f"mode must be 'full' or 'conv', got {mode}"
        assert full_fallback_mode in ("disable", "conv"), f"full_fallback_mode must be 'disable' or 'conv'"

        self.diffusion_steps = int(diffusion_steps)
        self.target_alpha = float(target_alpha)
        self.warmup_steps = int(warmup_steps)
        self.early_stop_eps = float(early_stop_eps)
        self.detach_P = bool(detach_P)
        self.temp = float(temp) if temp > 0.0 else 1.0
        self.mode = mode
        self.enable_qgfd = bool(enable_qgfd)
        self.max_alpha = float(max_alpha)
        self.max_full_seq_len = int(max_full_seq_len)
        self.full_fallback_mode = full_fallback_mode
        self.mask_threshold = float(mask_threshold)
        self.debug = bool(debug)
        self.learnable_alpha = bool(learnable_alpha)
        self.num_heads = num_heads
        self.is_causal = bool(is_causal)

        if self.learnable_alpha:
            assert num_heads is not None, "num_heads must be provided when learnable_alpha=True"
            self.alpha_param = nn.Parameter(torch.full((num_heads,), fill_value=float(target_alpha)))

        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

        if self.mode == "conv" or self.full_fallback_mode == "conv":
            assert kernel_size >= 1 and kernel_size % 2 == 1, "kernel_size must be odd >= 1"
            self.kernel_size = kernel_size
            kernel = torch.ones(kernel_size, dtype=torch.float32)
            center = kernel_size // 2
            kernel[center] = 2.0
            kernel = kernel / kernel.sum()
            self.register_parameter(
                "conv_kernel",
                nn.Parameter(kernel.view(1, 1, kernel_size), requires_grad=(self.mode == "conv")),
            )

    def set_step(self, step: int) -> None:
        """Externally set the global training step driving the alpha warmup."""
        self.step_count.fill_(int(step))

    def get_alpha(self) -> Union[float, torch.Tensor]:
        """Calculate effective alpha based on step_count and max_alpha bound."""
        if not self.training:
            if self.learnable_alpha:
                return torch.clamp(self.alpha_param, -self.max_alpha, self.max_alpha).view(1, -1, 1, 1)
            return float(max(-self.max_alpha, min(self.max_alpha, self.target_alpha)))

        if self.warmup_steps <= 0:
            factor = 1.0
        else:
            factor = min(1.0, float(self.step_count.item()) / float(self.warmup_steps))

        if self.learnable_alpha:
            alpha = self.alpha_param * factor
            return torch.clamp(alpha, -self.max_alpha, self.max_alpha).view(1, -1, 1, 1)

        alpha = self.target_alpha * factor
        return float(max(-self.max_alpha, min(self.max_alpha, alpha)))

    @staticmethod
    def _eps(x: torch.Tensor) -> float:
        return 1e-3 if x.dtype in (torch.float16, torch.bfloat16) else 1e-6

    def build_transition_from_keys(
        self,
        K: torch.Tensor,
        target_heads: Optional[int] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Build key-based row-stochastic transition matrix P from key projections.
        
        Args:
            K: (B, H_k, Lk, head_dim)
            target_heads: Number of query heads H (repeats K heads if GQA/MQA)
            is_causal: If True, applies lower-triangular causal masking to P.
        Returns:
            P: (B, H, Lk, Lk) transition matrix
        """
        if is_causal is None:
            is_causal = getattr(self, "is_causal", False)
        B, H_k, Lk, head_dim = K.shape

        K_norm = F.normalize(K, p=2, dim=-1, eps=self._eps(K))
        sim = torch.matmul(K_norm, K_norm.transpose(-1, -2))

        if is_causal:
            causal_mask = torch.tril(torch.ones((Lk, Lk), device=K.device, dtype=torch.bool))
            sim = sim.masked_fill(~causal_mask[None, None, :, :], torch.finfo(sim.dtype).min)

        P = F.softmax(sim / self.temp, dim=-1)

        if Lk > 1:
            P = P.clone()
            P[:, :, 0, :] = 0.0
            P[:, :, 0, 0] = 1.0

        jitter = self._eps(P)
        P = P * (1.0 - jitter) + (jitter / P.size(-1))

        if target_heads is not None and H_k != target_heads:
            repeat_factor = target_heads // H_k
            P = torch.repeat_interleave(P, repeat_factor, dim=1)

        if self.detach_P:
            P = P.detach()

        return P

    def diffuse_via_conv(
        self,
        p0: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        alpha_eff: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Perform 1D local conv-based diffusion over the key sequence dimension."""
        B, H, Lq, Lk = p0.shape
        p = p0
        prev_p = None

        if not hasattr(self, "conv_kernel") or self.conv_kernel is None:
            kernel_size = getattr(self, "kernel_size", 5)
            kernel = torch.ones(kernel_size, dtype=torch.float32)
            center = kernel_size // 2
            kernel[center] = 2.0
            kernel = kernel / kernel.sum()
            self.register_parameter(
                "conv_kernel",
                nn.Parameter(kernel.view(1, 1, kernel_size).to(device=p0.device, dtype=p0.dtype)),
            )

        kernel = self.conv_kernel.to(device=p0.device, dtype=p0.dtype)
        K_size = kernel.shape[-1]

        for _ in range(self.diffusion_steps):
            x = p.view(B * H * Lq, 1, Lk)
            x_padded = F.pad(x, (K_size - 1, 0))
            x_conv = F.conv1d(x_padded, kernel, groups=1)
            p_conv = x_conv.view(B, H, Lq, Lk)

            p_conv = p_conv.clamp(min=self._eps(p_conv))
            if valid_mask is not None:
                p_conv = p_conv * valid_mask.to(p_conv.dtype)

            Z = p_conv.sum(dim=-1, keepdim=True).clamp(min=self._eps(p_conv))
            p_conv = p_conv / Z

            p_next = (1.0 - alpha_eff) * p0 + alpha_eff * p_conv
            if valid_mask is not None:
                p_next = p_next * valid_mask.to(p_next.dtype)
            Z = p_next.sum(dim=-1, keepdim=True).clamp(min=self._eps(p_next))
            p_next = p_next / Z

            if prev_p is not None and torch.max(torch.abs(p_next - prev_p)) < self.early_stop_eps:
                p = p_next
                break

            prev_p = p
            p = p_next

        return p

    def apply_attention_mask(self, scores: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply boolean or additive attention mask to raw score logits."""
        if attention_mask is None:
            return scores

        if attention_mask.dtype == torch.bool:
            additive = (~attention_mask).to(scores.dtype) * -1e9
        else:
            additive = attention_mask.to(scores.dtype)

        if additive.dim() == 2:
            additive = additive[:, None, None, :]

        return scores + additive

    def _build_valid_mask(
        self,
        scores: torch.Tensor,
        p0: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                valid = attention_mask
            else:
                valid = attention_mask > -1e4
            if valid.all():
                return None
            return valid

        mask = (scores > self.mask_threshold) & (p0 > 1e-12)
        if mask.all():
            return None
        return mask

    def forward(
        self,
        scores: torch.Tensor,
        key_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and not torch.is_grad_enabled() and not hasattr(self, "_warned_training_mode"):
            import warnings
            warnings.warn(
                "QGFDOperator.forward called with module.training=True. If you're doing "
                "inference/generation, call model.eval() first — dropout and the alpha "
                "warmup schedule are both training-mode-dependent.",
                stacklevel=2,
            )
            self._warned_training_mode = True

        assert scores.dim() == 4, f"QGFDOperator expected scores [B,H,Q,K], got {scores.shape}"
        B, H, Lq, Lk = scores.shape

        if attention_mask is not None:
            scores = self.apply_attention_mask(scores, attention_mask)

        p0 = F.softmax(scores.to(torch.float32), dim=-1)

        alpha_eff = self.get_alpha()
        alpha_nonzero = (
            (alpha_eff != 0.0).any().item()
            if isinstance(alpha_eff, torch.Tensor)
            else abs(alpha_eff) > 0.0
        )
        qgfd_active = self.enable_qgfd and self.diffusion_steps > 0 and alpha_nonzero

        if not qgfd_active:
            p = p0
        else:
            valid_mask = self._build_valid_mask(scores, p0, attention_mask)
            mode = self.mode
            if mode == "full" and Lk > self.max_full_seq_len:
                if self.full_fallback_mode == "conv":
                    mode = "conv"
                else:
                    p = p0
                    mode = None

            if mode is None:
                p = p0
            elif mode == "full":
                if key_states is None:
                    raise ValueError("QGFDOperator in 'full' mode requires key_states")
                P = self.build_transition_from_keys(key_states, target_heads=H, is_causal=self.is_causal)

                if self.diffusion_steps == 1 and valid_mask is None and self.early_stop_eps <= 0.0:
                    p = (1.0 - alpha_eff) * p0 + alpha_eff * torch.matmul(p0, P)
                else:
                    p = p0
                    prev_p = None

                    for _ in range(self.diffusion_steps):
                        p_next = (1.0 - alpha_eff) * p0 + alpha_eff * torch.matmul(p, P)

                        if valid_mask is not None:
                            p_next = p_next * valid_mask.to(p_next.dtype)
                        Z = p_next.sum(dim=-1, keepdim=True).clamp(min=self._eps(p_next))
                        p_next = p_next / Z

                        if prev_p is not None and self.early_stop_eps > 0.0 and torch.max(torch.abs(p_next - prev_p)) < self.early_stop_eps:
                            p = p_next
                            break

                        prev_p = p
                        p = p_next
            else:
                p = self.diffuse_via_conv(p0, valid_mask, alpha_eff)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            p = p * head_mask

        assert p.dim() == 4 and p.shape == (B, H, Lq, Lk), (
            f"QGFDOperator changed attention shape: expected {(B, H, Lq, Lk)}, got {p.shape}"
        )

        if self.debug:
            print("[QGFDOperator OUT]", "P=", tuple(p.shape), "alpha=", alpha_eff)

        return p.to(scores.dtype)


def collect_qgfd_operators(model) -> list:
    """Return every QGFDOperator instance found in `model`."""
    operators = []
    for module in model.modules():
        if isinstance(module, QGFDOperator):
            operators.append(module)
    return operators


def register_qgfd_operator_step_callback(trainer, model):
    """Register a step callback on `trainer` for all QGFDOperators in `model`."""
    try:
        from transformers import TrainerCallback
    except ImportError:
        raise ImportError("register_qgfd_operator_step_callback requires `transformers` (TrainerCallback).")

    operators = collect_qgfd_operators(model)
    if not operators:
        return None

    class QGFDOperatorStepCallback(TrainerCallback):
        def __init__(self, operators):
            self.operators = list(operators)

        def on_step_begin(self, args, state, control, **kwargs):
            step = int(getattr(state, "global_step", 0))
            for op in self.operators:
                op.set_step(step)

        def on_optimizer_step(self, args, state, control, **kwargs):
            step = int(getattr(state, "global_step", 0))
            for op in self.operators:
                op.set_step(step)

    callback = QGFDOperatorStepCallback(operators)
    trainer.add_callback(callback)
    return callback