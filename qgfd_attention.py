import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadQGFDLayer(nn.Module):
    """
    Production-grade QGFD Multi-Head Attention.

    Key properties:
    - Drop-in replacement for standard multi-head attention.
    - No full L×L transition matrix P is ever constructed.
    - Adds a local, causal convolutional "diffusion" over the attention
      distribution along the key dimension.
    - Preserves causal + padding masks.
    - When target_alpha=0 or diffusion_steps=0, it exactly reduces to
      standard attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        proj_dim: Optional[int] = None,
        diffusion_steps: int = 1,
        target_alpha: float = 0.02,
        warmup_steps: int = 0,
        use_bias: bool = True,
        kernel_size: int = 5,
        early_stop_eps: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.proj_dim = int(proj_dim) if proj_dim is not None else int(embed_dim)
        assert self.proj_dim % self.num_heads == 0, "proj_dim must be divisible by num_heads"
        self.head_dim = self.proj_dim // self.num_heads

        self.diffusion_steps = int(diffusion_steps)
        self.target_alpha = float(target_alpha)
        self.warmup_steps = int(warmup_steps)
        self.early_stop_eps = float(early_stop_eps)

        # Warmup counter for alpha
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

        # Projections
        self.q_proj = nn.Linear(self.embed_dim, self.proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(self.embed_dim, self.proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.proj_dim, self.embed_dim, bias=use_bias)

        # Local smoothing kernel over the key dimension (conv on probs, not logits).
        assert kernel_size >= 1 and kernel_size % 2 == 1, "kernel_size must be odd >= 1"
        self.kernel_size = int(kernel_size)
        self.kernel = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1,  # causal; we trim the right side
            bias=False,
        )

        # Initialize kernel ≈ identity (no-op at start)
        with torch.no_grad():
            self.kernel.weight.zero_()
            center = self.kernel_size - 1  # causal: last position = "self"
            self.kernel.weight[0, 0, center] = 1.0

    # ----------------------------------------------------------------------
    # alpha / warmup
    # ----------------------------------------------------------------------
    def get_alpha(self) -> float:
        """
        Compute effective alpha with optional warmup.
        If target_alpha <= 0 or diffusion_steps <= 0, returns 0.
        """
        if self.target_alpha <= 0.0 or self.diffusion_steps <= 0:
            return 0.0

        if self.warmup_steps <= 0:
            return float(self.target_alpha)

        step = float(self.step_count.item())
        factor = min(1.0, step / float(self.warmup_steps))
        return float(self.target_alpha * factor)

    # ----------------------------------------------------------------------
    # attention mask helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _apply_attention_mask_to_scores(
        scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply attention mask to raw scores.

        attention_mask can be:
            - bool mask, (B, Lk) or (B, 1, 1, Lk); True = keep, False = mask.
            - 0/1 mask of shape (B, Lk)   (1=keep, 0=mask)
            - additive mask (B, 1, 1, Lk) (0=keep, negative=mask)
        """
        if attention_mask is None:
            return scores

        if attention_mask.dtype == torch.bool:
            additive = (~attention_mask).to(scores.dtype) * -1e9
        else:
            if (
                attention_mask.dim() == 2
                and attention_mask.max() <= 1.0
                and attention_mask.min() >= 0.0
            ):
                additive = (1.0 - attention_mask.to(scores.dtype)) * -1e9
            else:
                additive = attention_mask.to(scores.dtype)

        if additive.dim() == 2:
            additive = additive[:, None, None, :]  # (B,1,1,Lk)

        return scores + additive

    @staticmethod
    def _extract_key_valid_mask(
        attention_mask: Optional[torch.Tensor],
        Lk: int,
    ) -> Optional[torch.Tensor]:
        """
        Returns a (B, Lk) bool mask: True = key is valid (non-padding).
        Used to zero out smoothed probabilities on padding positions.
        """
        if attention_mask is None:
            return None

        if attention_mask.dtype == torch.bool:
            if attention_mask.dim() == 2:
                return attention_mask
            if attention_mask.dim() == 4:
                return attention_mask[:, 0, 0, :]

        if attention_mask.dim() == 2:
            if attention_mask.max() <= 1.0 and attention_mask.min() >= 0.0:
                return attention_mask > 0.5
            return attention_mask >= 0.0

        if attention_mask.dim() == 4:
            m = attention_mask[:, 0, 0, :]
            return m >= 0.0

        return None

    # ----------------------------------------------------------------------
    # local diffusion over attention probabilities
    # ----------------------------------------------------------------------
    def _local_diffusion_probs(
        self,
        p0: torch.Tensor,
        key_valid: Optional[torch.Tensor] = None,
        alpha: float = 0.0,
    ) -> torch.Tensor:
        """
        Perform local diffusion over attention probabilities along key dimension.

        p0: (B,H,Lq,Lk), each row sums to 1 and is already masked.
        key_valid: (B,Lk) bool or None (True = valid key; False = padding).
        alpha: mixing coefficient.
        """
        if alpha <= 0.0 or self.diffusion_steps <= 0:
            return p0

        B, H, Lq, Lk = p0.shape
        p = p0

        for _ in range(self.diffusion_steps):
            p_flat = p.view(B * H * Lq, 1, Lk)  # (B*H*Lq, 1, Lk)

            kernel = F.softmax(self.kernel.weight, dim=-1)  # (1,1,K)
            p_smooth = F.conv1d(p_flat, kernel, padding=self.kernel_size - 1)
            p_smooth = p_smooth[..., :Lk]
            p_smooth = p_smooth.view(B, H, Lq, Lk)

            if key_valid is not None:
                mask = key_valid[:, None, None, :]  # (B,1,1,Lk)
                p_smooth = p_smooth * mask

            denom = p_smooth.sum(dim=-1, keepdim=True) + 1e-9
            p_smooth = p_smooth / denom

            p_next = (1.0 - alpha) * p + alpha * p_smooth

            if self.early_stop_eps > 0.0:
                if torch.max(torch.abs(p_next - p)) < self.early_stop_eps:
                    p = p_next
                    break

            p = p_next

        return p

    # ----------------------------------------------------------------------
    # forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states : (B, Lq, D)
            kv            : (B, Lk, D) or None (self-attention if None).
            attention_mask: HF-style attention mask applied to QK scores.
            output_attentions: If True, returns final probabilities p.

        Returns:
            attn_out: (B, Lq, D)
            p       : (B, H, Lq, Lk) if output_attentions=True, else None
        """
        B, Lq, D = hidden_states.shape
        if kv is None:
            kv = hidden_states
        B2, Lk, D2 = kv.shape
        assert B2 == B and D2 == D, "hidden_states and kv must have same batch and dim"

        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Lq,hd)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Lk,hd)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Lk,hd)

        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(self.head_dim)
        scores = self._apply_attention_mask_to_scores(scores, attention_mask)

        p0 = F.softmax(scores, dim=-1)  # (B,H,Lq,Lk)
        key_valid = self._extract_key_valid_mask(attention_mask, Lk)
        alpha = float(self.get_alpha())

        p = self._local_diffusion_probs(p0, key_valid=key_valid, alpha=alpha)

        attn_out = torch.einsum("bhqk,bhkd->bhqd", p, V)
        attn_out = (
            attn_out.transpose(1, 2)
            .contiguous()
            .view(B, Lq, self.proj_dim)
        )
        attn_out = self.out_proj(attn_out).to(dtype)

        with torch.no_grad():
            self.step_count += 1
            if self.step_count.item() > 10**12:
                self.step_count.zero_()

        if output_attentions:
            return attn_out, p
        return attn_out, None
