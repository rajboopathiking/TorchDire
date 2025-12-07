import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadQGFDLayer(nn.Module):
    """
    Multi-head QGFD attention layer.

    Args:
        embed_dim: Dimensionality of input embeddings.
        num_heads: Number of attention heads.
        proj_dim: Dimensionality of projected Q/K/V vectors (defaults to embed_dim).
        diffusion_steps: Number of diffusion steps to apply.
        target_alpha: Target diffusion mixing factor α.
        warmup_steps: Steps to linearly ramp up α (for stable training).
        use_bias: Whether to use bias in linear projections.
        early_stop_eps: Threshold to stop diffusion early if updates are small.
        detach_P: Whether to detach transition matrix P from autograd.
        temp: Temperature scaling for the key-based transition softmax.
    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        proj_dim=None,
        diffusion_steps=4,
        target_alpha=0.02,
        warmup_steps=20000,
        use_bias=True,
        early_stop_eps=1e-5,
        detach_P=False,
        temp=1.0,
        **kwargs
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_dim = proj_dim if proj_dim is not None else embed_dim
        assert self.proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"
        self.head_dim = self.proj_dim // num_heads

        self.diffusion_steps = int(diffusion_steps)
        self.target_alpha = float(target_alpha)
        self.warmup_steps = int(warmup_steps)
        self.early_stop_eps = float(early_stop_eps)
        self.detach_P = bool(detach_P)
        self.temp = float(temp) if temp > 0.0 else 1.0

        # used for alpha warmup (and eval-time control)
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

        # projections
        self.q_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.proj_dim, embed_dim, bias=use_bias)

    # ------------ alpha schedule ------------
    def get_alpha(self) -> float:
        """
        Linear warmup from 0 to target_alpha over warmup_steps.
        For evaluation you can set warmup_steps=0 to use target_alpha directly.
        """
        if self.warmup_steps <= 0:
            return float(self.target_alpha)
        factor = min(1.0, float(self.step_count.item()) / float(self.warmup_steps))
        return float(self.target_alpha * factor)

    # ------------ key-based transition matrix ------------
    def build_transition_from_keys(self, K):
        """
        Build key-based transition matrix P from keys.

        Args:
            K: (B, H, Lk, head_dim)

        Returns:
            P: (B, H, Lk, Lk) transition matrix based on cosine similarities.
        """
        # normalize last dim
        K_norm = F.normalize(K, p=2, dim=-1, eps=1e-6)  # (B,H,Lk,head_dim)

        # cosine similarity: (B,H,Lk,Lk)
        sim = torch.einsum("bhid,bhjd->bhij", K_norm, K_norm)

        # scale and temperature
        sim = sim / max(1.0, math.sqrt(self.head_dim))
        sim = sim / self.temp

        # softmax to get transitions
        P = F.softmax(sim, dim=-1)

        # tiny jitter for numerical stability
        P = P * (1.0 - 1e-6) + (1e-6 / P.size(-1))

        if self.detach_P:
            P = P.detach()

        return P

    # ------------ mask handling ------------
    def apply_attention_mask(self, scores, attention_mask):
        """
        attention_mask: can be
            - (B, Lk)
            - (B, 1, 1, Lk)
            - bool mask
            - additive mask already in logits space
        """
        if attention_mask is None:
            return scores

        # bool mask: True = keep, False = mask
        if attention_mask.dtype == torch.bool:
            additive = (~attention_mask).to(scores.dtype) * -1e9
        else:
            additive = attention_mask.to(scores.dtype)

        if additive.dim() == 2:
            additive = additive[:, None, None, :]  # (B,1,1,Lk)

        return scores + additive

    # ------------ forward ------------
    def forward(
        self,
        hidden_states,
        kv=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        """
        Args:
            hidden_states: (B, Lq, D)
            kv: optional key/value input (B, Lk, D). If None, self-attn on hidden_states.
            attention_mask: HF-style attention mask.
            head_mask: optional per-head scaling.
            output_attentions: whether to return attention probabilities.

        Returns:
            If output_attentions:
                (attn_output, p)
            else:
                (attn_output,)
        """
        B, Lq, D = hidden_states.shape
        if kv is None:
            kv = hidden_states
        Lk = kv.shape[1]

        # project and reshape to (B, H, L, head_dim)
        Q = (
            self.q_proj(hidden_states)
            .view(B, Lq, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B,H,Lq,hd)
        K = (
            self.k_proj(kv)
            .view(B, Lk, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B,H,Lk,hd)
        V = (
            self.v_proj(kv)
            .view(B, Lk, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B,H,Lk,hd)

        # scaled dot-product scores (before softmax)
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(self.head_dim)

        # apply attention mask
        scores = self.apply_attention_mask(scores, attention_mask)

        # baseline softmax attention distribution
        p0 = F.softmax(scores, dim=-1)  # (B,H,Lq,Lk)

        # diffusion
        alpha_eff = float(self.get_alpha())
        if alpha_eff <= 0.0 or self.diffusion_steps <= 0:
            p = p0
        else:
            P = self.build_transition_from_keys(K)  # (B,H,Lk,Lk)
            p = p0
            prev_p = None

            for _ in range(self.diffusion_steps):
                # p_{t+1} = (1 - alpha) * p0 + alpha * (p_t @ P)
                p_next = (1.0 - alpha_eff) * p0 + alpha_eff * torch.einsum(
                    "bhqn,bhnm->bhqm", p, P
                )

                # early stop on small change
                if (
                    prev_p is not None
                    and torch.max(torch.abs(p_next - prev_p)) < self.early_stop_eps
                ):
                    p = p_next
                    break

                prev_p = p
                p = p_next

        # optional per-head scaling
        if head_mask is not None:
            p = p * head_mask.view(1, -1, 1, 1)

        # attention output
        attn_output_raw = torch.einsum("bhqk,bhkd->bhqd", p, V)  # (B,H,Lq,hd)
        attn_output_raw = (
            attn_output_raw.transpose(1, 2)
            .contiguous()
            .view(B, Lq, self.proj_dim)
        )
        attn_output = self.out_proj(attn_output_raw)  # final projection back to embed_dim

        # increment step counter with overflow protection
        self.step_count += 1
        if self.step_count.item() > 10**12:
            self.step_count.zero_()

        if output_attentions:
            return (attn_output, p)
        return (attn_output,)
