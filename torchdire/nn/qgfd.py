import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadQGFDLayer(nn.Module):
    """
    Multi-head Query-Graph Flow Diffusion (QGFD) attention layer.

    QGFD extends standard scaled dot-product attention by modeling attention refinement
    as an iterative graph diffusion process over key similarities.

    Initial Attention:
        p^(0) = softmax(Q K^T / sqrt(d_k))

    Key Transition Matrix (Full Mode):
        P = softmax(K K^T / (sqrt(d_k) * temp))

    Iterative Update Rule:
        p^(t+1) = (1 - alpha) * p^(0) + alpha * (p^(t) @ P)

    Final Attention Output:
        h^(T) = p^(T) V
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        proj_dim: int | None = None,
        diffusion_steps: int = 4,
        target_alpha: float = 0.02,
        warmup_steps: int = 20000,
        use_bias: bool = True,
        early_stop_eps: float = 1e-5,
        detach_P: bool = False,
        temp: float = 1.0,
        mode: str = "full",
        kernel_size: int = 5,
        enable_qgfd: bool = True,
        max_alpha: float = 0.10,
        max_full_seq_len: int = 512,
        full_fallback_mode: str = "disable",
        mask_threshold: float = -1e4,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_dim = proj_dim if proj_dim is not None else embed_dim
        assert self.proj_dim % num_heads == 0, f"proj_dim ({self.proj_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = self.proj_dim // num_heads

        self.diffusion_steps = int(diffusion_steps)
        self.target_alpha = float(target_alpha)
        self.warmup_steps = int(warmup_steps)
        self.early_stop_eps = float(early_stop_eps)
        self.detach_P = bool(detach_P)
        self.temp = float(temp) if temp > 0.0 else 1.0

        assert mode in ("full", "conv"), f"mode must be 'full' or 'conv', got {mode}"
        self.mode = mode

        self.enable_qgfd = bool(enable_qgfd)
        self.max_alpha = float(max_alpha)
        self.max_full_seq_len = int(max_full_seq_len)
        assert full_fallback_mode in ("disable", "conv"), f"full_fallback_mode must be 'disable' or 'conv'"
        self.full_fallback_mode = full_fallback_mode
        self.mask_threshold = float(mask_threshold)
        self.debug = bool(debug)
        self.learnable_alpha = bool(kwargs.get("learnable_alpha", False))

        if self.learnable_alpha:
            self.alpha_param = nn.Parameter(torch.full((num_heads,), fill_value=float(target_alpha)))

        # Buffer to keep track of steps for warmup
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.proj_dim, embed_dim, bias=use_bias)

        # Conv mode parameter
        if self.mode == "conv" or self.full_fallback_mode == "conv":
            assert kernel_size >= 1 and kernel_size % 2 == 1, "kernel_size must be odd >= 1"
            self.kernel_size = kernel_size
            kernel = torch.ones(kernel_size, dtype=torch.float32)
            center = kernel_size // 2
            kernel[center] = 2.0
            kernel = kernel / kernel.sum()
            self.register_parameter(
                "conv_kernel",
                nn.Parameter(kernel.view(1, 1, kernel_size)),
            )

    def get_alpha(self) -> float | torch.Tensor:
        """Calculate effective alpha (scalar or per-head tensor) based on warmup step count and max_alpha bound."""
        if self.warmup_steps <= 0:
            factor = 1.0
        else:
            factor = min(1.0, float(self.step_count.item()) / float(self.warmup_steps))

        if self.learnable_alpha:
            alpha = self.alpha_param * factor
            return torch.clamp(alpha, -self.max_alpha, self.max_alpha).view(1, self.num_heads, 1, 1)

        alpha = self.target_alpha * factor
        return float(max(-self.max_alpha, min(self.max_alpha, alpha)))

    def build_transition_from_keys(self, K: torch.Tensor) -> torch.Tensor:
        """
        Build key-based row-stochastic transition matrix P from Key projections.

        Args:
            K: Key tensor of shape (B, H, Lk, head_dim)

        Returns:
            P: Row-stochastic transition matrix of shape (B, H, Lk, Lk)
        """
        K_norm = F.normalize(K, p=2, dim=-1, eps=self._eps(K))
        sim = torch.einsum("bhid,bhjd->bhij", K_norm, K_norm)
        sim = sim / max(1.0, math.sqrt(self.head_dim))
        sim = sim / self.temp

        P = F.softmax(sim, dim=-1)

        # Apply numerical jitter to prevent isolated zero-probability modes
        jitter = self._eps(P)
        P = P * (1.0 - jitter) + (jitter / P.size(-1))

        if self.detach_P:
            P = P.detach()

        return P

    def diffuse_via_conv(
        self,
        p0: torch.Tensor,
        attention_mask: torch.Tensor | None,
        alpha_eff: float,
    ) -> torch.Tensor:
        """Perform local 1D convolutional diffusion over the key sequence dimension."""
        B, H, Lq, Lk = p0.shape
        p = p0
        prev_p = None

        key_mask = self._build_key_mask(attention_mask, B, Lk)
        kernel = self.conv_kernel
        K = kernel.shape[-1]

        for _ in range(self.diffusion_steps):
            x = p.view(B * H * Lq, 1, Lk)
            x_padded = F.pad(x, (K - 1, 0))
            x_conv = F.conv1d(x_padded, kernel, groups=1)
            p_conv = x_conv.view(B, H, Lq, Lk)

            if key_mask is not None:
                p_conv = p_conv * key_mask.view(B, 1, 1, Lk).to(p_conv.dtype)

            p_conv = p_conv.clamp(min=self._eps(p_conv))
            Z = p_conv.sum(dim=-1, keepdim=True).clamp(min=self._eps(p_conv))
            p_conv = p_conv / Z

            p_next = (1.0 - alpha_eff) * p0 + alpha_eff * p_conv

            if prev_p is not None and torch.max(torch.abs(p_next - prev_p)) < self.early_stop_eps:
                p = p_next
                break

            prev_p = p
            p = p_next

        return p

    def apply_attention_mask(self, scores: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
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

    def _build_key_mask(self, attention_mask: torch.Tensor | None, B: int, Lk: int) -> torch.Tensor | None:
        if attention_mask is None:
            return None

        if attention_mask.dtype == torch.bool:
            am = attention_mask
            if am.dim() == 4:
                am = am[:, 0, 0, :]
            return am

        am = attention_mask
        if am.dim() == 4:
            am = am[:, 0, 0, :]

        if am.dim() != 2 or am.shape[0] != B or am.shape[1] != Lk:
            return None

        return ~(am <= self.mask_threshold)

    @staticmethod
    def _eps(x: torch.Tensor) -> float:
        return 1e-3 if x.dtype in (torch.float16, torch.bfloat16) else 1e-6

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs,
    ):
        B, Lq, D = hidden_states.shape
        if kv is None:
            kv = hidden_states
        Lk = kv.shape[1]

        Q = self.q_proj(hidden_states).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(self.head_dim)
        scores = self.apply_attention_mask(scores, attention_mask)
        p0 = F.softmax(scores, dim=-1)

        alpha_eff = self.get_alpha()
        alpha_nonzero = (alpha_eff != 0.0).any().item() if isinstance(alpha_eff, torch.Tensor) else abs(alpha_eff) > 0.0
        qgfd_active = self.enable_qgfd and self.diffusion_steps > 0 and alpha_nonzero

        if not qgfd_active:
            p = p0
        else:
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
                P = self.build_transition_from_keys(K)
                p = p0
                prev_p = None

                for _ in range(self.diffusion_steps):
                    p_next = (1.0 - alpha_eff) * p0 + alpha_eff * torch.einsum("bhqn,bhnm->bhqm", p, P)

                    if prev_p is not None and torch.max(torch.abs(p_next - prev_p)) < self.early_stop_eps:
                        p = p_next
                        break

                    prev_p = p
                    p = p_next
            else:
                p = self.diffuse_via_conv(p0, attention_mask, alpha_eff)

        if head_mask is not None:
            p = p * head_mask.view(1, -1, 1, 1)

        attn_output_raw = torch.einsum("bhqk,bhkd->bhqd", p, V)
        attn_output_raw = attn_output_raw.transpose(1, 2).contiguous().view(B, Lq, self.proj_dim)
        attn_output = self.out_proj(attn_output_raw)

        if self.training:
            self.step_count += 1
            if self.step_count.item() > 10**12:
                self.step_count.zero_()

        if output_attentions:
            return (attn_output, p)
        return (attn_output,)
