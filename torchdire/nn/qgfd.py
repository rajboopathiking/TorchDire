import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdire.nn.qgfd_kernel import QGFDKernel


class MultiHeadQGFDLayer(nn.Module):
    """
    Multi-head Query-Graph Flow Diffusion (QGFD) attention layer.

    Refactored to delegate diffusion computation to QGFDKernel, keeping linear projections
    and module interface intact.
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

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.proj_dim, embed_dim, bias=use_bias)

        # Delegate diffusion algorithm to QGFDKernel
        self.kernel = QGFDKernel(
            diffusion_steps=diffusion_steps,
            target_alpha=target_alpha,
            warmup_steps=warmup_steps,
            early_stop_eps=early_stop_eps,
            detach_P=detach_P,
            temp=temp,
            mode=mode,
            kernel_size=kernel_size,
            enable_qgfd=enable_qgfd,
            max_alpha=max_alpha,
            max_full_seq_len=max_full_seq_len,
            full_fallback_mode=full_fallback_mode,
            mask_threshold=mask_threshold,
            debug=debug,
            num_heads=num_heads,
            **kwargs,
        )

    # Forward legacy properties/methods to kernel for compatibility
    @property
    def step_count(self):
        return self.kernel.step_count

    @property
    def diffusion_steps(self):
        return self.kernel.diffusion_steps

    @property
    def mode(self):
        return self.kernel.mode

    @property
    def target_alpha(self):
        return self.kernel.target_alpha

    def get_alpha(self):
        return self.kernel.get_alpha()

    def build_transition_from_keys(self, K: torch.Tensor) -> torch.Tensor:
        return self.kernel.build_transition_from_keys(K, target_heads=self.num_heads)

    def diffuse_via_conv(self, p0: torch.Tensor, attention_mask: torch.Tensor | None, alpha_eff: float) -> torch.Tensor:
        valid_mask = self.kernel._build_valid_mask(p0, p0)
        return self.kernel.diffuse_via_conv(p0, valid_mask, alpha_eff)

    def apply_attention_mask(self, scores: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        return self.kernel.apply_attention_mask(scores, attention_mask)

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

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)

        p = self.kernel(
            scores=scores,
            key_states=K,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        attn_output_raw = torch.matmul(p, V)
        attn_output_raw = attn_output_raw.transpose(1, 2).contiguous().view(B, Lq, self.proj_dim)
        attn_output = self.out_proj(attn_output_raw)

        if output_attentions:
            return (attn_output, p)
        return (attn_output,)
