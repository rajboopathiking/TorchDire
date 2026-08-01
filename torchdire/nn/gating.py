import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QGFDMultiHeadAttention(nn.Module):
    """
    Query-Guided Feature Distribution (QGFD-Gating) attention module.

    As documented in research comparison reports, this module modulates attention
    logits using a single-pass query-dependent gating network:
        scores = Q K^T / sqrt(d_k)
        gates = sigmoid(Linear(ReLU(Linear(q_flat))))
        scores = scores * (1.0 + gates)
        attn_output = softmax(scores) V
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        proj_dim: int | None = None,
        use_bias: bool = True,
        gate_hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_dim = proj_dim if proj_dim is not None else embed_dim
        assert self.proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"
        self.head_dim = self.proj_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.proj_dim, embed_dim, bias=use_bias)

        # Gating network
        self.gate_net = nn.Sequential(
            nn.Linear(self.head_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
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

        # Compute query-dependent modulation gates
        gates = self.gate_net(Q)  # (B, H, Lq, 1)
        scores = scores * (1.0 + gates)

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                additive = (~attention_mask).to(scores.dtype) * -1e9
            else:
                additive = attention_mask.to(scores.dtype)
            if additive.dim() == 2:
                additive = additive[:, None, None, :]
            scores = scores + additive

        p = F.softmax(scores, dim=-1)

        attn_output_raw = torch.einsum("bhqk,bhkd->bhqd", p, V)
        attn_output_raw = attn_output_raw.transpose(1, 2).contiguous().view(B, Lq, self.proj_dim)
        attn_output = self.out_proj(attn_output_raw)

        if output_attentions:
            return (attn_output, p)
        return (attn_output,)
