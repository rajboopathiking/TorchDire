import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


# ---------------- QGFD layer ----------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class QGFDLayer(nn.Module):
    def __init__(self, d_model, alpha=0.5, T=5, eps=1e-5):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.alpha = alpha
        self.T = T
        self.eps = eps

    def forward(self, x, mask=None):
        Q = self.q_proj(x)    # [B, L, D]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # affinity / compatibility (same as attention logits)
        logits = torch.matmul(Q, K.transpose(-2,-1)) / (Q.size(-1)**0.5)
        if mask is not None:
            logits = logits.masked_fill(mask==0, -1e9)
        A = F.softmax(logits, dim=-1)   # stochastic matrix

        # initialize p^(0) as one-step attention output (or uniform)
        p = torch.matmul(A, V)  # [B, L, D]

        # fixed-point iterations: p <- (1-alpha)*p + alpha * A p
        for t in range(1, self.T):
            p_next = (1-self.alpha) * p + self.alpha * torch.matmul(A, p)
            if torch.norm(p_next - p) < self.eps:
                p = p_next
                break
            p = p_next

        return self.out(p)

# ---------------- Stable QGFD ---------------- #


class StableQGFDLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, diffusion_steps=1, alpha=0.05, use_bias=True, detach_P=True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.proj_dim = self.head_dim * num_heads

        self.diffusion_steps = diffusion_steps
        self.alpha = alpha
        self.detach_P = detach_P

        self.q_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.proj_dim, embed_dim, bias=use_bias)

    def build_transition_from_keys(self, K):
        sim = torch.einsum("bhid,bhjd->bhij", K, K) / math.sqrt(self.embed_dim)
        P = F.softmax(sim, dim=-1)
        if self.detach_P:
            P = P.detach()
        return P

    def forward(self, hidden_states, kv=None, attention_mask=None):
        B, N, D = hidden_states.shape
        if kv is None:
            kv = hidden_states
        K_len = kv.shape[1]

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        p = F.softmax(scores, dim=-1)
        p0 = p.clone()

        P = self.build_transition_from_keys(K)

        for _ in range(self.diffusion_steps):
            p = (1 - self.alpha) * p0 + self.alpha * torch.einsum("bhqn,bhnm->bhqm", p, P)

        attn_output = torch.einsum("bhqk,bhkd->bhqd", p, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.proj_dim)
        return self.out_proj(attn_output), p

# ---------------- Multi Head QGFD ---------------- #

class MultiHeadQGFDLayer(nn.Module):
    """
    Multi-Head Query-Guided Feature Diffusion (QGFD)
    Drop-in replacement for MultiHeadAttention
    """
    def __init__(self, embed_dim, num_heads=8, proj_dim=None,
                 diffusion_steps=2, alpha=0.3, use_bias=True,
                 early_stop_eps=1e-4, warmup_steps=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.proj_dim = proj_dim if proj_dim is not None else embed_dim
        assert self.proj_dim % num_heads == 0, "proj_dim must be divisible by num_heads"
        self.head_dim = self.proj_dim // num_heads

        # Diffusion settings
        self.diffusion_steps = diffusion_steps
        self.alpha = alpha
        self.early_stop_eps = early_stop_eps
        self.warmup_steps = warmup_steps
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

        # Linear projections
        self.q_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.proj_dim, embed_dim, bias=use_bias)

    def build_transition_from_keys(self, K):
        """
        Transition matrix P from key similarity
        Shape: [B, H, N, N]
        """
        sim = torch.einsum("bhid,bhjd->bhij", K, K) / math.sqrt(self.head_dim)
        P = F.softmax(sim, dim=-1)
        return P

    def get_alpha(self):
        """Warmup schedule for alpha"""
        if self.warmup_steps > 0:
            factor = min(1.0, self.step_count.item() / self.warmup_steps)
        else:
            factor = 1.0
        return self.alpha * factor

    def forward(self, hidden_states, kv=None, attention_mask=None, past_key_values=None, use_cache=False):
        B, N, D = hidden_states.shape
        if kv is None:
            kv = hidden_states
        K_len = kv.shape[1]

        # Project Q, K, V
        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention logits
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (math.sqrt(self.head_dim) + 1e-8)
        if attention_mask is not None:
            scores = scores + attention_mask
        p = F.softmax(scores, dim=-1)
        p0 = p.clone()

        # Build diffusion transition matrix
        P = self.build_transition_from_keys(K)

        # Diffusion iterations
        prev_p = None
        alpha_eff = self.get_alpha()
        for step in range(self.diffusion_steps):
            p = (1 - alpha_eff) * p0 + alpha_eff * torch.einsum("bhqn,bhnm->bhqm", p, P)
            if prev_p is not None and torch.max(torch.abs(p - prev_p)) < self.early_stop_eps:
                break
            prev_p = p.clone()

        # Weighted sum of values
        attn_output = torch.einsum("bhqk,bhkd->bhqd", p, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.proj_dim)
        attn_output = self.out_proj(attn_output)

        # Update step counter (for alpha warmup)
        self.step_count += 1

        return attn_output, p

# ---------------- Transformer Blocks ---------------- #

class QGFDTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim,
                 diffusion_steps=2, alpha=0.3, warmup_steps=0):
        super().__init__()
        self.self_attn = MultiHeadQGFDLayer(embed_dim, num_heads,
                                            diffusion_steps=diffusion_steps,
                                            alpha=alpha,
                                            warmup_steps=warmup_steps)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, attention_mask=None):
        attn_output, _ = self.self_attn(src, attention_mask=attention_mask)
        src = self.norm1(src + attn_output)
        ffn_output = self.ffn(src)
        src = self.norm2(src + ffn_output)
        return src


class QGFDTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim,
                 diffusion_steps=2, alpha=0.3, warmup_steps=0):
        super().__init__()
        self.self_attn = MultiHeadQGFDLayer(embed_dim, num_heads,
                                            diffusion_steps=diffusion_steps,
                                            alpha=alpha,
                                            warmup_steps=warmup_steps)
        self.cross_attn = MultiHeadQGFDLayer(embed_dim, num_heads,
                                             diffusion_steps=diffusion_steps,
                                             alpha=alpha,
                                             warmup_steps=warmup_steps)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, tgt, enc_out, self_attention_mask=None, cross_attention_mask=None):
        attn_output, _ = self.self_attn(tgt, attention_mask=self_attention_mask)
        tgt = self.norm1(tgt + attn_output)

        attn_output, _ = self.cross_attn(tgt, kv=enc_out, attention_mask=cross_attention_mask)
        tgt = self.norm2(tgt + attn_output)

        ffn_output = self.ffn(tgt)
        tgt = self.norm3(tgt + ffn_output)
        return tgt


class QGFDTransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim,
                 diffusion_steps=2, alpha=0.3, warmup_steps=0):
        super().__init__()
        self.layers = nn.ModuleList([
            QGFDTransformerEncoderLayer(embed_dim, num_heads, hidden_dim,
                                        diffusion_steps, alpha, warmup_steps)
            for _ in range(num_layers)
        ])

    def forward(self, src, attention_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, attention_mask=attention_mask)
        return output


class QGFDTransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, hidden_dim,
                 diffusion_steps=2, alpha=0.3, warmup_steps=0):
        super().__init__()
        self.layers = nn.ModuleList([
            QGFDTransformerDecoderLayer(embed_dim, num_heads, hidden_dim,
                                        diffusion_steps, alpha, warmup_steps)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, enc_out, self_attention_mask=None, cross_attention_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, enc_out,
                           self_attention_mask=self_attention_mask,
                           cross_attention_mask=cross_attention_mask)
        return output


class QGFDTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers,
                 embed_dim, num_heads, hidden_dim,
                 diffusion_steps=2, alpha=0.3, warmup_steps=0):
        super().__init__()
        self.encoder = QGFDTransformerEncoder(num_encoder_layers, embed_dim, num_heads,
                                              hidden_dim, diffusion_steps, alpha, warmup_steps)
        self.decoder = QGFDTransformerDecoder(num_decoder_layers, embed_dim, num_heads,
                                              hidden_dim, diffusion_steps, alpha, warmup_steps)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        enc_out = self.encoder(src, attention_mask=src_mask)
        dec_out = self.decoder(tgt, enc_out,
                               self_attention_mask=tgt_mask,
                               cross_attention_mask=cross_mask)
        return dec_out
