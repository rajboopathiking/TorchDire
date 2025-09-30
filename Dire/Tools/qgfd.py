import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


# QGFDLayer from qgfd.py (included for completeness)
class QGFDLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, diffusion_steps=4, alpha=0.5, use_bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.proj_dim = self.head_dim * num_heads
        self.diffusion_steps = diffusion_steps
        self.alpha = alpha
        self.use_bias = use_bias

        self.q_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.proj_dim, embed_dim, bias=use_bias)

    def build_transition_from_keys(self, K):
        B, H, N, Dk = K.shape
        sim = torch.einsum("bhid,bhjd->bhij", K, K) / math.sqrt(max(1e-8, Dk))
        if torch.isnan(sim).any() or torch.isinf(sim).any():
            raise ValueError("Invalid values in similarity matrix")
        P = F.softmax(sim, dim=-1)
        if torch.isnan(P).any() or torch.isinf(P).any():
            raise ValueError("Invalid values in transition matrix")
        return P

    def forward(self, hidden_states, kv=None, attention_mask=None, past_key_values=None, use_cache=False):
        B, N, D = hidden_states.shape
        if kv is None:
            kv = hidden_states
        K_len = kv.shape[1]

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.k_proj(kv).view(B, K_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.v_proj(kv).view(B, K_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (math.sqrt(self.head_dim) + 1e-8)
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            raise ValueError("Invalid values in attention scores")

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() != 4:
                raise ValueError(f"Expected attention_mask dims 2/3/4, got {attention_mask.dim()}")
            if attention_mask.shape[-1] != K_len:
                raise ValueError(f"Mask last dim {attention_mask.shape[-1]} != key length {K_len}")
            if attention_mask.shape[-2] != N and attention_mask.shape[-2] != 1:
                raise ValueError(f"Mask query dim {attention_mask.shape[-2]} != query length {N} or 1")
            scores = scores + attention_mask
            # Use a smaller negative value for masking when using FP16
            if scores.dtype == torch.float16:
                scores = torch.where(torch.isinf(scores), torch.full_like(scores, -1e4), scores)
            else:
                scores = torch.where(torch.isinf(scores), torch.full_like(scores, -1e9), scores)
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                raise ValueError("Invalid values in masked scores")

        p = F.softmax(scores, dim=-1)
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise ValueError("Invalid values in attention probabilities")
        p0 = p.clone()

        P = self.build_transition_from_keys(K)

        prev_p = None
        for step in range(self.diffusion_steps):
            p = (1 - self.alpha) * p0 + self.alpha * torch.einsum("bhqn,bhnm->bhqm", p, P)
            if torch.isnan(p).any() or torch.isinf(p).any():
                raise ValueError(f"Invalid values in diffusion step {step + 1}")
            if prev_p is not None and torch.max(torch.abs(p - prev_p)) < 1e-4:
                break
            prev_p = p.clone()

        attn_output = torch.einsum("bhqk,bhkd->bhqd", p, V)
        if torch.isnan(attn_output).any() or torch.isinf(attn_output).any():
            raise ValueError("Invalid values in attention output")

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, self.proj_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, p


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


class StableQGFDLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8,
                 diffusion_steps=1, alpha=0.05,
                 use_bias=True, detach_P=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.proj_dim = self.head_dim * num_heads

        self.diffusion_steps = diffusion_steps
        self.alpha = alpha
        self.detach_P = detach_P

        # projections
        self.q_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.k_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.v_proj = nn.Linear(embed_dim, self.proj_dim, bias=use_bias)
        self.out_proj = nn.Linear(self.proj_dim, embed_dim, bias=use_bias)

    def build_transition_from_keys(self, K):
        """
        Transition matrix P based on key similarity
        Scale by sqrt(embed_dim) (not head_dim) for stability.
        """
        sim = torch.einsum("bhid,bhjd->bhij", K, K) / math.sqrt(self.embed_dim)
        P = F.softmax(sim, dim=-1)
        if self.detach_P:
            P = P.detach()  # prevent gradients exploding through diffusion
        return P

    def forward(
    self,
    hidden_states,
    kv=None,
    attention_mask=None,
    past_key_values=None,   # explicit argument
    step=None,
    warmup_steps=0,
    use_cache=False, # Explicitly accept use_cache
    **kwargs
):
      B, N, D = hidden_states.shape
      if kv is None:
          kv = hidden_states
      K_len = kv.shape[1]

      # ignore past_key_values for now (will implement caching later)
      # past_key_values = past_key_values

      # project Q, K, V
      Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
      K = self.k_proj(kv).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)
      V = self.v_proj(kv).view(B, K_len, self.num_heads, self.head_dim).transpose(1, 2)

      # standard attention
      scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / math.sqrt(self.head_dim)
      if attention_mask is not None:
          scores = scores + attention_mask
      p = F.softmax(scores, dim=-1)
      p0 = p.clone()

      # diffusion matrix from keys
      P = self.build_transition_from_keys(K)

      # effective alpha (with warmup)
      if step is not None and warmup_steps > 0:
          factor = min(1.0, step / warmup_steps)
      else:
          factor = 1.0
      alpha_eff = self.alpha * factor

      # diffusion iterations
      prev_p = None
      for _ in range(self.diffusion_steps):
          p = (1 - alpha_eff) * p0 + alpha_eff * torch.einsum("bhqn,bhnm->bhqm", p, P)
          if prev_p is not None and torch.max(torch.abs(p - prev_p)) < 1e-4:
              break
          prev_p = p.clone()

      # apply to values
      attn_output = torch.einsum("bhqk,bhkd->bhqd", p, V)
      attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.proj_dim)
      attn_output = self.out_proj(attn_output)

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
