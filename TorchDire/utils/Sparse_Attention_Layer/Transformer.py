import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ----------- Sparse Scaled Dot-Product Attention -----------

class SparseScaledDotProductAttention(nn.Module):
    def __init__(self, window_size=3, dropout=0.1):
        super().__init__()
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
      B, H, N_q, D = Q.size()
      _, _, N_k, _ = K.size()

      scores = torch.full((B, H, N_q, N_k), float('-inf'), device=Q.device)

      half_win = self.window_size // 2

      for i in range(N_q):
          start = max(0, i - half_win)
          end = min(N_k, i + half_win + 1)
          scores[:, :, i, start:end] = torch.einsum("bhd,bhzd->bhz", Q[:, :, i], K[:, :, start:end])

      if mask is not None:
          scores = scores.masked_fill(mask == 0, float('-inf'))

      attn = F.softmax(scores / math.sqrt(D), dim=-1)
      attn = self.dropout(attn)
      out = torch.matmul(attn, V)  # (B, H, N_q, D)
      return out, attn



# ----------- Sparse Multi-Head Attention -----------

class SparseMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=3, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        self.attention = SparseScaledDotProductAttention(window_size, dropout)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    # Modified forward to accept Q, K, V
    def forward(self, Q_in, K_in, V_in, mask=None):
        B, N_q, _ = Q_in.size()
        _, N_k, _ = K_in.size()

        Q = self.W_q(Q_in).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_q, D)
        K = self.W_k(K_in).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_k, D)
        V = self.W_v(V_in).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N_k, D)


        out, attn = self.attention(Q, K, V, mask)  # Sparse attention

        out = out.transpose(1, 2).contiguous().view(B, N_q, self.embed_dim)  # concat heads
        out = self.fc(out)
        out = self.dropout(out)
        # Apply residual connection and layer norm to the query input
        out = self.layer_norm(Q_in + out)
        return out, attn


# ----------- Masked Sparse Multi-Head Attention (for decoder self-attention) -----------

class SparseMaskedMultiHeadAttention(SparseMultiHeadAttention):
    def forward(self, x, mask=None):
        B, N, _ = x.size()
        device = x.device

        if mask is None:
            causal_mask = torch.tril(torch.ones(N, N, device=device)).unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
        else:
            causal_mask = mask

        return super().forward(x, x, x, mask=causal_mask) # Self-attention uses x for Q, K, V


# ----------- Feed Forward Network -----------

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        return out


# ----------- Encoder Layer -----------

class SparseTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, window_size=3, dropout=0.1):
        super().__init__()
        # Encoder self-attention uses the same input for Q, K, V
        self.self_attn = SparseMultiHeadAttention(embed_dim, num_heads, window_size, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_hidden_dim, dropout)

    def forward(self, src, src_mask=None):
        src, attn = self.self_attn(src, src, src, src_mask) # Use src for Q, K, V
        src = self.feed_forward(src)
        return src, attn


# ----------- Decoder Layer -----------

class SparseTransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, window_size=3, dropout=0.1):
        super().__init__()
        self.self_attn = SparseMaskedMultiHeadAttention(embed_dim, num_heads, window_size, dropout)
        # Cross-attention takes query from decoder, key and value from encoder
        self.cross_attn = SparseMultiHeadAttention(embed_dim, num_heads, window_size, dropout)
        self.feed_forward = FeedForward(embed_dim, ff_hidden_dim, dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Decoder self-attention
        tgt, self_attn = self.self_attn(tgt, tgt_mask) # Masked self-attention uses tgt for Q, K, V

        # Decoder cross-attention
        # Query from decoder (tgt), Key and Value from encoder output (memory)
        tgt2, cross_attn = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + tgt2  # Residual connection before feed forward
        tgt = self.feed_forward(tgt)
        return tgt, self_attn, cross_attn


# ----------- Encoder Stack -----------

class SparseTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None):
        attn_weights = []
        for layer in self.layers:
            src, attn = layer(src, mask)
            attn_weights.append(attn)
        return src, attn_weights


# ----------- Decoder Stack -----------

class SparseTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        self_attn_weights = []
        cross_attn_weights = []
        for layer in self.layers:
            tgt, self_attn, cross_attn = layer(tgt, memory, tgt_mask, memory_mask)
            self_attn_weights.append(self_attn)
            cross_attn_weights.append(cross_attn)
        return tgt, self_attn_weights, cross_attn_weights


# ----------- Full Sparse Transformer -----------

class SparseTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, num_encoder_layers, num_decoder_layers,
                 window_size=3, dropout=0.1):
        super().__init__()
        encoder_layer = SparseTransformerEncoderLayer(embed_dim, num_heads, ff_hidden_dim, window_size, dropout)
        decoder_layer = SparseTransformerDecoderLayer(embed_dim, num_heads, ff_hidden_dim, window_size, dropout)

        self.encoder = SparseTransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = SparseTransformerDecoder(decoder_layer, num_decoder_layers)
        self.embed_dim = embed_dim

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory, enc_attn = self.encoder(src, src_mask)
        out, dec_self_attn, dec_cross_attn = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return out, enc_attn, dec_self_attn, dec_cross_attn
