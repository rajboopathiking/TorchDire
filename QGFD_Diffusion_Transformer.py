import math
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hugging Face
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoImageProcessor,
    ViTModel,
    AutoFeatureExtractor,
    Wav2Vec2Model,
)


# ================================================================
# QGFD Attention
# ================================================================

class QGFDMultiheadAttention(nn.Module):
    """
    QGFD-augmented multi-head attention.

    This is a drop-in-ish replacement for standard MHA:
    - computes scaled dot-product attention
    - optionally diffuses attention distributions via a Markov matrix P
      (only when qgfd=True AND the attention matrix is square, i.e. self-attention).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
        qgfd_steps: int = 0,
        qgfd_alpha: float = 0.5,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.qgfd_steps = qgfd_steps
        self.qgfd_alpha = qgfd_alpha

    def build_markov_matrix(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        Construct a Markov matrix P from attention scores.
        You can replace this with your own QGFD scheme.
        """
        P = torch.softmax(attn_scores, dim=-1)
        return P

    def diffuse_attention(self, attn: torch.Tensor, P: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Simple diffusion on attention distributions:
        attn_{t+1} = attn_t @ P
        """
        if steps <= 0:
            return attn
        attn_diff = attn
        for _ in range(steps):
            attn_diff = torch.matmul(attn_diff, P)
        return attn_diff

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        qgfd: bool = True,
    ) -> torch.Tensor:
        """
        x_q: [B, L_q, D]
        x_kv: [B, L_k, D]
        attn_mask: [B, 1, L_q, L_k] or broadcastable (True means "mask this position")
        qgfd: whether to apply QGFD diffusion (only if L_q == L_k)
        """
        B, L_q, _ = x_q.size()
        _, L_k, _ = x_kv.size()

        q = self.q_proj(x_q)
        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q = q.view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,L_q,L_k]

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))

        attn = torch.softmax(attn_scores, dim=-1)

        # Only apply QGFD when:
        # - explicitly enabled (qgfd=True),
        # - attention matrix is square (self-attention: L_q == L_k),
        # - we have >0 diffusion steps.
        if qgfd and L_q == L_k and self.qgfd_steps > 0:
            P = self.build_markov_matrix(attn_scores)
            attn_diff = self.diffuse_attention(attn, P, self.qgfd_steps)
            attn = self.qgfd_alpha * attn_diff + (1.0 - self.qgfd_alpha) * attn

        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # [B,H,L_q,head_dim]
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
        out = self.out_proj(out)
        return out


# ================================================================
# Basic Transformer blocks + Time Embedding
# ================================================================

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=torch.float32)
            * -(math.log(10000.0) / (half_dim - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        sin = torch.sin(args)
        cos = torch.cos(args)
        emb = torch.cat([sin, cos], dim=-1)
        emb = self.mlp(emb)
        return emb


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        qgfd_steps: int = 0,
    ):
        super().__init__()
        self.self_attn = QGFDMultiheadAttention(
            d_model, num_heads, dropout=dropout, qgfd_steps=qgfd_steps
        )
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, self_mask: Optional[torch.Tensor] = None):
        # QGFD enabled for encoder self-attention
        h = self.self_attn(self.norm1(x), self.norm1(x), attn_mask=self_mask, qgfd=True)
        x = x + self.dropout(h)
        h = self.ffn(self.norm2(x))
        x = x + self.dropout(h)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        qgfd_steps: int = 0,
    ):
        super().__init__()
        self.self_attn = QGFDMultiheadAttention(
            d_model, num_heads, dropout=dropout, qgfd_steps=qgfd_steps
        )
        self.cross_attn = QGFDMultiheadAttention(
            d_model, num_heads, dropout=dropout, qgfd_steps=qgfd_steps
        )
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        cross_mask: Optional[torch.Tensor] = None,
    ):
        # Self-attention (L_q == L_k) → QGFD can be applied
        h = self.self_attn(self.norm1(x), self.norm1(x), attn_mask=self_mask, qgfd=True)
        x = x + self.dropout(h)

        # Cross-attention (L_q != L_k in general) → DO NOT apply QGFD
        h = self.cross_attn(self.norm2(x), enc_out, attn_mask=cross_mask, qgfd=False)
        x = x + self.dropout(h)

        # FFN
        h = self.ffn(self.norm3(x))
        x = x + self.dropout(h)
        return x


# ================================================================
# Base QGFD Diffusion Transformer (denoiser)
# ================================================================

@dataclass
class DiffusionConfig:
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    qgfd_steps: int = 1
    latent_dim: int = 256
    num_timesteps: int = 1000


class QGFDDiffusionBackbone(nn.Module):
    """
    Generic diffusion denoiser:
    - inputs: x_t (latent tokens), t (timesteps), optional cond tokens
    - predicts epsilon over latent space
    """

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.time_embed = TimestepEmbedding(cfg.d_model)
        self.time_to_tokens = nn.Linear(cfg.d_model, cfg.d_model)
        self.latent_proj_in = nn.Linear(cfg.latent_dim, cfg.d_model)
        self.latent_proj_out = nn.Linear(cfg.d_model, cfg.latent_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=cfg.d_model,
                    num_heads=cfg.num_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    qgfd_steps=cfg.qgfd_steps,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def _add_time(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        t_emb = self.time_to_tokens(t_emb).unsqueeze(1)
        return h + t_emb

    def forward(
        self,
        x_t: torch.Tensor,               # [B, L, latent_dim]
        t: torch.Tensor,                 # [B]
        cond_tokens: Optional[torch.Tensor] = None,  # [B, Lc, d_model]
        cond_mask: Optional[torch.Tensor] = None,    # [B, Lc]
    ) -> torch.Tensor:
        B, _, _ = x_t.size()
        h = self.latent_proj_in(x_t)
        h = self._add_time(h, t)

        if cond_tokens is None:
            cond_tokens = torch.zeros(B, 1, self.cfg.d_model, device=x_t.device)
            cond_mask = torch.ones(B, 1, device=x_t.device)

        cross_mask = (cond_mask[:, None, None, :] == 0)

        for layer in self.layers:
            h = layer(
                h,
                cond_tokens,
                self_mask=None,
                cross_mask=cross_mask,
            )

        h = self.norm(h)
        eps_pred = self.latent_proj_out(h)
        return eps_pred


# ================================================================
# DDPM utilities
# ================================================================

class DDPMBetaSchedule:
    def __init__(self, num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self


def q_sample(x0: torch.Tensor, t: torch.Tensor, schedule: DDPMBetaSchedule, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    alpha_bar_t = schedule.alphas_cumprod[t].view(-1, 1, 1)
    return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise, noise


# ================================================================
# TEXT MODALITY: T5 seq2seq + QGFD diffusion over text embeddings
# ================================================================

class TextQGFDDiffusionModel(nn.Module):
    """
    - Uses HF T5 as text encoder/decoder (pretrained weights)
    - Uses QGFDDiffusionBackbone as denoiser over embedding space
    """

    def __init__(self, hf_name: str, diffusion_cfg: DiffusionConfig):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(hf_name)
        d_model = self.t5.config.d_model
        diffusion_cfg.d_model = d_model
        diffusion_cfg.latent_dim = d_model  # text embeddings
        self.diffusion_cfg = diffusion_cfg
        self.denoiser = QGFDDiffusionBackbone(diffusion_cfg)

    @property
    def tokenizer(self):
        return T5Tokenizer.from_pretrained(self.t5.name_or_path)

    def encode_text(self, input_ids, attention_mask):
        enc_out = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return enc_out  # [B,L,D]

    def seq2seq(self, **kwargs):
        # pass-through to T5
        return self.t5(**kwargs)

    def diffusion_forward(
        self,
        x0_embeds: torch.Tensor,    # [B,L,D] clean text embeddings
        t: torch.Tensor,            # [B]
        cond_input_ids: torch.Tensor,
        cond_attention_mask: torch.Tensor,
        schedule: DDPMBetaSchedule,
    ):
        # forward diffusion
        x_t, eps = q_sample(x0_embeds, t, schedule)
        # conditioning tokens from encoder
        cond_tokens = self.encode_text(cond_input_ids, cond_attention_mask)  # [B,Lc,D]
        eps_pred = self.denoiser(
            x_t,
            t,
            cond_tokens=cond_tokens,
            cond_mask=cond_attention_mask,
        )
        loss = F.mse_loss(eps_pred, eps)
        return loss


# ================================================================
# IMAGE MODALITY: ViT encoder + QGFD denoiser on patch tokens
# ================================================================

class ImageQGFDDiffusionModel(nn.Module):
    """
    - HF ViTModel as latent encoder of images (patch embeddings).
    - QGFDDiffusionBackbone as denoiser in patch-embedding space.
    """

    def __init__(self, hf_name: str, diffusion_cfg: DiffusionConfig):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(hf_name)
        self.vit = ViTModel.from_pretrained(hf_name)
        d_model = self.vit.config.hidden_size
        diffusion_cfg.d_model = d_model
        diffusion_cfg.latent_dim = d_model
        self.diffusion_cfg = diffusion_cfg
        self.denoiser = QGFDDiffusionBackbone(diffusion_cfg)

    def image_to_latents(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Use ViT patch embeddings as x0 latents (exclude CLS).
        pixel_values: [B,3,H,W]
        """
        outputs = self.vit(pixel_values=pixel_values)
        # outputs.last_hidden_state: [B, 1+num_patches, D]
        latents = outputs.last_hidden_state[:, 1:, :]  # drop CLS
        return latents  # [B, L, D]

    def diffusion_forward(
        self,
        x0_latent: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: Optional[torch.Tensor],
        cond_mask: Optional[torch.Tensor],
        schedule: DDPMBetaSchedule,
    ):
        x_t, eps = q_sample(x0_latent, t, schedule)
        eps_pred = self.denoiser(
            x_t,
            t,
            cond_tokens=cond_tokens,
            cond_mask=cond_mask,
        )
        loss = F.mse_loss(eps_pred, eps)
        return loss


# ================================================================
# AUDIO MODALITY: Wav2Vec2 encoder + QGFD denoiser on frame tokens
# ================================================================

class AudioQGFDDiffusionModel(nn.Module):
    """
    - HF Wav2Vec2Model as latent encoder of audio.
    - QGFDDiffusionBackbone as denoiser in frame embedding space.
    """

    def __init__(self, hf_name: str, diffusion_cfg: DiffusionConfig):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(hf_name)
        self.wav2vec = Wav2Vec2Model.from_pretrained(hf_name)
        d_model = self.wav2vec.config.hidden_size
        diffusion_cfg.d_model = d_model
        diffusion_cfg.latent_dim = d_model
        self.diffusion_cfg = diffusion_cfg
        self.denoiser = QGFDDiffusionBackbone(diffusion_cfg)

    def audio_to_latents(self, input_values: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        outputs = self.wav2vec(input_values=input_values, attention_mask=attention_mask)
        latents = outputs.last_hidden_state  # [B, L, D]
        return latents

    def diffusion_forward(
        self,
        x0_latent: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: Optional[torch.Tensor],
        cond_mask: Optional[torch.Tensor],
        schedule: DDPMBetaSchedule,
    ):
        x_t, eps = q_sample(x0_latent, t, schedule)
        eps_pred = self.denoiser(
            x_t,
            t,
            cond_tokens=cond_tokens,
            cond_mask=cond_mask,
        )
        loss = F.mse_loss(eps_pred, eps)
        return loss


# ================================================================
# TABULAR MODALITY: simple MLP to tokens + QGFD denoiser
# ================================================================

class TabularQGFDDiffusionModel(nn.Module):
    """
    - Simple linear projection from tabular features to token sequence.
    - QGFDDiffusionBackbone as denoiser in token space.
    """

    def __init__(self, num_features: int, seq_len: int, diffusion_cfg: DiffusionConfig):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.token_dim = diffusion_cfg.latent_dim

        self.feat_to_tokens = nn.Linear(num_features, seq_len * self.token_dim)
        self.denoiser = QGFDDiffusionBackbone(diffusion_cfg)

    def tabular_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_features]
        returns [B, seq_len, token_dim]
        """
        B = x.size(0)
        tokens = self.feat_to_tokens(x)  # [B, seq_len * token_dim]
        tokens = tokens.view(B, self.seq_len, self.token_dim)
        return tokens

    def diffusion_forward(
        self,
        x0_latent: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: Optional[torch.Tensor],
        cond_mask: Optional[torch.Tensor],
        schedule: DDPMBetaSchedule,
    ):
        x_t, eps = q_sample(x0_latent, t, schedule)
        eps_pred = self.denoiser(
            x_t,
            t,
            cond_tokens=cond_tokens,
            cond_mask=cond_mask,
        )
        loss = F.mse_loss(eps_pred, eps)
        return loss


# ================================================================
# VIDEO MODALITY: simple frame-patch embedding + QGFD denoiser
# (skeleton – you’d plug in VideoMAE/TimeSformer here)
# ================================================================

class VideoQGFDDiffusionModel(nn.Module):
    """
    Skeleton: treat video as sequence of frame latents (e.g., via ViT per frame)
    and run QGFDDiffusionBackbone on that sequence.
    """

    def __init__(self, frame_latent_dim: int, diffusion_cfg: DiffusionConfig):
        super().__init__()
        diffusion_cfg.latent_dim = frame_latent_dim
        self.denoiser = QGFDDiffusionBackbone(diffusion_cfg)

    def video_to_latents(self, video_latents: torch.Tensor) -> torch.Tensor:
        """
        video_latents: [B, T, L_frame, D] -> flatten to [B, T*L_frame, D]
        """
        B, T, Lf, D = video_latents.shape
        return video_latents.view(B, T * Lf, D)

    def diffusion_forward(
        self,
        x0_latent: torch.Tensor,
        t: torch.Tensor,
        cond_tokens: Optional[torch.Tensor],
        cond_mask: Optional[torch.Tensor],
        schedule: DDPMBetaSchedule,
    ):
        x_t, eps = q_sample(x0_latent, t, schedule)
        eps_pred = self.denoiser(
            x_t,
            t,
            cond_tokens=cond_tokens,
            cond_mask=cond_mask,
        )
        loss = F.mse_loss(eps_pred, eps)
        return loss


# ================================================================
# Factory: create_qgfd_diffusion_model(...)
# ================================================================

def create_qgfd_diffusion_model(
    modality: Literal["text", "image", "audio", "video", "tabular"],
    backbone_name: str,
    diffusion_cfg: Optional[DiffusionConfig] = None,
    **kwargs,
):
    """
    Returns:
        model  - modality-specific QGFD diffusion model
        hf_obj - tokenizer/processor or None
        schedule - DDPMBetaSchedule
    """
    if diffusion_cfg is None:
        diffusion_cfg = DiffusionConfig()

    if modality == "text":
        model = TextQGFDDiffusionModel(backbone_name, diffusion_cfg)
        tokenizer = model.tokenizer
        schedule = DDPMBetaSchedule(diffusion_cfg.num_timesteps)
        return model, tokenizer, schedule

    elif modality == "image":
        model = ImageQGFDDiffusionModel(backbone_name, diffusion_cfg)
        processor = model.processor
        schedule = DDPMBetaSchedule(diffusion_cfg.num_timesteps)
        return model, processor, schedule

    elif modality == "audio":
        model = AudioQGFDDiffusionModel(backbone_name, diffusion_cfg)
        feature_extractor = model.feature_extractor
        schedule = DDPMBetaSchedule(diffusion_cfg.num_timesteps)
        return model, feature_extractor, schedule

    elif modality == "tabular":
        num_features = kwargs.get("num_features", 32)
        seq_len = kwargs.get("seq_len", 8)
        # latent_dim already on diffusion_cfg; keep or adjust here if needed
        model = TabularQGFDDiffusionModel(num_features, seq_len, diffusion_cfg)
        schedule = DDPMBetaSchedule(diffusion_cfg.num_timesteps)
        return model, None, schedule

    elif modality == "video":
        frame_latent_dim = kwargs.get("frame_latent_dim", diffusion_cfg.latent_dim)
        model = VideoQGFDDiffusionModel(frame_latent_dim, diffusion_cfg)
        schedule = DDPMBetaSchedule(diffusion_cfg.num_timesteps)
        return model, None, schedule

    else:
        raise ValueError(f"Unsupported modality: {modality}")
