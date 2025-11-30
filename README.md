# QGFD: Query Graph Flow Diffusion Transformers

**QGFD (Query Graph Flow Diffusion)** is a research-oriented library for turning **pretrained Transformers** (T5, ViT, Wav2Vec2, etc.) into **diffusion denoisers** across multiple modalities, while injecting a custom **QGFD attention kernel** into their self-attention.

The core idea:

> Use a Transformer (with QGFD attention) as the diffusion model  
> **εθ(xₜ, t, cond)** in latent spaces for **text, images, audio, video, tabular data**.

---

## ✨ Features

- **QGFD Attention** (`QGFDMultiheadAttention`)
  - Drop-in replacement for multi-head attention.
  - Applies a Markov-style diffusion over **self-attention** distributions.
  - Leaves **cross-attention** untouched (no rectangular matmul issues).

- **Transformer-as-Diffusion-Backbone**
  - `QGFDDiffusionBackbone`: a stack of QGFD-augmented Transformer blocks that predicts noise in latent space.
  - Compatible with DDPM-style training (`DDPMBetaSchedule`, `q_sample`).

- **Modality-aware wrappers**
  - **Text**: T5-based (`TextQGFDDiffusionModel`), diffusion over T5 embeddings (Diffusion-LM style).
  - **Images**: ViT-based (`ImageQGFDDiffusionModel`), diffusion over ViT patch embeddings.
  - **Audio**: Wav2Vec2-based (`AudioQGFDDiffusionModel`), diffusion over frame embeddings.
  - **Video**: generic frame-latent wrapper (`VideoQGFDDiffusionModel`).
  - **Tabular**: simple MLP → token sequence (`TabularQGFDDiffusionModel`).

- **HuggingFace integration**
  - Uses `t5-small`, `google/vit-base-patch16-224`, `facebook/wav2vec2-base`, etc.
  - You can swap any compatible HF backbone.

- **One-line factory**
  - `create_qgfd_diffusion_model(modality, backbone_name, diffusion_cfg, ...)`
    returns a model, HF processor/tokenizer, and a DDPM schedule.

---

## 📦 Installation

This is intended as a Python library you host on GitHub.

Basic local dev install:

```bash
git clone https://github.com/rajboopathiking/TorchDire.git
cd TorchDire

```
Required packages (minimal):

```bash
pip install torch torchvision
pip install transformers datasets

```
📁 Library Structure
Key file:

QGFD_Diffusion_Transformer.py

Main components inside:

Core :

    QGFDMultiheadAttention
    
    PositionwiseFFN
    
    TimestepEmbedding
    
    EncoderBlock, DecoderBlock
    
    QGFDDiffusionBackbone
    
    DiffusionConfig
    
    DDPMBetaSchedule, q_sample
    
Modalities :
    
    TextQGFDDiffusionModel
    
    ImageQGFDDiffusionModel
    
    AudioQGFDDiffusionModel
    
    TabularQGFDDiffusionModel
    
    VideoQGFDDiffusionModel
    
Factory
    
    create_qgfd_diffusion_model(...)

🚀 Quickstart
1. Import and create a model
```python
import torch
from QGFD_Diffusion_Transformer import (
    create_qgfd_diffusion_model,
    DiffusionConfig,
    DDPMBetaSchedule,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

diff_cfg = DiffusionConfig(
    d_model=512,
    num_heads=8,
    num_layers=4,
    d_ff=2048,
    latent_dim=512,
    num_timesteps=1000,
    qgfd_steps=1,
)
# TEXT diffusion model (T5-based)

text_model, tokenizer, schedule = create_qgfd_diffusion_model(
    modality="text",
    backbone_name="t5-small",
    diffusion_cfg=diff_cfg,
)
text_model.to(device)
schedule = schedule.to(device)

```
# 2. Dummy text diffusion step
```python
texts = ["hello world", "query graph flow diffusion"]
batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)

# Encode clean embeddings
with torch.no_grad():
    x0_embeds = text_model.encode_text(
        batch["input_ids"], batch["attention_mask"]
    )  # [B,L,D]

B = x0_embeds.size(0)
t = torch.randint(0, diff_cfg.num_timesteps, (B,), device=device)

loss = text_model.diffusion_forward(
    x0_embeds=x0_embeds,
    t=t,
    cond_input_ids=batch["input_ids"],
    cond_attention_mask=batch["attention_mask"],
    schedule=schedule,
)

print("Text diffusion loss:", loss.item())
```
You should see a positive finite loss (≈ 1–2 for untrained denoiser).

# 🧠 Architecture Overview
# QGFD Attention
```python
class QGFDMultiheadAttention(nn.Module):
    # ...
    def forward(self, x_q, x_kv, attn_mask=None, qgfd=True):
        # compute standard attention
        # if qgfd=True and L_q == L_k:
        #   build Markov matrix P from attn_scores
        #   apply diffusion attn_diff = attn @ P^k
        #   mix: attn = α * attn_diff + (1 - α) * attn
        # return projected output
```
Self-attention (L_q == L_k):

QGFD applied.

Cross-attention (L_q != L_k):

qgfd=False, no diffusion (safe for ViT, text-image, etc.).

# Diffusion Backbone
```python
class QGFDDiffusionBackbone(nn.Module):
    def forward(self, x_t, t, cond_tokens=None, cond_mask=None):
        # project latents -> model dim
        # add time embedding
        # run stack of QGFD DecoderBlocks with cond_tokens as "encoder"
        # project back to latent_dim → epsilon prediction
        return eps_pred
```
This is your εθ(xₜ, t, cond) used by all modality wrappers.

## 🧩 Shared Backbone: One QGFD Diffusion Transformer for Text + Image

By default, the library exposes *per-modality* diffusion models:

- `TextQGFDDiffusionModel` (T5-based)
- `ImageQGFDDiffusionModel` (ViT-based)
- ...

Each one owns its own `QGFDDiffusionBackbone`.

For research, we often want a **single shared Transformer** that acts as εθ across modalities — e.g.,

- first train on **text** (MRPC),
- then continue training on **images** (CIFAR-10),
- and see whether QGFD improves cross-modal generalization.

### Design: shared QGFD backbone + modality adapters

We introduce a simple pattern:

- **One shared denoiser**: `QGFDDiffusionBackbone`
- **Per-modality adapters**:
  - `TextAdapter` – T5 encoder → latent tokens & conditioning
  - `ImageAdapter` – ViT encoder → latent tokens & (optional) conditioning
  - (Audio/Tabular/Video can be added the same way)

Each adapter does:

1. Encode raw data → **x₀_latent** (shape `[B, L_m, D_m]`).
2. Project latents to the **shared latent_dim** via a linear layer.
3. Provide **conditioning tokens** in the same `d_model` as the backbone (or `None`).

The shared backbone sees everything as:

```python
eps_pred = backbone(
    x_t,           # [B, L_shared, latent_dim]
    t,             # [B]
    cond_tokens,   # [B, L_cond, d_model] or None
    cond_mask,     # [B, L_cond] or None
)
```
```python 
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoImageProcessor,
    ViTModel,
)
from TorchDire.QGFD_Diffusion_Transformer import (
    QGFDDiffusionBackbone,
    DiffusionConfig,
    DDPMBetaSchedule,
    q_sample
)


class TextAdapter(nn.Module):
    """
    Text modality adapter:
    - Uses pretrained T5 encoder to get text embeddings.
    - Projects embeddings into the shared latent space (latent_dim).
    - Also returns encoder outputs as conditioning tokens.
    """

    def __init__(self, hf_name: str, shared_cfg: DiffusionConfig):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(hf_name)
        self.tokenizer = T5Tokenizer.from_pretrained(hf_name)
        self.d_model = self.t5.config.d_model

        # project text embeddings -> shared latent_dim
        self.proj_in = nn.Linear(self.d_model, shared_cfg.latent_dim)

    def encode_to_latents(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        returns:
            x0_latent: [B, L, latent_dim]
            cond_tokens: [B, L, d_model]
            cond_mask: [B, L]
        """
        enc_out = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # [B,L,D_text]

        x0_latent = self.proj_in(enc_out)  # [B,L,latent_dim]
        cond_tokens = enc_out              # [B,L,D_text] (backbone expects d_model == shared_cfg.d_model)
        cond_mask = attention_mask         # [B,L]

        return x0_latent, cond_tokens, cond_mask


class ImageAdapter(nn.Module):
    """
    Image modality adapter:
    - Uses pretrained ViT to get patch embeddings.
    - Projects patch embeddings into the shared latent space (latent_dim).
    - For now, no extra conditioning tokens are used (unconditional diffusion).
    """

    def __init__(self, hf_name: str, shared_cfg: DiffusionConfig):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(hf_name)
        self.vit = ViTModel.from_pretrained(hf_name)
        self.d_model = self.vit.config.hidden_size

        # project ViT patch embeddings -> shared latent_dim
        self.proj_in = nn.Linear(self.d_model, shared_cfg.latent_dim)

    def encode_to_latents(self, pixel_values: torch.Tensor):
        """
        pixel_values: [B,3,H,W]
        returns:
            x0_latent: [B, L_patches, latent_dim]
            cond_tokens: None
            cond_mask: None
        """
        out = self.vit(pixel_values=pixel_values)
        patches = out.last_hidden_state[:, 1:, :]  # drop CLS
        x0_latent = self.proj_in(patches)          # [B,L_patches,latent_dim]

        cond_tokens = None
        cond_mask = None
        return x0_latent, cond_tokens, cond_mask


class SharedQGFDDiffusionModel(nn.Module):
    """
    Shared QGFD diffusion backbone across multiple modalities.

    - backbone: QGFDDiffusionBackbone (Transformer εθ with QGFD attention).
    - text_adapter: T5 encoder + projection.
    - image_adapter: ViT encoder + projection.
    """

    def __init__(
        self,
        shared_cfg: DiffusionConfig,
        text_hf_name: str = "t5-small",
        image_hf_name: str = "google/vit-base-patch16-224",
    ):
        super().__init__()
        self.cfg = shared_cfg
        self.backbone = QGFDDiffusionBackbone(shared_cfg)

        self.text_adapter = TextAdapter(text_hf_name, shared_cfg)
        self.image_adapter = ImageAdapter(image_hf_name, shared_cfg)

    # --------------------------------------------------
    # TEXT MODE: MRPC diffusion loss
    # --------------------------------------------------
    def text_diffusion_loss(
        self,
        input_ids: torch.Tensor,          # [B,L]
        attention_mask: torch.Tensor,     # [B,L]
        schedule: DDPMBetaSchedule,
    ) -> torch.Tensor:
        x0_latent, cond_tokens, cond_mask = self.text_adapter.encode_to_latents(
            input_ids, attention_mask
        )  # [B,L,latent_dim], [B,L,d_model], [B,L]

        B = x0_latent.size(0)
        device = x0_latent.device
        t = torch.randint(0, self.cfg.num_timesteps, (B,), device=device)

        x_t, eps = q_sample(x0_latent, t, schedule)
        eps_pred = self.backbone(
            x_t,
            t,
            cond_tokens=cond_tokens,
            cond_mask=cond_mask,
        )

        loss = F.mse_loss(eps_pred, eps)
        return loss

    # --------------------------------------------------
    # IMAGE MODE: CIFAR-10 diffusion loss
    # --------------------------------------------------
    def image_diffusion_loss(
        self,
        pixel_values: torch.Tensor,   # [B,3,H,W]
        schedule: DDPMBetaSchedule,
    ) -> torch.Tensor:
        x0_latent, cond_tokens, cond_mask = self.image_adapter.encode_to_latents(
            pixel_values
        )  # [B,L_patches,latent_dim], cond=None

        B = x0_latent.size(0)
        device = x0_latent.device
        t = torch.randint(0, self.cfg.num_timesteps, (B,), device=device)

        x_t, eps = q_sample(x0_latent, t, schedule)
        eps_pred = self.backbone(
            x_t,
            t,
            cond_tokens=cond_tokens,
            cond_mask=cond_mask,
        )

        loss = F.mse_loss(eps_pred, eps)
        return loss
```
