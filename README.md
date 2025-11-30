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
git clone https://github.com/yourname/TorchDire.git](https://github.com/rajboopathiking/TorchDire.git
cd TorchDire
pip install -e

```
Required packages (minimal):

```bash
Copy code
pip install torch torchvision
pip install transformers datasets
You can also add these to your pyproject.toml / setup.cfg later.
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
