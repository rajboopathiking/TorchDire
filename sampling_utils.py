import torch
from .QGFD_Diffusion_Transformer import DDPMBetaSchedule

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    AutoImageProcessor,
    ViTModel,
)
from .QGFD_Diffusion_Transformer import (
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


def p_sample_backbone(
    backbone,
    x_t: torch.Tensor,
    t_scalar: int,
    schedule: DDPMBetaSchedule,
    cond_tokens: torch.Tensor,
    cond_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Single DDPM reverse step x_t -> x_{t-1} using epsilon prediction.
    backbone: QGFDDiffusionBackbone
    """
    device = x_t.device
    B = x_t.size(0)
    t = torch.full((B,), t_scalar, device=device, dtype=torch.long)

    betas = schedule.betas
    alphas = schedule.alphas
    alphas_cumprod = schedule.alphas_cumprod

    beta_t = betas[t_scalar]
    alpha_t = alphas[t_scalar]
    alpha_bar_t = alphas_cumprod[t_scalar]
    alpha_bar_prev = alphas_cumprod[t_scalar - 1] if t_scalar > 0 else torch.tensor(1.0, device=device)

    beta_t = beta_t.view(1, 1, 1)
    alpha_t = alpha_t.view(1, 1, 1)
    alpha_bar_t = alpha_bar_t.view(1, 1, 1)
    alpha_bar_prev = alpha_bar_prev.view(1, 1, 1)

    # predict epsilon
    eps_pred = backbone(
        x_t,
        t,
        cond_tokens=cond_tokens,
        cond_mask=cond_mask,
    )

    # predict x0
    x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

    # posterior variance (simple version)
    posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)

    if t_scalar > 0:
        noise = torch.randn_like(x_t)
    else:
        noise = torch.zeros_like(x_t)

    mean = (
        torch.sqrt(alpha_bar_prev) * x0_pred
        + torch.sqrt(1.0 - alpha_bar_prev) * eps_pred
    )
    x_prev = mean + torch.sqrt(posterior_var) * noise
    return x_prev


def ddpm_sample(
    backbone,
    schedule: DDPMBetaSchedule,
    shape,                      # (B, L, D)
    cond_tokens: torch.Tensor,  # [B,Lc,D_model] or None
    cond_mask: torch.Tensor,    # [B,Lc] or None
    device=None,
):
    """
    Full DDPM sampling loop: x_T ~ N(0,I) -> x_0.
    """
    if device is None:
        device = next(backbone.parameters()).device

    B, L, D = shape
    x = torch.randn(B, L, D, device=device)
    T = schedule.num_timesteps

    for t_scalar in reversed(range(T)):
        x = p_sample_backbone(
            backbone,
            x,
            t_scalar,
            schedule,
            cond_tokens,
            cond_mask,
        )

    return x  # [B,L,D], approx x_0 sample


def ddim_sample(
    backbone,
    schedule: DDPMBetaSchedule,
    shape,                      # (B,L,D)
    cond_tokens: torch.Tensor,
    cond_mask: torch.Tensor,
    steps: int = 50,
    eta: float = 0.0,
    device=None,
):
    """
    DDIM sampler: uses subset of timesteps for faster sampling.
    For eta=0, sampling is deterministic given the cond + random seed.
    """
    if device is None:
        device = next(backbone.parameters()).device

    B, L, D = shape
    x = torch.randn(B, L, D, device=device)

    T = schedule.num_timesteps
    alphas_cumprod = schedule.alphas_cumprod.to(device)

    # choose 'steps' timesteps from [0..T-1]
    ts = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)

    for i in range(steps):
        t_scalar = int(ts[i].item())
        t = torch.full((B,), t_scalar, device=device, dtype=torch.long)

        alpha_bar_t = alphas_cumprod[t_scalar].view(1, 1, 1)
        eps_pred = backbone(
            x,
            t,
            cond_tokens=cond_tokens,
            cond_mask=cond_mask,
        )

        x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        if i == steps - 1:
            x = x0
        else:
            t_next = int(ts[i + 1].item())
            alpha_bar_next = alphas_cumprod[t_next].view(1, 1, 1)

            sigma = eta * torch.sqrt(
                (1.0 - alpha_bar_t / alpha_bar_next)
                * (1.0 - alpha_bar_next)
                / (1.0 - alpha_bar_t)
            )
            noise = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)

            x = (
                torch.sqrt(alpha_bar_next) * x0
                + torch.sqrt(1.0 - alpha_bar_next - sigma**2) * eps_pred
                + sigma * noise
            )

    return x  # [B,L,D], approx x_0 sample
