import torch
from QGFD_Diffusion_Transformer import create_qgfd_diffusion_model, DiffusionConfig, DDPMBetaSchedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use a smaller config for fast tests
diff_cfg = DiffusionConfig(
    d_model=256,
    num_heads=4,
    num_layers=2,
    d_ff=1024,
    latent_dim=128,
    num_timesteps=100,
    qgfd_steps=1,
)

# ------------------------------------------------------------
# Helper for generating random t
# ------------------------------------------------------------
def get_time_batch(B, schedule):
    return torch.randint(0, schedule.num_timesteps, (B,), device=device)


# ------------------------------------------------------------
# 1. TEXT SANITY CHECK
# ------------------------------------------------------------
def sanity_text():
    print("\n=== TEXT SANITY CHECK ===")

    model, tokenizer, schedule = create_qgfd_diffusion_model(
        modality="text",
        backbone_name="t5-small",
        diffusion_cfg=diff_cfg,
    )
    model.to(device)
    schedule = schedule.to(device)

    # Dummy text input
    text = ["hello world", "test diffusion"]
    encoded = tokenizer(text, return_tensors="pt", padding=True).to(device)

    # Encode text → latent embeddings (x0)
    with torch.no_grad():
        x0_embeds = model.encode_text(
            encoded["input_ids"], encoded["attention_mask"]
        )  # [B,L,D]

    t = get_time_batch(x0_embeds.size(0), schedule)

    loss = model.diffusion_forward(
        x0_embeds=x0_embeds,
        t=t,
        cond_input_ids=encoded["input_ids"],
        cond_attention_mask=encoded["attention_mask"],
        schedule=schedule,
    )

    print("TEXT loss:", loss.item())
    assert torch.isfinite(loss), "TEXT diffusion returned NaN/inf."


# ------------------------------------------------------------
# 2. IMAGE SANITY CHECK
# ------------------------------------------------------------
def sanity_image():
    print("\n=== IMAGE SANITY CHECK ===")

    model, processor, schedule = create_qgfd_diffusion_model(
        modality="image",
        backbone_name="google/vit-base-patch16-224",
        diffusion_cfg=diff_cfg,
    )
    model.to(device)
    schedule = schedule.to(device)

    # Dummy random image batch: [B,3,224,224]
    pixel_values = torch.randn(2, 3, 224, 224, device=device)

    # Encode → latents (patch embeddings)
    with torch.no_grad():
        latents = model.image_to_latents(pixel_values)  # [B, L, D]

    # Dummy conditioning tokens: zero text
    B = latents.size(0)
    cond_tokens = torch.zeros(B, 1, diff_cfg.d_model, device=device)
    cond_mask = torch.ones(B, 1, device=device)

    t = get_time_batch(B, schedule)

    loss = model.diffusion_forward(
        x0_latent=latents,
        t=t,
        cond_tokens=cond_tokens,
        cond_mask=cond_mask,
        schedule=schedule,
    )

    print("IMAGE loss:", loss.item())
    assert torch.isfinite(loss), "IMAGE diffusion returned NaN/inf."


# ------------------------------------------------------------
# 3. AUDIO SANITY CHECK
# ------------------------------------------------------------
def sanity_audio():
    print("\n=== AUDIO SANITY CHECK ===")

    model, feature_extractor, schedule = create_qgfd_diffusion_model(
        modality="audio",
        backbone_name="facebook/wav2vec2-base",
        diffusion_cfg=diff_cfg,
    )
    model.to(device)
    schedule = schedule.to(device)

    # Dummy 1-second audio input (~16k samples)
    input_values = torch.randn(2, 16000, device=device)

    with torch.no_grad():
        latents = model.audio_to_latents(input_values)  # [B,L,D]

    B = latents.size(0)
    cond_tokens = torch.zeros(B, 1, diff_cfg.d_model, device=device)
    cond_mask = torch.ones(B, 1, device=device)

    t = get_time_batch(B, schedule)

    loss = model.diffusion_forward(
        x0_latent=latents,
        t=t,
        cond_tokens=cond_tokens,
        cond_mask=cond_mask,
        schedule=schedule,
    )

    print("AUDIO loss:", loss.item())
    assert torch.isfinite(loss), "AUDIO diffusion returned NaN/inf."


# ------------------------------------------------------------
# 4. TABULAR SANITY CHECK
# ------------------------------------------------------------
def sanity_tabular():
    print("\n=== TABULAR SANITY CHECK ===")

    model, _, schedule = create_qgfd_diffusion_model(
        modality="tabular",
        backbone_name="N/A",
        diffusion_cfg=diff_cfg,
        num_features=32,
        seq_len=8,
    )
    model.to(device)
    schedule = schedule.to(device)

    # Dummy tabular (batch 4)
    x = torch.randn(4, 32, device=device)

    with torch.no_grad():
        latents = model.tabular_to_latents(x)  # [B,L,D]

    B = latents.size(0)
    cond_tokens = torch.zeros(B, 1, diff_cfg.d_model, device=device)
    cond_mask = torch.ones(B, 1, device=device)

    t = get_time_batch(B, schedule)

    loss = model.diffusion_forward(
        x0_latent=latents,
        t=t,
        cond_tokens=cond_tokens,
        cond_mask=cond_mask,
        schedule=schedule,
    )

    print("TABULAR loss:", loss.item())
    assert torch.isfinite(loss), "TABULAR diffusion returned NaN/inf."


# ------------------------------------------------------------
# 5. VIDEO SANITY CHECK
# ------------------------------------------------------------
def sanity_video():
    print("\n=== VIDEO SANITY CHECK ===")

    model, _, schedule = create_qgfd_diffusion_model(
        modality="video",
        backbone_name="N/A",
        diffusion_cfg=diff_cfg,
        frame_latent_dim=diff_cfg.latent_dim,
    )
    model.to(device)
    schedule = schedule.to(device)

    # Fake video latents:
    # B=2, T=4 frames, each frame -> 8 tokens, latent_dim=128
    video_latents = torch.randn(2, 4, 8, diff_cfg.latent_dim, device=device)

    with torch.no_grad():
        flat_latents = model.video_to_latents(video_latents)  # [B, T*Lf, D]

    B = flat_latents.size(0)
    cond_tokens = torch.zeros(B, 1, diff_cfg.d_model, device=device)
    cond_mask = torch.ones(B, 1, device=device)

    t = get_time_batch(B, schedule)

    loss = model.diffusion_forward(
        x0_latent=flat_latents,
        t=t,
        cond_tokens=cond_tokens,
        cond_mask=cond_mask,
        schedule=schedule,
    )

    print("VIDEO loss:", loss.item())
    assert torch.isfinite(loss), "VIDEO diffusion returned NaN/inf."


# ------------------------------------------------------------
# RUN ALL SANITY CHECKS
# ------------------------------------------------------------
if __name__ == "__main__":
    sanity_text()
    sanity_image()
    sanity_audio()
    sanity_tabular()
    sanity_video()
    print("\n✔ All modalities passed sanity checks.")
