"""
QGFD_Sanity_Checks.py

Standalone sanity tests for:
  - MultiHeadQGFDLayer (qgfd_attention.py)
  - SafeWrappedAttention / wrap_model_with_qgfd (universal_qgfd_replacer.py)

These tests are meant to be *lightweight* and runnable without internet.
They will:
  1. Build a tiny toy model with a custom Attention module.
  2. Wrap it using wrap_model_with_qgfd.
  3. Check forward pass shapes, gradients, and mask handling.
  4. Optionally, if `transformers` is installed, try wrapping a tiny HF model.

Run directly:

    python QGFD_Sanity_Checks.py
"""

import math
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qgfd_attention import MultiHeadQGFDLayer
from .universal_qgfd_replacer import SafeWrappedAttention, wrap_model_with_qgfd


# ---------------------------------------------------------------------------
# Utility: device & pretty header printing
# ---------------------------------------------------------------------------

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------------
# Tiny attention module (to be wrapped)
# ---------------------------------------------------------------------------

class TinySelfAttention(nn.Module):
    """
    Minimal attention block that matches the heuristics in is_leaf_attention:
     - class name contains "Attention"
     - has q, k, v projections

    Shape:
      hidden_states: (B, L, D)
      Returns: (B, L, D)
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.o = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        B, L, D = hidden_states.shape
        H = self.num_heads
        hd = self.head_dim

        q = self.q(hidden_states).view(B, L, H, hd).transpose(1, 2)  # (B,H,L,hd)
        k = self.k(hidden_states).view(B, L, H, hd).transpose(1, 2)
        v = self.v(hidden_states).view(B, L, H, hd).transpose(1, 2)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(hd)  # (B,H,L,L)

        if attention_mask is not None:
            # support (B,L) mask or broadcasted
            if attention_mask.dtype == torch.bool:
                additive = (~attention_mask).to(scores.dtype) * -1e9
            else:
                additive = attention_mask.to(scores.dtype)
            if additive.dim() == 2:
                additive = additive[:, None, None, :]
            scores = scores + additive

        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v)  # (B,H,L,hd)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.o(out)
        return out


class TinyModel(nn.Module):
    """
    Toy model with:
      - Token embedding
      - 2 layers of TinySelfAttention
      - Final linear head

    This is just to test wrap_model_with_qgfd in a controlled setting.
    """
    def __init__(self, vocab_size=100, embed_dim=32, num_heads=4, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList(
            [TinySelfAttention(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B,L)
        x = self.embed(input_ids)  # (B,L,D)
        for layer in self.layers:
            x = x + layer(x, attention_mask=attention_mask)
        x = self.ln(x)
        logits = self.head(x)
        return logits


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def check_qgfd_forward_shapes():
    print_header("CHECK 1: MultiHeadQGFDLayer forward shapes")

    device = get_device()
    B, Lq, Lk, D = 2, 5, 7, 32
    H = 4

    layer = MultiHeadQGFDLayer(
        embed_dim=D,
        num_heads=H,
        proj_dim=D,
        diffusion_steps=3,
        target_alpha=0.05,
        warmup_steps=10,
        detach_P=False,
        temp=1.0,
    ).to(device)

    hidden_states = torch.randn(B, Lq, D, device=device)
    kv = torch.randn(B, Lk, D, device=device)

    attn_out, p = layer(hidden_states, kv=kv, attention_mask=None)
    print("attn_out.shape:", attn_out.shape)
    print("p.shape:", p.shape)

    assert attn_out.shape == (B, Lq, D), "Unexpected attn_out shape"
    assert p.shape == (B, H, Lq, Lk), "Unexpected attention prob shape"

    # check that step_count increments
    step_before = layer.step_count.item()
    _ = layer(hidden_states, kv=kv)
    step_after = layer.step_count.item()
    print("step_count before/after:", step_before, step_after)
    assert step_after == step_before + 1, "step_count did not increment"

    print("✔ MultiHeadQGFDLayer basic shape & step_count sanity passed.")


def check_qgfd_attention_mask():
    print_header("CHECK 2: MultiHeadQGFDLayer attention mask handling")

    device = get_device()
    B, L, D = 2, 6, 32
    H = 4

    layer = MultiHeadQGFDLayer(
        embed_dim=D,
        num_heads=H,
        proj_dim=D,
        diffusion_steps=2,
        target_alpha=0.1,
        warmup_steps=0,  # alpha on from the start
    ).to(device)

    hidden_states = torch.randn(B, L, D, device=device)

    # build a mask that hides the last token
    # mask: True means "keep", False means "mask out"
    keep_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    keep_mask[:, -1] = False

    attn_out, p = layer(hidden_states, kv=None, attention_mask=keep_mask)
    print("attn_out.shape:", attn_out.shape, "p.shape:", p.shape)

    # we can't do exact equality checks here, but at least ensure it's finite
    assert torch.isfinite(attn_out).all(), "NaNs/Infs in output with mask"
    assert torch.isfinite(p).all(), "NaNs/Infs in attention probs with mask"

    print("✔ MultiHeadQGFDLayer attention_mask sanity passed.")


def check_wrap_tiny_model():
    print_header("CHECK 3: wrap_model_with_qgfd on TinyModel")

    device = get_device()
    torch.manual_seed(0)

    model = TinyModel(vocab_size=50, embed_dim=32, num_heads=4, num_layers=2)
    model.to(device)

    print("Original model:")
    print(model)

    # Wrap attention modules with SafeWrappedAttention + QGFD
    model = wrap_model_with_qgfd(
        model,
        MultiHeadQGFDLayer=MultiHeadQGFDLayer,
        diffusion_steps=3,
        target_alpha=0.05,
        warmup_steps=5,
        detach_P=False,
        temp=1.0,
        verbose=True,
    )

    print("\nWrapped model:")
    print(model)

    # quick forward & backward to check gradients
    B, L = 3, 7
    input_ids = torch.randint(0, 50, (B, L), device=device)
    # simple causal mask example: (B,L)
    causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))[None, :, :].expand(B, L, L)
    # convert to (B,L) style "keep" mask by requiring that each position has at least one True
    # (this is a bit contrived, but just to exercise masking path)
    keep_mask = causal_mask[:, -1, :]  # (B, L)

    model.train()
    logits = model(input_ids, attention_mask=keep_mask)
    target = torch.randint(0, 50, (B, L), device=device)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

    print("Loss:", loss.item())
    loss.backward()

    # check some gradients exist
    grad_norms = []
    for n, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
    print("Num parameters with grad:", len(grad_norms))
    assert len(grad_norms) > 0, "No gradients flowed through wrapped model"

    print("✔ wrap_model_with_qgfd + TinyModel forward/backward sanity passed.")


def check_wrap_hf_model_if_available():
    """
    Optional: if transformers is installed and a small model is available without download,
    we try to wrap it as a more realistic smoke test.

    This is extremely defensive and can be safely ignored if it fails.
    """
    print_header("CHECK 4 (optional): wrap_model_with_qgfd on HF model if transformers is available")

    try:
        from transformers import AutoModelForSeq2SeqLM
    except Exception as e:
        print("transformers not available or import failed; skipping HF test.")
        print("Reason:", repr(e))
        return

    device = get_device()
    model_name = "t5-small"

    try:
        print(f"Attempting to load {model_name} (may require internet or cached weights)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(device)
    except Exception as e:
        print(f"Could not load {model_name}; skipping HF wrap test.")
        print("Reason:", repr(e))
        return

    try:
        model = wrap_model_with_qgfd(
            model,
            MultiHeadQGFDLayer=MultiHeadQGFDLayer,
            diffusion_steps=2,
            target_alpha=0.02,
            warmup_steps=100,
            detach_P=False,
            temp=1.0,
            verbose=True,
        )
        print("HF model wrapped successfully.")
    except Exception as e:
        print("HF model wrapping failed:")
        traceback.print_exc()
        return

    print("✔ HF model wrapping smoke test completed (if model loading succeeded).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        check_qgfd_forward_shapes()
        check_qgfd_attention_mask()
        check_wrap_tiny_model()
        check_wrap_hf_model_if_available()
        print_header("ALL QGFD SANITY CHECKS COMPLETED")
        print("If you see all checkmarks (✔), your QGFD integration is plausibly healthy.")
    except AssertionError as e:
        print_header("SANITY CHECK FAILED (AssertionError)")
        print("Reason:", e)
    except Exception as e:
        print_header("SANITY CHECK FAILED (Exception)")
        traceback.print_exc()
