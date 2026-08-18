"""
Universal QGFD replacer — drop-in replacement of softmax attention with
Query-Graph Flow Diffusion (QGFD) across arbitrary PyTorch/HuggingFace models.

Entry points
------------
- wrap_model_with_qgfd(model, ...): auto-detects Llama (zero-overhead
  subclassing via patch_llama_with_qgfd) or any other architecture
  (SafeWrappedAttention).
- patch_llama_with_qgfd(model, ...): direct Llama replacement reusing the
  original q/k/v/o projections (keeps LoRA/QLoRA adapters intact).
- LlamaQGFDAttention / SafeWrappedAttention: per-layer building blocks.

Canonical defaults (validated by QGFD_Sanity_Checks)
-----------------------------------------------------
Stronger diffusion (e.g. alpha=0.1, diffusion_steps=3) degrades eval loss vs
softmax while only adding compute; the proven configuration is a gentle
single-step diffusion:

    diffusion_steps = 1      # one p <- (1-alpha)*p0 + alpha*p0*P step
    target_alpha   = 0.05    # fraction of key-graph diffusion mass
    warmup_steps   = 0       # alpha at full strength immediately (no silent
                             # softmax period during training)
    early_stop_eps = 0.0     # no data-dependent early stopping
    mode           = "full"  # full O(L^2) key-transition diffusion up to
    max_full_seq_len = 512   # max_full_seq_len, then...
    full_fallback_mode = "conv"  # ...linear-cost local diffusion (O(L*k))
                             # instead of silently disabling QGFD at long ctx

Usage
-----
    from universal_qgfd_replacer import wrap_model_with_qgfd, DEFAULT_QGFD_CONFIG

    model = wrap_model_with_qgfd(model, **DEFAULT_QGFD_CONFIG, verbose=True)
"""
from torchdire.utils.replacer import SafeWrappedAttention, wrap_model_with_qgfd, is_leaf_attention
from torchdire.nn.llama_qgfd import LlamaQGFDAttention, patch_llama_with_qgfd

DEFAULT_QGFD_CONFIG = {
    "diffusion_steps": 1,
    "target_alpha": 0.05,
    "warmup_steps": 0,
    "early_stop_eps": 0.0,
    "max_full_seq_len": 512,
    "full_fallback_mode": "conv",
}

__all__ = [
    "SafeWrappedAttention",
    "wrap_model_with_qgfd",
    "is_leaf_attention",
    "LlamaQGFDAttention",
    "patch_llama_with_qgfd",
    "DEFAULT_QGFD_CONFIG",
]