"""
Universal QGFD Replacer — Operator-Based Architecture.

This module provides a clean operator-based approach for replacing only the
attention probability computation (softmax) with QGFD, while preserving all
other attention mechanics (projections, RoPE, KV cache, GQA, causal masking).

Entry Points
------------
- wrap_model_with_qgfd_operator(model, operator, ...): Apply a probability operator
- wrap_model_with_qgfd(model, ...): Backward-compatible QGFD-specific wrapper
- patch_llama_with_qgfd(model, ...): Legacy direct Llama subclassing (deprecated)
- AttentionProbabilityOperator, SoftmaxOperator, QGFDOperator: Operator classes

Canonical Defaults (validated by QGFD_Sanity_Checks)
-----------------------------------------------------
    diffusion_steps = 1      # one p <- (1-alpha)*p0 + alpha*p0*P step
    target_alpha   = 0.05    # fraction of key-graph diffusion mass
    warmup_steps   = 0       # alpha at full strength immediately
    early_stop_eps = 0.0     # no data-dependent early stopping
    mode           = "full"  # full O(L^2) key-transition diffusion up to
    max_full_seq_len = 512   # max_full_seq_len, then...
    full_fallback_mode = "conv"  # linear-cost local diffusion (O(L*k))
                               # instead of silently disabling QGFD at long ctx
    detach_P       = True    # stop gradients through transition matrix

Usage
-----
    from universal_qgfd_replacer import wrap_model_with_qgfd_operator, QGFDOperator, DEFAULT_QGFD_CONFIG
    from torchdire.nn.attention_operators import SoftmaxOperator

    # Apply QGFD operator
    operator = QGFDOperator(**DEFAULT_QGFD_CONFIG)
    model = wrap_model_with_qgfd_operator(model, operator, verbose=True)

    # Or apply plain softmax (for ablation)
    operator = SoftmaxOperator()
    model = wrap_model_with_qgfd_operator(model, operator, verbose=True)
"""

import torch.nn as nn
from torchdire.nn.attention_operators import (
    AttentionProbabilityOperator,
    SoftmaxOperator,
    QGFDOperator,
    collect_qgfd_operators,
    register_qgfd_operator_step_callback,
)
from torchdire.nn.attention_adapters import (
    patch_model_with_operator,
    create_attention_adapter,
    AttentionOperatorAdapter,
    LlamaAttentionAdapter,
    Qwen2AttentionAdapter,
    MistralAttentionAdapter,
    GPTNeoXAttentionAdapter,
)

# Legacy imports (deprecated - use operator-based approach instead)
try:
    from torchdire.utils.replacer import SafeWrappedAttention, wrap_model_with_qgfd as _legacy_wrap, is_leaf_attention
    from torchdire.nn.llama_qgfd import LlamaQGFDAttention, patch_llama_with_qgfd
    HAS_LEGACY = True
except ImportError:
    HAS_LEGACY = False
    SafeWrappedAttention = None
    LlamaQGFDAttention = None
    patch_llama_with_qgfd = None
    is_leaf_attention = None

DEFAULT_QGFD_CONFIG = {
    "diffusion_steps": 1,
    "target_alpha": 0.05,
    "warmup_steps": 0,
    "early_stop_eps": 0.0,
    "detach_P": True,
    "mode": "full",
    "kernel_size": 5,
    "enable_qgfd": True,
    "max_alpha": 0.10,
    "max_full_seq_len": 512,
    "full_fallback_mode": "conv",
    "mask_threshold": -1e4,
    "debug": False,
}


def wrap_model_with_qgfd_operator(
    model,
    operator: AttentionProbabilityOperator,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply an AttentionProbabilityOperator to all attention layers in the model.
    
    This is the recommended operator-based approach that preserves all
    attention architecture mechanics and only replaces the softmax computation.
    
    Args:
        model: The model to patch
        operator: An AttentionProbabilityOperator instance (e.g., QGFDOperator, SoftmaxOperator)
        verbose: Whether to print progress
        
    Returns:
        The patched model (modified in-place)
    """
    import gc
    gc.collect()

    if hasattr(model, "config"):
        model.config.use_cache = True

    return patch_model_with_operator(model, operator, verbose=verbose)


def wrap_model_with_qgfd(
    model,
    diffusion_steps: int = 1,
    target_alpha: float = 0.05,
    warmup_steps: int = 0,
    early_stop_eps: float = 0.0,
    detach_P: bool = True,
    mode: str = "full",
    kernel_size: int = 5,
    enable_qgfd: bool = True,
    max_alpha: float = 0.10,
    max_full_seq_len: int = 512,
    full_fallback_mode: str = "conv",
    mask_threshold: float = -1e4,
    debug: bool = False,
    verbose: bool = True,
    use_operator_approach: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Backward-compatible QGFD wrapper.
    
    If use_operator_approach=True (default), uses the new operator-based architecture
    that preserves all attention mechanics. If False, falls back to legacy SafeWrappedAttention.
    """
    if use_operator_approach:
        operator = QGFDOperator(
            diffusion_steps=diffusion_steps,
            target_alpha=target_alpha,
            warmup_steps=warmup_steps,
            early_stop_eps=early_stop_eps,
            detach_P=detach_P,
            mode=mode,
            kernel_size=kernel_size,
            enable_qgfd=enable_qgfd,
            max_alpha=max_alpha,
            max_full_seq_len=max_full_seq_len,
            full_fallback_mode=full_fallback_mode,
            mask_threshold=mask_threshold,
            debug=debug,
            **kwargs,
        )
        return wrap_model_with_qgfd_operator(model, operator, verbose=verbose)
    
    # Legacy fallback
    if not HAS_LEGACY:
        raise RuntimeError("Legacy replacer not available. Use use_operator_approach=True.")
    return _legacy_wrap(
        model,
        diffusion_steps=diffusion_steps,
        target_alpha=target_alpha,
        warmup_steps=warmup_steps,
        early_stop_eps=early_stop_eps,
        kernel_size=kernel_size,
        max_full_seq_len=max_full_seq_len,
        full_fallback_mode=full_fallback_mode,
        verbose=verbose,
        **kwargs,
    )


__all__ = [
    # Operator-based API (recommended)
    "AttentionProbabilityOperator",
    "SoftmaxOperator",
    "QGFDOperator",
    "wrap_model_with_qgfd_operator",
    "patch_model_with_operator",
    "create_attention_adapter",
    "collect_qgfd_operators",
    "register_qgfd_operator_step_callback",
    "DEFAULT_QGFD_CONFIG",
    
    # Adapter classes
    "AttentionOperatorAdapter",
    "LlamaAttentionAdapter",
    "Qwen2AttentionAdapter",
    "MistralAttentionAdapter",
    "GPTNeoXAttentionAdapter",
    
    # Legacy API (deprecated)
    "SafeWrappedAttention",
    "wrap_model_with_qgfd",
    "is_leaf_attention",
    "LlamaQGFDAttention",
    "patch_llama_with_qgfd",
]