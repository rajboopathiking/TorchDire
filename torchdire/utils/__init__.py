"""
Utilities for model wrapping and attention layer replacement.
"""

from torchdire.utils.replacer import SafeWrappedAttention, wrap_model_with_qgfd

__all__ = ["SafeWrappedAttention", "wrap_model_with_qgfd"]
