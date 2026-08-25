"""
Neural network modules and kernels for QGFD attention.
"""

# Operator-based API (recommended)
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

# Legacy kernel-based API
from torchdire.nn.qgfd_kernel import QGFDKernel
from torchdire.nn.qgfd import MultiHeadQGFDLayer
from torchdire.nn.gating import QGFDMultiHeadAttention
from torchdire.nn.llama_qgfd import LlamaQGFDAttention, patch_llama_with_qgfd

__all__ = [
    # Operator-based API (recommended)
    "AttentionProbabilityOperator",
    "SoftmaxOperator",
    "QGFDOperator",
    "collect_qgfd_operators",
    "register_qgfd_operator_step_callback",
    "patch_model_with_operator",
    "create_attention_adapter",
    "AttentionOperatorAdapter",
    "LlamaAttentionAdapter",
    "Qwen2AttentionAdapter",
    "MistralAttentionAdapter",
    "GPTNeoXAttentionAdapter",
    
    # Legacy kernel-based API
    "QGFDKernel",
    "MultiHeadQGFDLayer",
    "QGFDMultiHeadAttention",
    "LlamaQGFDAttention",
    "patch_llama_with_qgfd",
]
