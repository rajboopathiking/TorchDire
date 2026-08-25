"""
TorchDire: Query-Graph Flow Diffusion (QGFD) Ecosystem for PyTorch
================================================-------------------
A research and development library for diffusion-regularized attention mechanisms in Transformers.
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
from torchdire.nn.qgfd_kernel import (
    QGFDKernel,
    QGFDStepCallback,
    collect_qgfd_kernels,
    dump_learned_alphas,
    register_qgfd_step_callback,
    unfreeze_qgfd_alpha,
)
from torchdire.nn.qgfd import MultiHeadQGFDLayer
from torchdire.nn.gating import QGFDMultiHeadAttention
from torchdire.nn.llama_qgfd import LlamaQGFDAttention, patch_llama_with_qgfd
from torchdire.utils.replacer import SafeWrappedAttention, wrap_model_with_qgfd

# Unified replacer (supports both approaches)
from torchdire.universal_qgfd_replacer import (
    wrap_model_with_qgfd_operator,
    DEFAULT_QGFD_CONFIG,
)

from torchdire.theory.verifier import verify_qgfd_theorems, QGFDTheoremVerifier
from torchdire.profiler.efficiency import profile_qgfd_efficiency, QGFDProfiler
from torchdire.experiments.ablation import QGFDAblator, run_ablation_study
from torchdire.benchmarks.tradeoff import compare_qgfd_vs_softmax, run_single_benchmark

__version__ = "1.1.0"
__author__ = "Raj Boopathi"

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
    "AttentionOperatorAdapter",
    "LlamaAttentionAdapter",
    "Qwen2AttentionAdapter",
    "MistralAttentionAdapter",
    "GPTNeoXAttentionAdapter",
    
    # Legacy kernel-based API
    "QGFDKernel",
    "QGFDStepCallback",
    "collect_qgfd_kernels",
    "dump_learned_alphas",
    "register_qgfd_step_callback",
    "unfreeze_qgfd_alpha",
    "MultiHeadQGFDLayer",
    "QGFDMultiHeadAttention",
    "LlamaQGFDAttention",
    "patch_llama_with_qgfd",
    "SafeWrappedAttention",
    "wrap_model_with_qgfd",
    
    # Theory & Benchmarks
    "verify_qgfd_theorems",
    "QGFDTheoremVerifier",
    "profile_qgfd_efficiency",
    "QGFDProfiler",
    "QGFDAblator",
    "run_ablation_study",
    "compare_qgfd_vs_softmax",
    "run_single_benchmark",
]

