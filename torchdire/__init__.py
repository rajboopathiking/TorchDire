"""
TorchDire: Query-Graph Flow Diffusion (QGFD) Ecosystem for PyTorch
================================================-------------------
A research and development library for diffusion-regularized attention mechanisms in Transformers.
"""

from torchdire.nn.qgfd_kernel import (
    QGFDKernel,
    QGFDStepCallback,
    collect_qgfd_kernels,
    register_qgfd_step_callback,
    unfreeze_qgfd_alpha,
)
from torchdire.nn.qgfd import MultiHeadQGFDLayer
from torchdire.nn.gating import QGFDMultiHeadAttention
from torchdire.nn.llama_qgfd import LlamaQGFDAttention, patch_llama_with_qgfd
from torchdire.utils.replacer import SafeWrappedAttention, wrap_model_with_qgfd
from torchdire.theory.verifier import verify_qgfd_theorems, QGFDTheoremVerifier
from torchdire.profiler.efficiency import profile_qgfd_efficiency, QGFDProfiler
from torchdire.experiments.ablation import QGFDAblator, run_ablation_study
from torchdire.benchmarks.tradeoff import compare_qgfd_vs_softmax, run_single_benchmark

__version__ = "1.0.1"
__author__ = "Raj Boopathi"

__all__ = [
    "QGFDKernel",
    "QGFDStepCallback",
    "collect_qgfd_kernels",
    "register_qgfd_step_callback",
    "unfreeze_qgfd_alpha",
    "MultiHeadQGFDLayer",
    "QGFDMultiHeadAttention",
    "LlamaQGFDAttention",
    "patch_llama_with_qgfd",
    "SafeWrappedAttention",
    "wrap_model_with_qgfd",
    "verify_qgfd_theorems",
    "QGFDTheoremVerifier",
    "profile_qgfd_efficiency",
    "QGFDProfiler",
    "QGFDAblator",
    "run_ablation_study",
    "compare_qgfd_vs_softmax",
    "run_single_benchmark",
]

