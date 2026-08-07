"""
Neural network modules and kernels for QGFD attention.
"""

from torchdire.nn.qgfd_kernel import QGFDKernel
from torchdire.nn.qgfd import MultiHeadQGFDLayer
from torchdire.nn.gating import QGFDMultiHeadAttention
from torchdire.nn.llama_qgfd import LlamaQGFDAttention, patch_llama_with_qgfd

__all__ = [
    "QGFDKernel",
    "MultiHeadQGFDLayer",
    "QGFDMultiHeadAttention",
    "LlamaQGFDAttention",
    "patch_llama_with_qgfd",
]
