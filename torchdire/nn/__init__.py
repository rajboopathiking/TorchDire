"""
Neural network modules for QGFD attention.
"""

from torchdire.nn.qgfd import MultiHeadQGFDLayer
from torchdire.nn.gating import QGFDMultiHeadAttention

__all__ = ["MultiHeadQGFDLayer", "QGFDMultiHeadAttention"]
