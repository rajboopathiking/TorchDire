"""
Benchmark datasets and standardized trainer for QGFD R&D evaluation.
"""

from torchdire.benchmarks.dataset import (
    GraphMultiHopDataset,
    PasskeyRetrievalDataset,
    TextSummarizationDataset,
)
from torchdire.benchmarks.trainer import QGFDTrainer

__all__ = [
    "GraphMultiHopDataset",
    "PasskeyRetrievalDataset",
    "TextSummarizationDataset",
    "QGFDTrainer",
]
