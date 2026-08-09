"""
Benchmark datasets and standardized trainer for QGFD R&D evaluation.
"""

from torchdire.benchmarks.dataset import (
    GraphMultiHopDataset,
    PasskeyRetrievalDataset,
    TextSummarizationDataset,
)
from torchdire.benchmarks.trainer import QGFDTrainer
from torchdire.benchmarks.tradeoff import (
    compare_qgfd_vs_softmax,
    run_single_benchmark,
    compute_perplexity,
    measure_generation_performance,
)

__all__ = [
    "GraphMultiHopDataset",
    "PasskeyRetrievalDataset",
    "TextSummarizationDataset",
    "QGFDTrainer",
    "compare_qgfd_vs_softmax",
    "run_single_benchmark",
    "compute_perplexity",
    "measure_generation_performance",
]

