"""
RAGBench - A comprehensive benchmarking framework for Retrieval-Augmented Generation (RAG) systems.
"""

from ragbench.datasets_loader import DataLoaderFactory, DatasetName, RagDataLoader
from ragbench.experiment import Experiment

__version__ = "0.1.0"

__all__ = [
    "DataLoaderFactory",
    "DatasetName",
    "RagDataLoader",
    "Experiment",
]
