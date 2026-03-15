"""
RAG Benchmark Dataset Loaders.

This module provides data loaders for various RAG benchmark datasets, along with
a factory class for creating loader instances. It also supports registration of
custom data loaders for external datasets.

Example:
    Using built-in datasets:
    >>> from ragworkbench.datasets_loader import DataLoaderFactory, DatasetName
    >>> loader = DataLoaderFactory.create_loader(DatasetName.BIOASQ, split="train")
    >>> corpus = loader.get_corpus()
    >>> benchmark = loader.get_benchmark()

    Registering and using custom datasets:
    >>> from ragworkbench.datasets_loader import DataLoaderFactory, RagDataLoader
    >>> class MyLoader(RagDataLoader):
    ...     def _get_documents(self): return [...]
    ...     def _get_benchmark_entries(self, split): return [...]
    >>> DataLoaderFactory.register_loader("my_dataset", MyLoader)
    >>> loader = DataLoaderFactory.create_loader("my_dataset", split="train")
"""

from ragworkbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragworkbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragworkbench.datasets_loader.data_loader_factory import DataLoaderFactory
from ragworkbench.datasets_loader.dataset_names import DatasetName
from ragworkbench.datasets_loader.hotpot_qa_data_loader import HotpotQaDataLoader
from ragworkbench.datasets_loader.kramabench_data_loader import KramabenchDataLoader
from ragworkbench.datasets_loader.watsonx_data_loader import WatsonxDocsQADataLoader

__all__ = [
    "DataLoaderFactory",
    "RagDataLoader",
    "DatasetName",
]
