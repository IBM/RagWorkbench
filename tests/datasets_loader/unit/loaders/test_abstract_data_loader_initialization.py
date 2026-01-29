"""
Tests for RagDataLoader initialization.

This module tests the initialization behavior of the RagDataLoader abstract
base class, including split handling and corpus/benchmark creation.
"""

from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus
from tests.datasets_loader.helpers.mock_data_loader import MockRagDataLoader


class TestRagDataLoaderInitialization:
    """Test suite for RagDataLoader initialization."""

    def test_initialization_with_default_sampling(self):
        """Test initialization with default sampling parameters (no sampling)."""
        loader = MockRagDataLoader(num_docs=10, num_questions=8)

        # With default params (no sampling), all data should be included
        assert len(loader.get_benchmark()) == 8
        assert len(loader.get_corpus()) == 10

    def test_split_handling_train(self):
        """Test initialization with 'train' split."""
        loader = MockRagDataLoader(split="train", num_docs=15, num_questions=10)

        assert loader.split == "train"
        assert isinstance(loader.get_benchmark(), RagBenchmark)
        assert isinstance(loader.get_corpus(), RagCorpus)

    def test_split_handling_test(self):
        """Test initialization with 'test' split."""
        loader = MockRagDataLoader(split="test", num_docs=15, num_questions=10)

        assert loader.split == "test"
        assert isinstance(loader.get_benchmark(), RagBenchmark)
        assert isinstance(loader.get_corpus(), RagCorpus)

    def test_split_handling_none(self):
        """Test initialization with None split (full dataset)."""
        loader = MockRagDataLoader(split=None, num_docs=15, num_questions=10)

        assert loader.split is None
        assert isinstance(loader.get_benchmark(), RagBenchmark)
        assert isinstance(loader.get_corpus(), RagCorpus)

    def test_corpus_and_benchmark_creation(self):
        """Test that corpus and benchmark are properly created during initialization."""
        loader = MockRagDataLoader(num_docs=12, num_questions=8)

        # Check that instances are created
        assert hasattr(loader, "benchmark")
        assert hasattr(loader, "rag_corpus")
        assert isinstance(loader.benchmark, RagBenchmark)
        assert isinstance(loader.rag_corpus, RagCorpus)

        # Check that they contain data
        assert len(loader.benchmark) > 0
        assert len(loader.rag_corpus) > 0

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters specified."""
        from ragbench.datasets_loader.data_models.data_sampling_params import (
            DataSamplingParams,
        )
        from ragbench.datasets_loader.data_models.dataset_names import DatasetName

        sampling_params = DataSamplingParams(
            question_limit=5, document_factor=2, seed=42
        )
        loader = MockRagDataLoader(
            dataset_name=DatasetName.AI_ARXIV,
            split="train",
            sampling_params=sampling_params,
            num_docs=20,
            num_questions=15,
        )

        assert loader.dataset_name == DatasetName.AI_ARXIV
        assert loader.split == "train"
        assert len(loader.get_benchmark()) == 5  # question_limit applied
        assert isinstance(loader.get_corpus(), RagCorpus)
