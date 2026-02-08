"""
Integration tests for DataLoaderFactory with real data loading.

This module contains integration tests that verify the DataLoaderFactory
can successfully create loaders and load real data from various sources.
"""

import pytest

from ragbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragbench.datasets_loader.data_loader_factory import DataLoaderFactory
from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus
from ragbench.datasets_loader.dataset_names import DatasetName


@pytest.mark.integration
class TestDataLoaderFactoryIntegration:
    """Integration tests for DataLoaderFactory with real data."""

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "bioasq",
            "clap_nq",
            "da_code",
            "dabstep",
            "hotpot_qa",
            "kramabench",
            "mini_wiki",
            "mldr",
            "narrative_qa",
            "officeqa",
            "qasper",
            "secque",
            "watsonx_docs_qa",
        ],
    )
    def test_factory_creates_working_loader(self, dataset_name: str):
        """
        Test that factory creates a working loader for each dataset.

        This test verifies that:
        1. The factory can create a loader instance
        2. The loader is of the correct type
        3. The loader can retrieve corpus and benchmark
        4. The data structures are valid
        """
        # Create loader with small sample to speed up test
        loader = DataLoaderFactory.create_loader(
            dataset_name=dataset_name,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5, document_factor=2),
        )

        # Verify loader is correct type
        assert isinstance(loader, RagDataLoader)
        assert loader.dataset_name.value == dataset_name

        # Verify corpus can be retrieved
        corpus = loader.get_corpus()
        assert isinstance(corpus, RagCorpus)
        assert len(corpus) > 0

        # Verify benchmark can be retrieved
        benchmark = loader.get_benchmark()
        assert isinstance(benchmark, RagBenchmark)
        assert len(benchmark) > 0

    @pytest.mark.parametrize("split", ["train", "test", None])
    def test_factory_respects_split_parameter(self, split):
        """Test that factory correctly passes split parameter to loaders."""
        loader = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.BIOASQ,
            split=split,
            sampling_params=DataSamplingParams(question_limit=5),
        )

        assert loader.split == split

    def test_factory_with_sampling_params(self):
        """Test that factory correctly applies sampling parameters."""
        sampling_params = DataSamplingParams(
            question_limit=10, document_factor=2, seed=42
        )

        loader = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=sampling_params,
        )

        benchmark = loader.get_benchmark()
        # Should have at most 10 questions due to sampling
        assert len(benchmark) <= 10

    def test_factory_with_hotpot_qa_level_parameter(self):
        """Test that factory passes loader-specific parameters (HotpotQA level)."""
        loader = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.HOTPOT_QA,
            split="train",
            level="hard",
            sampling_params=DataSamplingParams(question_limit=5),
        )

        assert isinstance(loader, RagDataLoader)
        # Verify loader was created successfully (level is stored internally)
        assert loader.dataset_name == DatasetName.HOTPOT_QA

    def test_factory_with_kramabench_verbose_parameter(self):
        """Test that factory passes loader-specific parameters (Kramabench verbose)."""
        loader = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.KRAMABENCH,
            split="train",
            verbose=False,
            progress_every=100,
            sampling_params=DataSamplingParams(question_limit=5),
        )

        assert isinstance(loader, RagDataLoader)
        # Verify loader was created successfully (verbose/progress_every are stored internally)
        assert loader.dataset_name == DatasetName.KRAMABENCH

    def test_factory_creates_different_loaders_for_different_datasets(self):
        """Test that factory creates different loader instances for different datasets."""
        loader1 = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5),
        )

        loader2 = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.HOTPOT_QA,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5),
        )

        # Should be different instances
        assert loader1 is not loader2

        # Should have different dataset names
        assert loader1.dataset_name != loader2.dataset_name

        # Should have different data
        corpus1 = loader1.get_corpus()
        corpus2 = loader2.get_corpus()
        assert corpus1 is not corpus2

    def test_factory_with_enum_and_string_produce_same_result(self):
        """Test that using enum or string dataset name produces equivalent loaders."""
        sampling_params = DataSamplingParams(question_limit=5, seed=42)

        loader_enum = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=sampling_params,
        )

        loader_string = DataLoaderFactory.create_loader(
            dataset_name="bioasq", split="train", sampling_params=sampling_params
        )

        # Should have same dataset name
        assert loader_enum.dataset_name == loader_string.dataset_name

        # Should have same split
        assert loader_enum.split == loader_string.split

        # Should load same amount of data (with same seed)
        assert len(loader_enum.get_corpus()) == len(loader_string.get_corpus())
        assert len(loader_enum.get_benchmark()) == len(loader_string.get_benchmark())

    def test_factory_loader_corpus_and_benchmark_are_consistent(self):
        """Test that corpus and benchmark from factory-created loader are consistent."""
        loader = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=10),
        )

        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Get all document IDs referenced in benchmark
        benchmark_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)

        # Get all document IDs in corpus
        corpus_doc_ids = {doc.name for doc in corpus.documents}

        # All benchmark doc IDs should exist in corpus
        # (This is a key integrity check)
        missing_docs = benchmark_doc_ids - corpus_doc_ids
        assert (
            len(missing_docs) == 0
        ), f"Benchmark references documents not in corpus: {missing_docs}"

    def test_factory_multiple_calls_create_independent_loaders(self):
        """Test that multiple factory calls create independent loader instances."""
        loader1 = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5, seed=42),
        )

        loader2 = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5, seed=42),
        )

        # Should be different instances
        assert loader1 is not loader2

        # But should have same data (same seed)
        assert len(loader1.get_corpus()) == len(loader2.get_corpus())
        assert len(loader1.get_benchmark()) == len(loader2.get_benchmark())

    def test_factory_with_no_sampling_loads_full_dataset(self):
        """Test that factory without sampling params loads full dataset."""
        # Create loader without sampling
        loader_full = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.BIOASQ, split="train", sampling_params=None
        )

        # Create loader with sampling
        loader_sampled = DataLoaderFactory.create_loader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5),
        )

        # Full dataset should have more questions
        assert len(loader_full.get_benchmark()) > len(loader_sampled.get_benchmark())

    @pytest.mark.parametrize(
        "dataset_name,expected_min_docs,expected_min_questions",
        [
            ("bioasq", 10, 5),  # Expect at least some data
            ("hotpot_qa", 10, 5),
            ("kramabench", 10, 5),
        ],
    )
    def test_factory_loads_minimum_expected_data(
        self, dataset_name: str, expected_min_docs: int, expected_min_questions: int
    ):
        """Test that factory-created loaders load minimum expected amounts of data."""
        loader = DataLoaderFactory.create_loader(
            dataset_name=dataset_name,
            split="train",
            sampling_params=DataSamplingParams(question_limit=20),
        )

        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        assert (
            len(corpus) >= expected_min_docs
        ), f"Expected at least {expected_min_docs} documents, got {len(corpus)}"
        assert (
            len(benchmark) >= expected_min_questions
        ), f"Expected at least {expected_min_questions} questions, got {len(benchmark)}"

    def test_factory_error_handling_with_invalid_parameters(self):
        """Test that factory provides clear errors for invalid parameters."""
        # Invalid dataset name
        with pytest.raises(ValueError) as exc_info:
            DataLoaderFactory.create_loader(dataset_name="invalid_dataset")
        assert "Invalid dataset name" in str(exc_info.value)

        # Invalid split (this might be caught by type checker, but test runtime too)
        # Note: This test depends on loader implementation
        # Some loaders might accept invalid splits, so we skip this for now

    def test_factory_list_available_datasets_matches_working_loaders(self):
        """Test that all datasets listed as available can actually be loaded."""
        available_datasets = DataLoaderFactory.list_available_datasets()

        # Try to create a loader for each available dataset
        for dataset_name in available_datasets:
            try:
                loader = DataLoaderFactory.create_loader(
                    dataset_name=dataset_name,
                    split="train",
                    sampling_params=DataSamplingParams(question_limit=2),
                )
                assert isinstance(loader, RagDataLoader)
            except Exception as e:
                pytest.fail(
                    f"Failed to create loader for '{dataset_name}' "
                    f"which is listed as available: {e}"
                )
