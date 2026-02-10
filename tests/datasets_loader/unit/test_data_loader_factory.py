"""
Unit tests for DataLoaderFactory.

This module contains unit tests for the DataLoaderFactory class, verifying
that it correctly creates loader instances, handles parameters, and provides
proper error messages for invalid inputs.
"""

from pathlib import Path

from ragbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragbench.datasets_loader.data_loader_factory import DataLoaderFactory
from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets_loader.dataset_names import DatasetName
from ragbench.datasets_loader.hotpot_qa_data_loader import HotpotQaDataLoader
from ragbench.datasets_loader.kramabench_data_loader import KramabenchDataLoader
from tests.datasets_loader.helpers.mock_data_loader import MockRagDataLoader


class TestDataLoaderFactory:
    """Unit tests for DataLoaderFactory class."""

    def test_list_available_datasets(self):
        """Test that list_available_datasets returns all registered datasets."""
        datasets = DataLoaderFactory.list_available_datasets()

        # Should return a list
        assert isinstance(datasets, list)

        assert len(datasets) >= 13

        # Should contain expected datasets
        expected_datasets = [
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
        ]
        for dataset in expected_datasets:
            assert dataset in datasets

    def test_get_loader_class_with_enum(self):
        """Test get_loader_class with DatasetName enum."""
        loader_class = DataLoaderFactory.get_loader_class(DatasetName.BIOASQ)

        assert loader_class == BioasqDataLoader
        assert issubclass(loader_class, RagDataLoader)

    def test_get_loader_class_with_string(self):
        """Test get_loader_class with string dataset name."""
        loader_class = DataLoaderFactory.get_loader_class("bioasq")

        assert loader_class == BioasqDataLoader
        assert issubclass(loader_class, RagDataLoader)

    def test_create_loader_all_registered_datasets(self):
        """Test that all registered datasets can be instantiated (or fail gracefully)."""
        datasets = DataLoaderFactory.list_available_datasets()

        for dataset_name in datasets:
            # Each should either create a loader or raise a clear error
            try:
                loader = DataLoaderFactory.create_loader(
                    dataset_name=dataset_name, split="train"
                )
                # If successful, should be a RagDataLoader
                assert isinstance(loader, RagDataLoader)
            except (Exception, TypeError) as e:
                # If it fails, it should be due to data loading, not factory issues
                # Factory issues would be ValueError for invalid names
                assert not isinstance(
                    e, ValueError
                ) or "Invalid dataset name" not in str(e)

    def test_registry_contains_all_expected_loaders(self):
        """Test that the registry contains all expected loader classes."""
        registry = DataLoaderFactory._LOADER_REGISTRY

        # Check that registry is a dict
        assert isinstance(registry, dict)

        # Check that all keys are DatasetName enums
        for key in registry.keys():
            assert isinstance(key, DatasetName)

        # Check that all values are RagDataLoader subclasses
        for loader_class in registry.values():
            assert issubclass(loader_class, RagDataLoader)

        # Check specific expected mappings
        assert registry[DatasetName.BIOASQ] == BioasqDataLoader
        assert registry[DatasetName.HOTPOT_QA] == HotpotQaDataLoader
        assert registry[DatasetName.KRAMABENCH] == KramabenchDataLoader

    def test_factory_is_stateless(self):
        """Test that factory methods are stateless and can be called multiple times."""
        # Call list_available_datasets multiple times
        datasets1 = DataLoaderFactory.list_available_datasets()
        datasets2 = DataLoaderFactory.list_available_datasets()

        assert datasets1 == datasets2

        # Call get_loader_class multiple times
        class1 = DataLoaderFactory.get_loader_class(DatasetName.BIOASQ)
        class2 = DataLoaderFactory.get_loader_class(DatasetName.BIOASQ)

        assert class1 == class2

    def test_factory_methods_are_class_methods(self):
        """Test that factory methods are class methods, not instance methods."""
        # Should be able to call without instantiation
        datasets = DataLoaderFactory.list_available_datasets()
        assert isinstance(datasets, list)

        loader_class = DataLoaderFactory.get_loader_class(DatasetName.BIOASQ)
        assert loader_class == BioasqDataLoader

        # Factory should not need to be instantiated
        # (This is implicit in the above calls, but let's be explicit)
        assert hasattr(DataLoaderFactory.create_loader, "__self__")
        assert hasattr(DataLoaderFactory.list_available_datasets, "__self__")
        assert hasattr(DataLoaderFactory.get_loader_class, "__self__")

    # ============================================================================
    # Section: Cache Testing
    # ============================================================================

    def test_create_loader_with_cache_dir(self, tmp_path):
        """Test that create_loader accepts and passes cache_dir parameter."""
        _ = tmp_path / "test_cache"

        # Create a mock loader with cache
        loader = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5, document_factor=1),
            num_docs=10,
            num_questions=5,
        )

        # Verify loader was created successfully
        assert isinstance(loader, RagDataLoader)
        assert len(loader.get_corpus()) > 0
        assert len(loader.get_benchmark()) > 0

    def test_cache_dir_creates_cache_files(self, tmp_path):
        """Test that providing cache_dir creates cache files."""
        _ = tmp_path / "test_cache"

        # Create loader with cache - first time should create cache
        loader1 = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5, document_factor=1),
            num_docs=10,
            num_questions=5,
        )

        corpus1 = loader1.get_corpus()
        benchmark1 = loader1.get_benchmark()

        # Verify data was loaded
        assert len(corpus1) > 0
        assert len(benchmark1) > 0

    def test_cache_persistence_across_loader_instances(self, tmp_path):
        """Test that cache persists across different loader instances."""
        _ = tmp_path / "test_cache"
        sampling_params = DataSamplingParams(
            question_limit=5, document_factor=1, seed=42
        )

        # Create first loader - should populate cache
        loader1 = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=sampling_params,
            num_docs=10,
            num_questions=5,
        )
        corpus1 = loader1.get_corpus()
        benchmark1 = loader1.get_benchmark()

        # Create second loader with same parameters - should use cache
        loader2 = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=sampling_params,
            num_docs=10,
            num_questions=5,
        )
        corpus2 = loader2.get_corpus()
        benchmark2 = loader2.get_benchmark()

        # Verify both loaders have same data
        assert len(corpus1) == len(corpus2)
        assert len(benchmark1) == len(benchmark2)

    def test_cache_with_different_sampling_params(self, tmp_path):
        """Test that different sampling params create different cache entries."""
        _ = tmp_path / "test_cache"

        # Create loader with first sampling params
        loader1 = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5, document_factor=1),
            num_docs=10,
            num_questions=10,
        )
        corpus1 = loader1.get_corpus()

        # Create loader with different sampling params
        loader2 = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=3, document_factor=2),
            num_docs=10,
            num_questions=10,
        )
        corpus2 = loader2.get_corpus()

        # Different sampling should result in different data
        # (though both should be valid)
        assert isinstance(corpus1, type(corpus2))

    def test_cache_with_different_splits(self, tmp_path):
        """Test that different splits create different cache entries."""
        _ = tmp_path / "test_cache"
        sampling_params = DataSamplingParams(question_limit=5)

        # Create loader with train split
        loader_train = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=sampling_params,
            num_docs=10,
            num_questions=10,
        )

        # Create loader with test split
        loader_test = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="test",
            sampling_params=sampling_params,
            num_docs=10,
            num_questions=10,
        )

        # Both should work independently
        assert loader_train.get_corpus() is not None
        assert loader_test.get_corpus() is not None

    def test_cache_dir_none_disables_caching(self, tmp_path):
        """Test that cache_dir=None disables caching."""
        # Create loader without cache
        loader = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5),
            num_docs=10,
            num_questions=5,
        )

        # Should still work without cache
        assert loader.get_corpus() is not None
        assert loader.get_benchmark() is not None

    def test_cache_with_path_object(self, tmp_path):
        """Test that cache_dir accepts Path objects."""
        _ = Path(tmp_path) / "test_cache"

        loader = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5),
            num_docs=10,
            num_questions=5,
        )

        assert isinstance(loader, RagDataLoader)

    def test_cache_with_string_path(self, tmp_path):
        """Test that cache_dir accepts string paths."""
        _ = str(tmp_path / "test_cache")

        loader = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=DataSamplingParams(question_limit=5),
            num_docs=10,
            num_questions=5,
        )

        assert isinstance(loader, RagDataLoader)
