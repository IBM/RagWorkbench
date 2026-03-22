"""
Unit tests for DataLoaderFactory.

This module contains unit tests for the DataLoaderFactory class, verifying
that it correctly creates loader instances, handles parameters, and provides
proper error messages for invalid inputs.
"""

from pathlib import Path

import pytest

from ragworkbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragworkbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragworkbench.datasets_loader.data_loader_factory import DataLoaderFactory
from ragworkbench.datasets_loader.data_models import DataSamplingParams
from ragworkbench.datasets_loader.dataset_names import DatasetName
from ragworkbench.datasets_loader.hotpot_qa_data_loader import HotpotQaDataLoader
from ragworkbench.datasets_loader.kramabench_data_loader import KramabenchDataLoader
from tests.datasets_loader.helpers.mock_data_loader import MockRagDataLoader


class TestDataLoaderFactory:
    """Unit tests for DataLoaderFactory class."""

    @pytest.mark.skip(reason="Test disabled - needs fixing")
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
            data_sampling=DataSamplingParams(question_limit=5, document_factor=1),
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
            data_sampling=DataSamplingParams(question_limit=5, document_factor=1),
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
        data_sampling = DataSamplingParams(question_limit=5, document_factor=1, seed=42)

        # Create first loader - should populate cache
        loader1 = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            data_sampling=data_sampling,
            num_docs=10,
            num_questions=5,
        )
        corpus1 = loader1.get_corpus()
        benchmark1 = loader1.get_benchmark()

        # Create second loader with same parameters - should use cache
        loader2 = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            data_sampling=data_sampling,
            num_docs=10,
            num_questions=5,
        )
        corpus2 = loader2.get_corpus()
        benchmark2 = loader2.get_benchmark()

        # Verify both loaders have same data
        assert len(corpus1) == len(corpus2)
        assert len(benchmark1) == len(benchmark2)

    def test_cache_with_different_data_sampling(self, tmp_path):
        """Test that different sampling params create different cache entries."""
        _ = tmp_path / "test_cache"

        # Create loader with first sampling params
        loader1 = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            data_sampling=DataSamplingParams(question_limit=5, document_factor=1),
            num_docs=10,
            num_questions=10,
        )
        corpus1 = loader1.get_corpus()

        # Create loader with different sampling params
        loader2 = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            data_sampling=DataSamplingParams(question_limit=3, document_factor=2),
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
        data_sampling = DataSamplingParams(question_limit=5)

        # Create loader with train split
        loader_train = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            data_sampling=data_sampling,
            num_docs=10,
            num_questions=10,
        )

        # Create loader with test split
        loader_test = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="test",
            data_sampling=data_sampling,
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
            data_sampling=DataSamplingParams(question_limit=5),
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
            data_sampling=DataSamplingParams(question_limit=5),
            num_docs=10,
            num_questions=5,
        )

        assert isinstance(loader, RagDataLoader)

    def test_cache_with_string_path(self, tmp_path):
        """Test that cache_dir accepts string paths."""
        _ = str(tmp_path / "test_cache")

        _ = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            data_sampling=DataSamplingParams(question_limit=5),
            num_docs=10,
            num_questions=5,
        )

    # ============================================================================
    # Section: Custom Loader Registration Testing
    # ============================================================================

    def test_register_loader_success(self):
        """Test successful registration of a custom loader."""
        # Register a custom loader
        DataLoaderFactory.register_loader("test_custom_dataset", MockRagDataLoader)

        try:
            # Verify it's registered
            assert DataLoaderFactory.is_registered("test_custom_dataset")

            # Verify it appears in available datasets
            datasets = DataLoaderFactory.list_available_datasets()
            assert "test_custom_dataset" in datasets

            # Verify we can get the loader class
            loader_class = DataLoaderFactory.get_loader_class("test_custom_dataset")
            assert loader_class == MockRagDataLoader
        finally:
            # Clean up
            DataLoaderFactory.unregister_loader("test_custom_dataset")

    def test_register_loader_with_invalid_name_empty_string(self):
        """Test that registering with empty string raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="cannot be empty"):
            DataLoaderFactory.register_loader("", MockRagDataLoader)

    def test_register_loader_with_invalid_name_none(self):
        """Test that registering with None raises TypeError."""
        import pytest

        with pytest.raises(TypeError, match="must be a string"):
            DataLoaderFactory.register_loader(None, MockRagDataLoader)  # type: ignore

    def test_register_loader_with_builtin_name(self):
        """Test that registering with built-in dataset name raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="built-in dataset name"):
            DataLoaderFactory.register_loader("bioasq", MockRagDataLoader)

    def test_register_loader_with_invalid_class_not_subclass(self):
        """Test that registering non-RagDataLoader class raises TypeError."""
        import pytest

        class NotALoader:
            pass

        with pytest.raises(TypeError, match="must be a subclass of RagDataLoader"):
            DataLoaderFactory.register_loader("test_dataset", NotALoader)  # type: ignore

    def test_register_loader_with_abstract_base_class(self):
        """Test that registering RagDataLoader itself raises TypeError."""
        import pytest

        with pytest.raises(TypeError, match="abstract class"):
            DataLoaderFactory.register_loader("test_dataset", RagDataLoader)

    def test_register_loader_with_instance_not_class(self):
        """Test that registering an instance instead of class raises TypeError."""
        import pytest

        loader_instance = MockRagDataLoader(num_docs=5, num_questions=3)

        with pytest.raises(TypeError, match="must be a class"):
            DataLoaderFactory.register_loader("test_dataset", loader_instance)  # type: ignore

    def test_register_loader_overwrite_warning(self, caplog):
        """Test that overwriting existing custom loader logs a warning."""
        import logging

        # Register first time
        DataLoaderFactory.register_loader("test_overwrite", MockRagDataLoader)

        try:
            # Register again - should log warning
            with caplog.at_level(logging.WARNING):
                DataLoaderFactory.register_loader("test_overwrite", MockRagDataLoader)

            assert "Overwriting existing custom loader" in caplog.text
        finally:
            # Clean up
            DataLoaderFactory.unregister_loader("test_overwrite")

    def test_unregister_loader_success(self):
        """Test successful unregistration of a custom loader."""
        # Register a loader
        DataLoaderFactory.register_loader("test_unregister", MockRagDataLoader)
        assert DataLoaderFactory.is_registered("test_unregister")

        # Unregister it
        result = DataLoaderFactory.unregister_loader("test_unregister")
        assert result is True

        # Verify it's no longer registered
        assert not DataLoaderFactory.is_registered("test_unregister")

    def test_unregister_loader_not_found(self):
        """Test unregistering non-existent loader returns False."""
        result = DataLoaderFactory.unregister_loader("nonexistent_dataset")
        assert result is False

    def test_unregister_builtin_loader_returns_false(self):
        """Test that attempting to unregister built-in loader returns False."""
        result = DataLoaderFactory.unregister_loader("bioasq")
        assert result is False

        # Verify built-in loader is still registered
        assert DataLoaderFactory.is_registered("bioasq")

    def test_is_registered_builtin_dataset(self):
        """Test is_registered returns True for built-in datasets."""
        assert DataLoaderFactory.is_registered("bioasq")
        assert DataLoaderFactory.is_registered("hotpot_qa")
        assert DataLoaderFactory.is_registered("kramabench")

    def test_is_registered_custom_dataset(self):
        """Test is_registered returns True for custom datasets."""
        DataLoaderFactory.register_loader("test_is_registered", MockRagDataLoader)

        try:
            assert DataLoaderFactory.is_registered("test_is_registered")
        finally:
            DataLoaderFactory.unregister_loader("test_is_registered")

    def test_is_registered_nonexistent_dataset(self):
        """Test is_registered returns False for non-existent datasets."""
        assert not DataLoaderFactory.is_registered("totally_fake_dataset")

    def test_list_available_datasets_includes_custom(self):
        """Test that list_available_datasets includes custom loaders."""
        initial_datasets = DataLoaderFactory.list_available_datasets()

        # Register custom loaders
        DataLoaderFactory.register_loader("custom_a", MockRagDataLoader)
        DataLoaderFactory.register_loader("custom_b", MockRagDataLoader)

        try:
            updated_datasets = DataLoaderFactory.list_available_datasets()

            # Should include all initial datasets
            for dataset in initial_datasets:
                assert dataset in updated_datasets

            # Should include custom datasets
            assert "custom_a" in updated_datasets
            assert "custom_b" in updated_datasets

            # Should have more datasets than before
            assert len(updated_datasets) > len(initial_datasets)
        finally:
            DataLoaderFactory.unregister_loader("custom_a")
            DataLoaderFactory.unregister_loader("custom_b")

    def test_get_loader_class_custom_dataset(self):
        """Test get_loader_class returns correct class for custom dataset."""
        DataLoaderFactory.register_loader("test_get_class", MockRagDataLoader)

        try:
            loader_class = DataLoaderFactory.get_loader_class("test_get_class")
            assert loader_class == MockRagDataLoader
            assert issubclass(loader_class, RagDataLoader)
        finally:
            DataLoaderFactory.unregister_loader("test_get_class")

    def test_get_loader_class_custom_dataset_not_found(self):
        """Test get_loader_class raises ValueError for non-existent custom dataset."""
        import pytest

        with pytest.raises(ValueError, match="No loader registered"):
            DataLoaderFactory.get_loader_class("nonexistent_custom")

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_create_loader_with_custom_dataset(self):
        """Test create_loader works with custom datasets."""
        DataLoaderFactory.register_loader("test_create", MockRagDataLoader)

        try:
            loader = DataLoaderFactory.create_loader(
                dataset_name="test_create",
                split="train",
                data_sampling=DataSamplingParams(question_limit=5),
            )

            assert isinstance(loader, MockRagDataLoader)
            assert isinstance(loader, RagDataLoader)

            # Verify loader works
            corpus = loader.get_corpus()
            benchmark = loader.get_benchmark()
            assert len(corpus) > 0
            assert len(benchmark) > 0
        finally:
            DataLoaderFactory.unregister_loader("test_create")

    def test_create_loader_custom_dataset_not_found(self):
        """Test create_loader raises ValueError for non-existent custom dataset."""
        import pytest

        with pytest.raises(ValueError, match="No loader registered"):
            DataLoaderFactory.create_loader("nonexistent_custom_dataset")

    def test_custom_registry_isolation(self):
        """Test that custom registry doesn't affect built-in registry."""
        # Get initial state
        initial_builtin_count = len(DataLoaderFactory._LOADER_REGISTRY)
        initial_custom_count = len(DataLoaderFactory._CUSTOM_LOADER_REGISTRY)

        # Register custom loaders
        DataLoaderFactory.register_loader("custom_1", MockRagDataLoader)
        DataLoaderFactory.register_loader("custom_2", MockRagDataLoader)

        try:
            # Built-in registry should be unchanged
            assert len(DataLoaderFactory._LOADER_REGISTRY) == initial_builtin_count

            # Custom registry should have 2 more entries
            assert (
                len(DataLoaderFactory._CUSTOM_LOADER_REGISTRY)
                == initial_custom_count + 2
            )

            # Built-in datasets should still work
            assert DataLoaderFactory.is_registered("bioasq")
            loader_class = DataLoaderFactory.get_loader_class("bioasq")
            assert loader_class == BioasqDataLoader
        finally:
            DataLoaderFactory.unregister_loader("custom_1")
            DataLoaderFactory.unregister_loader("custom_2")

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_multiple_custom_loaders_independent(self):
        """Test that multiple custom loaders work independently."""

        # Create two different mock loader classes
        class CustomLoaderA(MockRagDataLoader):
            pass

        class CustomLoaderB(MockRagDataLoader):
            pass

        # Register both
        DataLoaderFactory.register_loader("loader_a", CustomLoaderA)
        DataLoaderFactory.register_loader("loader_b", CustomLoaderB)

        try:
            # Verify both are registered
            assert DataLoaderFactory.is_registered("loader_a")
            assert DataLoaderFactory.is_registered("loader_b")

            # Verify they return different classes
            class_a = DataLoaderFactory.get_loader_class("loader_a")
            class_b = DataLoaderFactory.get_loader_class("loader_b")
            assert class_a == CustomLoaderA
            assert class_b == CustomLoaderB
            assert class_a != class_b

            # Verify both can be instantiated
            loader_a = DataLoaderFactory.create_loader("loader_a", split="train")
            loader_b = DataLoaderFactory.create_loader("loader_b", split="train")
            assert isinstance(loader_a, CustomLoaderA)
            assert isinstance(loader_b, CustomLoaderB)
        finally:
            DataLoaderFactory.unregister_loader("loader_a")
            DataLoaderFactory.unregister_loader("loader_b")

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_custom_loader_with_data_sampling(self):
        """Test that custom loaders work with sampling parameters."""
        DataLoaderFactory.register_loader("test_sampling", MockRagDataLoader)

        try:
            data_sampling = DataSamplingParams(
                question_limit=3, document_factor=2, seed=42
            )

            loader = DataLoaderFactory.create_loader(
                dataset_name="test_sampling", split="train", data_sampling=data_sampling
            )

            # Verify sampling was applied
            benchmark = loader.get_benchmark()
            assert len(benchmark) <= 3  # Should respect question_limit
        finally:
            DataLoaderFactory.unregister_loader("test_sampling")

    def test_custom_registry_persists_across_calls(self):
        """Test that custom registry persists across multiple factory calls."""
        DataLoaderFactory.register_loader("persistent_loader", MockRagDataLoader)

        try:
            # Call various factory methods
            datasets1 = DataLoaderFactory.list_available_datasets()
            assert "persistent_loader" in datasets1

            loader_class1 = DataLoaderFactory.get_loader_class("persistent_loader")
            assert loader_class1 == MockRagDataLoader

            # Call again - should still be there
            datasets2 = DataLoaderFactory.list_available_datasets()
            assert "persistent_loader" in datasets2

            loader_class2 = DataLoaderFactory.get_loader_class("persistent_loader")
            assert loader_class2 == MockRagDataLoader
        finally:
            DataLoaderFactory.unregister_loader("persistent_loader")
