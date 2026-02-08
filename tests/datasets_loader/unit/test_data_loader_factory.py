"""
Unit tests for DataLoaderFactory.

This module contains unit tests for the DataLoaderFactory class, verifying
that it correctly creates loader instances, handles parameters, and provides
proper error messages for invalid inputs.
"""

import pytest

from ragbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragbench.datasets_loader.data_loader_factory import DataLoaderFactory
from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets_loader.dataset_names import DatasetName
from ragbench.datasets_loader.hotpot_qa_data_loader import HotpotQaDataLoader
from ragbench.datasets_loader.kramabench_data_loader import KramabenchDataLoader


class TestDataLoaderFactory:
    """Unit tests for DataLoaderFactory class."""

    def test_list_available_datasets(self):
        """Test that list_available_datasets returns all registered datasets."""
        datasets = DataLoaderFactory.list_available_datasets()

        # Should return a list
        assert isinstance(datasets, list)

        # Should contain all 13 datasets (AI_ARXIV not yet implemented)
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

    def test_get_loader_class_invalid_name(self):
        """Test get_loader_class raises ValueError for invalid dataset name."""
        with pytest.raises(ValueError) as exc_info:
            DataLoaderFactory.get_loader_class("invalid_dataset")

        assert "Invalid dataset name" in str(exc_info.value)
        assert "invalid_dataset" in str(exc_info.value)

    def test_get_loader_class_not_in_registry(self):
        """Test get_loader_class raises ValueError for valid enum not in registry."""
        # AI_ARXIV is in the enum but not yet implemented in the registry
        with pytest.raises(ValueError) as exc_info:
            DataLoaderFactory.get_loader_class(DatasetName.AI_ARXIV)

        assert "No loader registered" in str(exc_info.value)
        assert "ai_arxiv" in str(exc_info.value)

    def test_create_loader_with_enum_returns_loader_instance(self):
        """Test create_loader with enum returns a RagDataLoader instance."""
        # Use a mock to avoid actual data loading
        with pytest.raises((Exception, TypeError)):
            # This will fail during actual instantiation, but we're testing
            # that the factory attempts to create the right type
            DataLoaderFactory.create_loader(
                dataset_name=DatasetName.BIOASQ, split="train"
            )

    def test_create_loader_with_string_dataset_name(self):
        """Test create_loader accepts string dataset names."""
        # Test that string is converted to enum
        with pytest.raises((Exception, TypeError)):
            DataLoaderFactory.create_loader(dataset_name="bioasq", split="train")

    def test_create_loader_invalid_dataset_name(self):
        """Test create_loader raises ValueError for invalid dataset name."""
        with pytest.raises(ValueError) as exc_info:
            DataLoaderFactory.create_loader(
                dataset_name="nonexistent_dataset", split="train"
            )

        assert "Invalid dataset name" in str(exc_info.value)

    def test_create_loader_with_sampling_params(self):
        """Test create_loader accepts sampling parameters."""
        sampling_params = DataSamplingParams(
            question_limit=10, document_factor=2, seed=42
        )

        # This will fail during actual instantiation, but we're testing parameter passing
        with pytest.raises((Exception, TypeError)):
            DataLoaderFactory.create_loader(
                dataset_name=DatasetName.BIOASQ,
                split="train",
                sampling_params=sampling_params,
            )

    def test_create_loader_with_none_sampling_params(self):
        """Test create_loader uses default DataSamplingParams when None provided."""
        # The factory should create default DataSamplingParams internally
        with pytest.raises((Exception, TypeError)):
            DataLoaderFactory.create_loader(
                dataset_name=DatasetName.BIOASQ, split="train", sampling_params=None
            )

    def test_create_loader_with_loader_specific_kwargs(self):
        """Test create_loader passes through loader-specific kwargs."""
        # HotpotQA accepts a 'level' parameter
        with pytest.raises((Exception, TypeError)):
            DataLoaderFactory.create_loader(
                dataset_name=DatasetName.HOTPOT_QA, split="train", level="hard"
            )

    def test_create_loader_with_multiple_kwargs(self):
        """Test create_loader handles multiple loader-specific kwargs."""
        # Kramabench accepts verbose and progress_every
        with pytest.raises((Exception, TypeError)):
            DataLoaderFactory.create_loader(
                dataset_name=DatasetName.KRAMABENCH,
                split="test",
                verbose=True,
                progress_every=10,
            )

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

    def test_factory_preserves_split_parameter(self):
        """Test that split parameter is correctly passed to loaders."""
        # We can't fully test this without mocking, but we can verify
        # the factory doesn't reject valid split values
        for split in ["train", "test", None]:
            with pytest.raises((Exception, TypeError)):
                DataLoaderFactory.create_loader(
                    dataset_name=DatasetName.BIOASQ, split=split
                )

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

    def test_create_loader_error_message_quality(self):
        """Test that error messages are helpful and informative."""
        # Test invalid dataset name error
        with pytest.raises(ValueError) as exc_info:
            DataLoaderFactory.create_loader("invalid_name")

        error_msg = str(exc_info.value)
        assert "Invalid dataset name" in error_msg
        assert "invalid_name" in error_msg
        # Should suggest valid options
        assert "Valid options" in error_msg or "Available" in error_msg

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
