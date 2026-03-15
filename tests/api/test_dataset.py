"""Tests for Dataset model and DatasetSplit conversion."""

import pytest

from ragworkbench import DatasetName
from ragworkbench.api.dataset import Dataset, DatasetSplit
from ragworkbench.datasets_loader.data_models import DataSamplingParams


class TestDatasetSplitConversion:
    """Test string to DatasetSplit enum conversion."""

    def test_dataset_split_from_lowercase_string(self):
        """Test that lowercase string 'train' is converted to DatasetSplit.TRAIN."""
        dataset = Dataset(name=DatasetName.BIOASQ, split="train")
        assert dataset.split == DatasetSplit.TRAIN
        assert isinstance(dataset.split, DatasetSplit)

    def test_dataset_split_from_uppercase_string(self):
        """Test that uppercase string 'TRAIN' is converted to DatasetSplit.TRAIN."""
        dataset = Dataset(name=DatasetName.BIOASQ, split="TRAIN")
        assert dataset.split == DatasetSplit.TRAIN
        assert isinstance(dataset.split, DatasetSplit)

    def test_dataset_split_from_mixed_case_string(self):
        """Test that mixed case string 'TrAiN' is converted to DatasetSplit.TRAIN."""
        dataset = Dataset(name=DatasetName.BIOASQ, split="TrAiN")
        assert dataset.split == DatasetSplit.TRAIN
        assert isinstance(dataset.split, DatasetSplit)

    def test_dataset_split_test_from_lowercase_string(self):
        """Test that lowercase string 'test' is converted to DatasetSplit.TEST."""
        dataset = Dataset(name=DatasetName.BIOASQ, split="test")
        assert dataset.split == DatasetSplit.TEST
        assert isinstance(dataset.split, DatasetSplit)

    def test_dataset_split_test_from_uppercase_string(self):
        """Test that uppercase string 'TEST' is converted to DatasetSplit.TEST."""
        dataset = Dataset(name=DatasetName.BIOASQ, split="TEST")
        assert dataset.split == DatasetSplit.TEST
        assert isinstance(dataset.split, DatasetSplit)

    def test_dataset_split_from_enum(self):
        """Test that DatasetSplit enum is preserved."""
        dataset = Dataset(name=DatasetName.BIOASQ, split=DatasetSplit.TRAIN)
        assert dataset.split == DatasetSplit.TRAIN
        assert isinstance(dataset.split, DatasetSplit)

    def test_dataset_split_none(self):
        """Test that None is preserved."""
        dataset = Dataset(name=DatasetName.BIOASQ, split=None)
        assert dataset.split is None

    def test_dataset_split_default_none(self):
        """Test that split defaults to None when not provided."""
        dataset = Dataset(name=DatasetName.BIOASQ)
        assert dataset.split is None

    def test_dataset_split_invalid_string(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError, match="'invalid' is not a valid DatasetSplit"):
            Dataset(name=DatasetName.BIOASQ, split="invalid")


class TestDatasetModel:
    """Test Dataset model functionality."""

    def test_dataset_id_with_string_split(self):
        """Test that dataset ID is correctly generated with string split."""
        dataset = Dataset(name=DatasetName.BIOASQ, split="train")
        dataset_id = dataset.id()
        assert "name-bioasq" in dataset_id
        assert "split-train" in dataset_id

    def test_dataset_id_without_split(self):
        """Test that dataset ID is correctly generated without split."""
        dataset = Dataset(name=DatasetName.BIOASQ)
        dataset_id = dataset.id()
        assert "name-bioasq" in dataset_id
        assert "split" not in dataset_id

    def test_dataset_with_sampling_params(self):
        """Test that dataset works with sampling parameters."""
        sampling = DataSamplingParams(question_limit=10, seed=42)
        dataset = Dataset(name=DatasetName.BIOASQ, split="test", sampling=sampling)
        assert dataset.split == DatasetSplit.TEST
        assert dataset.sampling.question_limit == 10
        assert dataset.sampling.seed == 42

    def test_dataset_id_with_string_split_and_sampling(self):
        """Test that dataset ID includes split and sampling info."""
        sampling = DataSamplingParams(question_limit=10, seed=42)
        dataset = Dataset(name=DatasetName.BIOASQ, split="train", sampling=sampling)
        dataset_id = dataset.id()
        assert "name-bioasq" in dataset_id
        assert "split-train" in dataset_id
        # Check that sampling info is included
        assert "q-10" in dataset_id
        assert "seed-42" in dataset_id


# Made with Bob
