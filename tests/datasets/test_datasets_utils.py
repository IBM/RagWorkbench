"""
Comprehensive tests for datasets utility functions.

This module tests the get_benchmark_split function, ensuring proper train/test
splitting behavior including no overlap, complete coverage, reproducibility,
and correct ratio handling.
"""

import pytest

from ragbench.datasets.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)
from ragbench.datasets.datasets_utils import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    get_benchmark_split,
)


class TestGetBenchmarkSplit:
    """Test suite for get_benchmark_split function."""

    def test_train_test_no_overlap(self, sample_benchmark_entries):
        """
        Test that train and test splits have no overlapping entries.

        This is a critical requirement: benchmarks in train and test must be
        completely different to prevent data leakage.
        """
        train_split = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.7, seed=42
        )
        test_split = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=0.7, seed=42
        )

        # Extract question IDs from both splits
        train_ids = {entry.question_id for entry in train_split}
        test_ids = {entry.question_id for entry in test_split}

        # Verify no overlap
        overlap = train_ids.intersection(test_ids)
        assert len(overlap) == 0, f"Found overlapping question IDs: {overlap}"
        assert train_ids.isdisjoint(test_ids), "Train and test splits must be disjoint"

    def test_train_test_union_equals_whole(self, sample_benchmark_entries):
        """
        Test that union of train and test splits equals the whole dataset.

        This ensures all entries are accounted for and none are lost during splitting.
        """
        train_split = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.7, seed=42
        )
        test_split = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=0.7, seed=42
        )

        # Verify total count matches
        assert len(train_split) + len(test_split) == len(sample_benchmark_entries)

        # Extract question IDs
        train_ids = {entry.question_id for entry in train_split}
        test_ids = {entry.question_id for entry in test_split}
        original_ids = {entry.question_id for entry in sample_benchmark_entries}

        # Verify union equals original
        union_ids = train_ids.union(test_ids)
        assert (
            union_ids == original_ids
        ), "Union of train and test must equal original dataset"

    def test_split_reproducibility(self, sample_benchmark_entries):
        """Test that splits are reproducible with the same seed."""
        # First call
        train_split_1 = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.6, seed=123
        )
        test_split_1 = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=0.6, seed=123
        )

        # Second call with same parameters
        train_split_2 = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.6, seed=123
        )
        test_split_2 = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=0.6, seed=123
        )

        # Verify identical results
        assert len(train_split_1) == len(train_split_2)
        assert len(test_split_1) == len(test_split_2)

        # Check same entries in same order
        train_ids_1 = [e.question_id for e in train_split_1]
        train_ids_2 = [e.question_id for e in train_split_2]
        assert train_ids_1 == train_ids_2

        test_ids_1 = [e.question_id for e in test_split_1]
        test_ids_2 = [e.question_id for e in test_split_2]
        assert test_ids_1 == test_ids_2

    def test_different_seeds_produce_different_splits(self, sample_benchmark_entries):
        """Test that different seeds produce different split orderings."""
        train_split_seed_42 = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.7, seed=42
        )
        train_split_seed_100 = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.7, seed=100
        )

        # Extract question IDs in order
        ids_seed_42 = [e.question_id for e in train_split_seed_42]
        ids_seed_100 = [e.question_id for e in train_split_seed_100]

        # Different seeds should produce different orderings
        # (with high probability for datasets with multiple entries)
        if len(sample_benchmark_entries) > 2:
            assert (
                ids_seed_42 != ids_seed_100
            ), "Different seeds should produce different orderings"

    def test_correct_split_ratios(self, large_benchmark_entries):
        """Test that split ratios are correctly applied."""
        test_cases = [
            (0.3, 0.3),  # 30% train
            (0.5, 0.5),  # 50% train
            (0.7, 0.7),  # 70% train (default)
            (0.8, 0.8),  # 80% train
        ]

        for train_ratio, expected_ratio in test_cases:
            train_split = get_benchmark_split(
                large_benchmark_entries, split="train", train_ratio=train_ratio, seed=42
            )
            test_split = get_benchmark_split(
                large_benchmark_entries, split="test", train_ratio=train_ratio, seed=42
            )

            total = len(large_benchmark_entries)
            actual_train_ratio = len(train_split) / total

            # Allow small rounding tolerance
            assert (
                abs(actual_train_ratio - expected_ratio) < 0.1
            ), f"Expected train ratio ~{expected_ratio}, got {actual_train_ratio}"

            # Verify splits sum to total
            assert len(train_split) + len(test_split) == total

    def test_split_none_returns_all(self, sample_benchmark_entries):
        """Test that split=None returns all entries unchanged."""
        result = get_benchmark_split(
            sample_benchmark_entries, split=None, train_ratio=0.7, seed=42
        )

        assert len(result) == len(sample_benchmark_entries)
        assert result == sample_benchmark_entries

    def test_invalid_train_ratio_raises_error(self, sample_benchmark_entries):
        """Test that invalid train_ratio values raise ValueError."""
        invalid_ratios = [0.0, 1.0, -0.5, 1.5, 2.0]

        for invalid_ratio in invalid_ratios:
            with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
                get_benchmark_split(
                    sample_benchmark_entries,
                    split="train",
                    train_ratio=invalid_ratio,
                    seed=42,
                )

    def test_single_entry_dataset(self):
        """Test splitting behavior with a single entry."""
        single_entry = [
            RagBenchmarkEntry(
                question_id="q_single",
                question="Single question?",
                ground_truth_answers=["Answer"],
                ground_truth_context_ids=[GroundTruthContextId(document_id="doc_1")],
                is_answerable=True,
            )
        ]

        # With train_ratio=0.7, single entry should go to train
        train_split = get_benchmark_split(
            single_entry, split="train", train_ratio=0.7, seed=42
        )
        test_split = get_benchmark_split(
            single_entry, split="test", train_ratio=0.7, seed=42
        )

        # One split should have the entry, the other should be empty
        assert len(train_split) + len(test_split) == 1
        assert len(train_split) == 1 or len(test_split) == 1

    def test_small_dataset_split(self):
        """Test splitting behavior with a small dataset (2-3 entries)."""
        small_entries = [
            RagBenchmarkEntry(
                question_id=f"q_{i}",
                question=f"Question {i}?",
                ground_truth_answers=[f"Answer {i}"],
                ground_truth_context_ids=[GroundTruthContextId(document_id=f"doc_{i}")],
                is_answerable=True,
            )
            for i in range(3)
        ]

        train_split = get_benchmark_split(
            small_entries, split="train", train_ratio=0.6, seed=42
        )
        test_split = get_benchmark_split(
            small_entries, split="test", train_ratio=0.6, seed=42
        )

        # Verify no overlap
        train_ids = {e.question_id for e in train_split}
        test_ids = {e.question_id for e in test_split}
        assert train_ids.isdisjoint(test_ids)

        # Verify union equals whole
        assert len(train_split) + len(test_split) == len(small_entries)

    def test_boundary_ratios(self, sample_benchmark_entries):
        """Test splitting with boundary train ratios (very small and very large)."""
        # Very small train ratio
        train_split_small = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.01, seed=42
        )
        test_split_small = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=0.01, seed=42
        )

        assert len(train_split_small) + len(test_split_small) == len(
            sample_benchmark_entries
        )

        # Very large train ratio
        train_split_large = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.99, seed=42
        )
        test_split_large = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=0.99, seed=42
        )

        assert len(train_split_large) + len(test_split_large) == len(
            sample_benchmark_entries
        )

    def test_default_parameters(self, sample_benchmark_entries):
        """Test that default parameters work correctly."""
        # Test with default train_ratio and seed
        train_split = get_benchmark_split(
            sample_benchmark_entries,
            split="train",
            train_ratio=DEFAULT_TRAIN_RATIO,
            seed=DEFAULT_SPLIT_SEED,
        )
        test_split = get_benchmark_split(
            sample_benchmark_entries,
            split="test",
            train_ratio=DEFAULT_TRAIN_RATIO,
            seed=DEFAULT_SPLIT_SEED,
        )

        # Verify basic properties
        assert len(train_split) > 0
        assert len(test_split) > 0
        assert len(train_split) + len(test_split) == len(sample_benchmark_entries)

    def test_preserves_entry_integrity(self, sample_benchmark_entries):
        """Test that splitting preserves the integrity of benchmark entries."""
        train_split = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.7, seed=42
        )

        # Verify entries are complete RagBenchmarkEntry objects
        for entry in train_split:
            assert isinstance(entry, RagBenchmarkEntry)
            assert entry.question_id is not None
            assert entry.question is not None
            # Verify the entry is one of the original entries
            assert entry in sample_benchmark_entries

    def test_large_dataset_performance(self, large_benchmark_entries):
        """Test splitting performance with a larger dataset."""
        # This test ensures the function works efficiently with larger datasets
        train_split = get_benchmark_split(
            large_benchmark_entries, split="train", train_ratio=0.7, seed=42
        )
        test_split = get_benchmark_split(
            large_benchmark_entries, split="test", train_ratio=0.7, seed=42
        )

        # Verify correctness with large dataset
        train_ids = {e.question_id for e in train_split}
        test_ids = {e.question_id for e in test_split}

        assert train_ids.isdisjoint(test_ids)
        assert len(train_split) + len(test_split) == len(large_benchmark_entries)
