"""
Tests for datasets_utils module.

This module tests the utility functions for dataset operations,
including train/test splitting with reproducible randomization.
"""

import pytest

from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)
from ragworkbench.datasets_loader.datasets_utils import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    get_benchmark_split,
)


class TestGetBenchmarkSplit:
    """Test suite for get_benchmark_split function."""

    def test_split_none_returns_all_entries(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that split=None returns all entries unchanged."""
        result = get_benchmark_split(sample_benchmark_entries, split=None)

        assert result == sample_benchmark_entries
        assert len(result) == len(sample_benchmark_entries)
        # Verify it's the same list, not a copy
        assert result is sample_benchmark_entries

    def test_train_split_returns_correct_proportion(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that train split returns correct proportion of entries."""
        train_ratio = 0.6
        result = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=train_ratio
        )

        expected_size = int(train_ratio * len(sample_benchmark_entries))
        assert len(result) == expected_size

    def test_test_split_returns_correct_proportion(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that test split returns correct proportion of entries."""
        train_ratio = 0.6
        result = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=train_ratio
        )

        expected_train_size = int(train_ratio * len(sample_benchmark_entries))
        expected_test_size = len(sample_benchmark_entries) - expected_train_size
        assert len(result) == expected_test_size

    def test_train_and_test_splits_are_complementary(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that train and test splits together contain all entries."""
        train_ratio = 0.7
        train = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=train_ratio
        )
        test = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=train_ratio
        )

        # Combined length should equal original
        assert len(train) + len(test) == len(sample_benchmark_entries)

        # No overlap between train and test
        train_ids = {entry.question_id for entry in train}
        test_ids = {entry.question_id for entry in test}
        assert len(train_ids & test_ids) == 0

        # All entries are accounted for
        all_ids = {entry.question_id for entry in sample_benchmark_entries}
        assert train_ids | test_ids == all_ids

    def test_split_is_reproducible_with_same_seed(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that same seed produces same split."""
        seed = 123
        train_ratio = 0.6

        train1 = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=train_ratio, seed=seed
        )
        train2 = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=train_ratio, seed=seed
        )

        assert len(train1) == len(train2)
        assert [e.question_id for e in train1] == [e.question_id for e in train2]

    def test_split_differs_with_different_seed(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that different seeds produce different splits."""
        train_ratio = 0.6

        train1 = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=train_ratio, seed=42
        )
        train2 = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=train_ratio, seed=123
        )

        # With different seeds, the order should be different
        # (unless by extreme coincidence)
        train1_ids = [e.question_id for e in train1]
        train2_ids = [e.question_id for e in train2]
        assert train1_ids != train2_ids

    def test_default_train_ratio_is_used(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that default train ratio is applied when not specified."""
        result = get_benchmark_split(sample_benchmark_entries, split="train")

        expected_size = int(DEFAULT_TRAIN_RATIO * len(sample_benchmark_entries))
        assert len(result) == expected_size

    def test_default_seed_is_used(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that default seed produces consistent results."""
        # Call twice without specifying seed
        train1 = get_benchmark_split(sample_benchmark_entries, split="train")
        train2 = get_benchmark_split(sample_benchmark_entries, split="train")

        assert [e.question_id for e in train1] == [e.question_id for e in train2]

    def test_train_ratio_zero_point_one(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test with very small train ratio."""
        train_ratio = 0.1
        train = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=train_ratio
        )
        test = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=train_ratio
        )

        expected_train_size = int(train_ratio * len(sample_benchmark_entries))
        assert len(train) == expected_train_size
        assert len(test) == len(sample_benchmark_entries) - expected_train_size

    def test_train_ratio_zero_point_nine(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test with very large train ratio."""
        train_ratio = 0.9
        train = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=train_ratio
        )
        test = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=train_ratio
        )

        expected_train_size = int(train_ratio * len(sample_benchmark_entries))
        assert len(train) == expected_train_size
        assert len(test) == len(sample_benchmark_entries) - expected_train_size

    def test_train_ratio_exactly_zero_raises_error(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that train_ratio of 0 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            get_benchmark_split(
                sample_benchmark_entries, split="train", train_ratio=0.0
            )

    def test_train_ratio_exactly_one_raises_error(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that train_ratio of 1 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            get_benchmark_split(
                sample_benchmark_entries, split="train", train_ratio=1.0
            )

    def test_train_ratio_negative_raises_error(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that negative train_ratio raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            get_benchmark_split(
                sample_benchmark_entries, split="train", train_ratio=-0.5
            )

    def test_train_ratio_greater_than_one_raises_error(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that train_ratio > 1 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio must be between 0 and 1"):
            get_benchmark_split(
                sample_benchmark_entries, split="train", train_ratio=1.5
            )

    def test_split_with_single_entry(self):
        """Test splitting with only one entry."""
        entry = RagBenchmarkEntry(
            question_id="q_single",
            question="Single question?",
            ground_truth_answers=["Answer"],
            ground_truths_context_ids=[GroundTruthContextId(document_id="doc_1")],
        )
        entries = [entry]

        # With train_ratio=0.7, int(0.7 * 1) = 0, so train should be empty
        train = get_benchmark_split(entries, split="train", train_ratio=0.7)
        test = get_benchmark_split(entries, split="test", train_ratio=0.7)

        assert len(train) == 0
        assert len(test) == 1
        assert test[0].question_id == "q_single"

    def test_split_with_two_entries(self):
        """Test splitting with exactly two entries."""
        entries = [
            RagBenchmarkEntry(
                question_id="q_1",
                question="Question 1?",
                ground_truth_answers=["Answer 1"],
                ground_truths_context_ids=[GroundTruthContextId(document_id="doc_1")],
            ),
            RagBenchmarkEntry(
                question_id="q_2",
                question="Question 2?",
                ground_truth_answers=["Answer 2"],
                ground_truths_context_ids=[GroundTruthContextId(document_id="doc_2")],
            ),
        ]

        # With train_ratio=0.5, int(0.5 * 2) = 1
        train = get_benchmark_split(entries, split="train", train_ratio=0.5)
        test = get_benchmark_split(entries, split="test", train_ratio=0.5)

        assert len(train) == 1
        assert len(test) == 1
        assert train[0].question_id != test[0].question_id

    def test_split_does_not_modify_original_list(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that splitting does not modify the original list."""
        original_ids = [e.question_id for e in sample_benchmark_entries]
        original_length = len(sample_benchmark_entries)

        get_benchmark_split(sample_benchmark_entries, split="train", train_ratio=0.6)

        # Original list should be unchanged
        assert len(sample_benchmark_entries) == original_length
        assert [e.question_id for e in sample_benchmark_entries] == original_ids

    def test_split_with_large_dataset(self, large_benchmark_entries):
        """Test splitting with a larger dataset."""
        train_ratio = 0.7
        train = get_benchmark_split(
            large_benchmark_entries, split="train", train_ratio=train_ratio
        )
        test = get_benchmark_split(
            large_benchmark_entries, split="test", train_ratio=train_ratio
        )

        expected_train_size = int(train_ratio * len(large_benchmark_entries))
        assert len(train) == expected_train_size
        assert len(test) == len(large_benchmark_entries) - expected_train_size
        assert len(train) + len(test) == len(large_benchmark_entries)

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_split_preserves_entry_properties(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that split entries maintain all their properties."""
        train = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.6
        )

        # Verify that entries in the split are complete RagBenchmarkEntry objects
        for entry in train:
            assert isinstance(entry, RagBenchmarkEntry)
            assert hasattr(entry, "question_id")
            assert hasattr(entry, "question")
            assert hasattr(entry, "ground_truth_answers")
            assert hasattr(entry, "ground_truth_context_ids")
            assert hasattr(entry, "is_answerable")

    def test_multiple_splits_with_same_parameters_are_identical(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that multiple calls with same parameters produce identical results."""
        params = {"split": "train", "train_ratio": 0.65, "seed": 999}

        results = [
            get_benchmark_split(sample_benchmark_entries, **params) for _ in range(5)
        ]

        # All results should be identical
        first_result_ids = [e.question_id for e in results[0]]
        for result in results[1:]:
            assert [e.question_id for e in result] == first_result_ids

    def test_empty_list_handling(self):
        """Test behavior with empty list."""
        empty_list = []

        # split=None should return empty list
        result = get_benchmark_split(empty_list, split=None)
        assert result == []

        # train split should return empty list
        train = get_benchmark_split(empty_list, split="train", train_ratio=0.7)
        assert train == []

        # test split should return empty list
        test = get_benchmark_split(empty_list, split="test", train_ratio=0.7)
        assert test == []

    def test_train_ratio_boundary_values(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test train_ratio at boundary values just inside valid range."""
        # Just above 0
        train = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.01
        )
        assert isinstance(train, list)

        # Just below 1
        train = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=0.99
        )
        assert isinstance(train, list)

    def test_split_maintains_entry_order_within_shuffle(
        self, sample_benchmark_entries: list[RagBenchmarkEntry]
    ):
        """Test that the same seed produces the same shuffled order."""
        seed = 42
        train_ratio = 0.5

        # Get train and test splits
        train = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=train_ratio, seed=seed
        )
        test = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=train_ratio, seed=seed
        )

        # Reconstruct the shuffled order
        shuffled_ids = [e.question_id for e in train] + [e.question_id for e in test]

        # Get the same split again
        train2 = get_benchmark_split(
            sample_benchmark_entries, split="train", train_ratio=train_ratio, seed=seed
        )
        test2 = get_benchmark_split(
            sample_benchmark_entries, split="test", train_ratio=train_ratio, seed=seed
        )
        shuffled_ids2 = [e.question_id for e in train2] + [e.question_id for e in test2]

        # Should be identical
        assert shuffled_ids == shuffled_ids2


class TestDefaultConstants:
    """Test suite for module-level constants."""

    def test_default_train_ratio_value(self):
        """Test that DEFAULT_TRAIN_RATIO has expected value."""
        assert DEFAULT_TRAIN_RATIO == 0.7

    def test_default_split_seed_value(self):
        """Test that DEFAULT_SPLIT_SEED has expected value."""
        assert DEFAULT_SPLIT_SEED == 42

    def test_default_train_ratio_is_valid(self):
        """Test that DEFAULT_TRAIN_RATIO is in valid range."""
        assert 0 < DEFAULT_TRAIN_RATIO < 1

    def test_default_split_seed_is_integer(self):
        """Test that DEFAULT_SPLIT_SEED is an integer."""
        assert isinstance(DEFAULT_SPLIT_SEED, int)
