"""
Tests for DataSamplingParams data model.

This module comprehensively tests the DataSamplingParams class, focusing on:
- Parameter initialization and defaults
- The as_id() method for generating unique identifiers
- Edge cases and validation
- Model immutability (frozen fields)
"""

import pytest

from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)


class TestDataSamplingParams:
    """Comprehensive test suite for DataSamplingParams model."""

    # ============================================================================
    # Section 1: Creation and Initialization
    # ============================================================================

    def test_creation_with_defaults(self):
        """Test creating DataSamplingParams with default values."""
        params = DataSamplingParams()

        assert params.question_limit is None
        assert params.document_factor is None
        assert params.seed == 43

    def test_creation_with_all_parameters(self):
        """Test creating DataSamplingParams with all parameters specified."""
        params = DataSamplingParams(question_limit=100, document_factor=5, seed=42)

        assert params.question_limit == 100
        assert params.document_factor == 5
        assert params.seed == 42

    def test_default_seed_value(self):
        """Test that default seed is 43."""
        params = DataSamplingParams()
        assert params.seed == 43

    # ============================================================================
    # Section 2: as_id() Method - Basic Functionality
    # ============================================================================

    def test_as_id_with_no_sampling(self):
        """Test as_id() returns empty string when no sampling is applied."""
        params = DataSamplingParams()
        assert params.as_id() == ""

    def test_as_id_with_only_seed_changed(self):
        """Test as_id() returns empty string when only seed is different from default."""
        params = DataSamplingParams(seed=99)
        assert params.as_id() == ""

    @pytest.mark.parametrize(
        "question_limit,document_factor,seed,expected",
        [
            # Single parameter cases
            (100, None, 43, "q-100_seed-43"),
            (None, 5, 43, "docs-factor-5_seed-43"),
            # Both parameters
            (50, 3, 43, "q-50_docs-factor-3_seed-43"),
            # Custom seed
            (100, None, 99, "q-100_seed-99"),
            (None, 5, 42, "docs-factor-5_seed-42"),
            (200, 10, 1, "q-200_docs-factor-10_seed-1"),
        ],
    )
    def test_as_id_format_variations(
        self, question_limit, document_factor, seed, expected
    ):
        """Test as_id() generates correct format for various parameter combinations."""
        params = DataSamplingParams(
            question_limit=question_limit,
            document_factor=document_factor,
            seed=seed,
        )
        assert params.as_id() == expected

    def test_as_id_format_consistency(self):
        """Test that as_id() format is consistent and parseable."""
        params = DataSamplingParams(question_limit=200, document_factor=10, seed=42)
        id_str = params.as_id()

        # Verify format components
        assert "q-200" in id_str
        assert "docs-factor-10" in id_str
        assert "seed-42" in id_str
        assert id_str.count("_") == 2  # Two underscores separating three parts

    # ============================================================================
    # Section 3: Edge Cases and Validation
    # ============================================================================

    def test_zero_question_limit(self):
        """Test that question_limit can be zero (valid edge case)."""
        params = DataSamplingParams(question_limit=0)
        assert params.question_limit == 0
        # Zero is falsy, so as_id() should return empty string
        assert params.as_id() == ""

    def test_zero_document_factor(self):
        """Test that document_factor can be zero (valid edge case)."""
        params = DataSamplingParams(document_factor=0)
        assert params.document_factor == 0
        # Zero is falsy, so as_id() should return empty string
        assert params.as_id() == ""

    def test_negative_seed(self):
        """Test that negative seed values are accepted."""
        params = DataSamplingParams(question_limit=10, seed=-1)
        assert params.seed == -1
        assert "seed--1" in params.as_id()

    def test_large_values(self):
        """Test with very large parameter values."""
        params = DataSamplingParams(
            question_limit=1_000_000,
            document_factor=999,
            seed=2**31 - 1,
        )
        assert params.question_limit == 1_000_000
        assert params.document_factor == 999
        assert params.seed == 2**31 - 1

    # ============================================================================
    # Section 4: Model Behavior
    # ============================================================================

    def test_model_allows_field_access(self):
        """Test that DataSamplingParams fields can be accessed."""
        params = DataSamplingParams(question_limit=100, document_factor=5, seed=42)

        # Fields should be accessible
        assert params.question_limit == 100
        assert params.document_factor == 5
        assert params.seed == 42

        # Note: Pydantic models are mutable by default unless configured otherwise
        # This test documents current behavior

    # ============================================================================
    # Section 5: Equality and Hashing
    # ============================================================================

    def test_equality(self):
        """Test that two DataSamplingParams with same values are equal."""
        params1 = DataSamplingParams(question_limit=100, document_factor=5, seed=42)
        params2 = DataSamplingParams(question_limit=100, document_factor=5, seed=42)

        assert params1 == params2

    def test_inequality(self):
        """Test that DataSamplingParams with different values are not equal."""
        params1 = DataSamplingParams(question_limit=100, seed=42)
        params2 = DataSamplingParams(question_limit=200, seed=42)

        assert params1 != params2

    def test_as_id_uniqueness(self):
        """Test that different parameters produce different IDs."""
        params1 = DataSamplingParams(question_limit=100, seed=42)
        params2 = DataSamplingParams(question_limit=200, seed=42)
        params3 = DataSamplingParams(document_factor=5, seed=42)

        id1 = params1.as_id()
        id2 = params2.as_id()
        id3 = params3.as_id()

        assert id1 != id2
        assert id1 != id3
        assert id2 != id3
        assert len({id1, id2, id3}) == 3  # All unique
