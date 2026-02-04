"""
Tests for DataSamplingParams data model.

This module tests the DataSamplingParams class focusing on the as_id() method
which generates unique identifiers based on sampling parameters.
"""

from ragbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)


class TestDataSamplingParams:
    """Test suite for DataSamplingParams model."""

    def test_as_id_with_no_sampling(self):
        """Test as_id() returns empty string when no sampling is applied."""
        params = DataSamplingParams()
        assert params.as_id() == ""

    def test_as_id_with_question_limit_only(self):
        """Test as_id() with only question_limit set."""
        params = DataSamplingParams(question_limit=100)
        assert params.as_id() == "q-100_seed-43"

    def test_as_id_with_document_factor_only(self):
        """Test as_id() with only document_factor set."""
        params = DataSamplingParams(document_factor=5)
        assert params.as_id() == "docs-factor-5_seed-43"

    def test_as_id_with_both_limits(self):
        """Test as_id() with both question_limit and document_factor set."""
        params = DataSamplingParams(question_limit=50, document_factor=3)
        assert params.as_id() == "q-50_docs-factor-3_seed-43"

    def test_as_id_with_custom_seed(self):
        """Test as_id() includes custom seed when sampling is applied."""
        params = DataSamplingParams(question_limit=100, seed=99)
        assert params.as_id() == "q-100_seed-99"

    def test_default_seed_value(self):
        """Test that default seed is 43."""
        params = DataSamplingParams()
        assert params.seed == 43

    def test_as_id_format_consistency(self):
        """Test that as_id() format is consistent and parseable."""
        params = DataSamplingParams(question_limit=200, document_factor=10, seed=42)
        id_str = params.as_id()

        # Verify format
        assert "q-200" in id_str
        assert "docs-factor-10" in id_str
        assert "seed-42" in id_str
        assert id_str.count("_") == 2  # Two underscores separating three parts
