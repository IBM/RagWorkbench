"""
Tests for GroundTruthContextId data model.

This module tests the GroundTruthContextId class including validation,
immutability, and field requirements.
"""

import pytest
from pydantic import ValidationError

from ragbench.datasets_loader.data_models.rag_benchmark import GroundTruthContextId


class TestGroundTruthContextId:
    """Test suite for GroundTruthContextId model."""

    def test_valid_creation_with_all_fields(self):
        """Test creating a GroundTruthContextId with all fields provided."""
        context_id = GroundTruthContextId(
            document_id="doc_123", page=5, table_id="table_1"
        )

        assert context_id.document_id == "doc_123"
        assert context_id.page == 5
        assert context_id.table_id == "table_1"

    def test_valid_creation_minimal_fields(self):
        """Test creating a GroundTruthContextId with only required fields."""
        context_id = GroundTruthContextId(document_id="doc_456")

        assert context_id.document_id == "doc_456"
        assert context_id.page is None
        assert context_id.table_id is None

    def test_immutability_frozen_fields(self):
        """Test that frozen fields cannot be modified after creation."""
        context_id = GroundTruthContextId(document_id="doc_789", page=3)

        with pytest.raises(ValidationError):
            context_id.document_id = "new_doc"

        with pytest.raises(ValidationError):
            context_id.page = 10

    def test_page_validation_positive(self):
        """Test that page number must be >= 1 when provided."""
        # Valid page numbers
        context_id = GroundTruthContextId(document_id="doc_1", page=1)
        assert context_id.page == 1

        context_id = GroundTruthContextId(document_id="doc_2", page=100)
        assert context_id.page == 100

        # Invalid page numbers
        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="doc_3", page=0)

        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="doc_4", page=-1)

    def test_document_id_required_and_non_empty(self):
        """Test that document_id is required and cannot be empty."""
        # Missing document_id
        with pytest.raises(ValidationError):
            GroundTruthContextId()  # type: ignore

        # Empty document_id
        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="")
