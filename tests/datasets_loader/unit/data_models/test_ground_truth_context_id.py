"""
Tests for GroundTruthContextId data model.

This module tests the GroundTruthContextId class focusing on business logic
and custom validation beyond basic Pydantic functionality.
"""

import pytest
from pydantic import ValidationError

from ragbench.datasets_loader.data_models.rag_benchmark import GroundTruthContextId


class TestGroundTruthContextId:
    """Test suite for GroundTruthContextId model."""

    def test_creation_with_all_fields(self):
        """Test creating a GroundTruthContextId with all optional fields."""
        context_id = GroundTruthContextId(
            document_id="doc_123", page=5, table_id="table_1"
        )

        assert context_id.document_id == "doc_123"
        assert context_id.page == 5
        assert context_id.table_id == "table_1"

    def test_creation_with_minimal_fields(self):
        """Test creating a GroundTruthContextId with only required fields."""
        context_id = GroundTruthContextId(document_id="doc_456")

        assert context_id.document_id == "doc_456"
        assert context_id.page is None
        assert context_id.table_id is None

    def test_page_validation(self):
        """Test page number validation (must be >= 1)."""
        # Valid page numbers
        GroundTruthContextId(document_id="doc_1", page=1)
        GroundTruthContextId(document_id="doc_2", page=100)

        # Invalid page numbers
        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="doc_3", page=0)

        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="doc_4", page=-1)

    def test_document_id_validation(self):
        """Test document_id validation (required and non-empty)."""
        with pytest.raises(ValidationError):
            GroundTruthContextId()  # type: ignore

        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="")
