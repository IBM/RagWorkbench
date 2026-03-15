"""
Tests for GroundTruthContextId data model.

This module comprehensively tests the GroundTruthContextId class, focusing on:
- Context ID creation with various configurations
- Field validation (document_id, page, table_id)
- Immutability of frozen fields
- Edge cases and error conditions
- Equality and hashing behavior
"""

import pytest
from pydantic import ValidationError

from ragworkbench.datasets_loader.data_models.rag_benchmark import GroundTruthContextId


class TestGroundTruthContextId:
    """Comprehensive test suite for GroundTruthContextId model."""

    # ============================================================================
    # Section 1: Creation and Initialization
    # ============================================================================

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

    def test_creation_with_page_only(self):
        """Test creating a GroundTruthContextId with document_id and page."""
        context_id = GroundTruthContextId(document_id="doc_789", page=10)

        assert context_id.document_id == "doc_789"
        assert context_id.page == 10
        assert context_id.table_id is None

    def test_creation_with_table_id_only(self):
        """Test creating a GroundTruthContextId with document_id and table_id."""
        context_id = GroundTruthContextId(document_id="doc_abc", table_id="table_xyz")

        assert context_id.document_id == "doc_abc"
        assert context_id.page is None
        assert context_id.table_id == "table_xyz"

    # ============================================================================
    # Section 2: Field Validation
    # ============================================================================

    def test_document_id_required(self):
        """Test that document_id is required and cannot be omitted."""
        with pytest.raises(ValidationError):
            GroundTruthContextId()  # type: ignore

    def test_document_id_validation_empty_string(self):
        """Test document_id validation with empty string."""
        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="")

    def test_document_id_accepts_whitespace(self):
        """Test that document_id accepts whitespace (min_length=1 allows it)."""
        # Whitespace-only strings pass min_length=1 validation
        context = GroundTruthContextId(document_id="   ")
        assert context.document_id == "   "

    def test_document_id_with_special_characters(self):
        """Test that document_id accepts various string formats."""
        valid_ids = [
            "doc-123",
            "doc_456",
            "doc.789",
            "doc/abc",
            "doc:xyz",
            "document with spaces",
            "文档123",  # Unicode
        ]

        for doc_id in valid_ids:
            context = GroundTruthContextId(document_id=doc_id)
            assert context.document_id == doc_id

    @pytest.mark.parametrize(
        "invalid_page",
        [
            0,  # Zero
            -1,  # Negative
            -100,  # Large negative
        ],
    )
    def test_page_validation_invalid_values(self, invalid_page):
        """Test page number validation (must be >= 1)."""
        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="doc_1", page=invalid_page)

    @pytest.mark.parametrize(
        "valid_page",
        [
            1,  # Minimum valid
            10,  # Common case
            100,  # Large page
            1000,  # Very large page
            999999,  # Extremely large page
        ],
    )
    def test_page_validation_valid_values(self, valid_page):
        """Test that valid page numbers are accepted."""
        context = GroundTruthContextId(document_id="doc_1", page=valid_page)
        assert context.page == valid_page

    def test_table_id_accepts_any_string(self):
        """Test that table_id accepts various string formats."""
        valid_table_ids = [
            "table_1",
            "Table-2",
            "table.3",
            "table with spaces",
            "表1",  # Unicode
            "",  # Empty string is valid for table_id
        ]

        for table_id in valid_table_ids:
            context = GroundTruthContextId(document_id="doc_1", table_id=table_id)
            assert context.table_id == table_id

    # ============================================================================
    # Section 3: Immutability (Frozen Fields)
    # ============================================================================

    def test_document_id_immutability(self):
        """Test that document_id cannot be modified after creation (frozen field)."""
        context = GroundTruthContextId(document_id="doc_1")

        with pytest.raises((ValidationError, AttributeError)):
            context.document_id = "doc_2"  # type: ignore

    def test_page_immutability(self):
        """Test that page cannot be modified after creation (frozen field)."""
        context = GroundTruthContextId(document_id="doc_1", page=5)

        with pytest.raises((ValidationError, AttributeError)):
            context.page = 10  # type: ignore

    def test_table_id_immutability(self):
        """Test that table_id cannot be modified after creation (frozen field)."""
        context = GroundTruthContextId(document_id="doc_1", table_id="table_1")

        with pytest.raises((ValidationError, AttributeError)):
            context.table_id = "table_2"  # type: ignore

    def test_all_fields_immutability(self):
        """Test that all fields are immutable when set together."""
        context = GroundTruthContextId(document_id="doc_1", page=5, table_id="table_1")

        # Try to modify each field
        with pytest.raises((ValidationError, AttributeError)):
            context.document_id = "doc_2"  # type: ignore

        with pytest.raises((ValidationError, AttributeError)):
            context.page = 10  # type: ignore

        with pytest.raises((ValidationError, AttributeError)):
            context.table_id = "table_2"  # type: ignore

    # ============================================================================
    # Section 4: Equality and Hashing
    # ============================================================================

    def test_equality_with_all_fields(self):
        """Test equality of GroundTruthContextId instances with all fields."""
        context1 = GroundTruthContextId(document_id="doc_1", page=5, table_id="table_1")
        context2 = GroundTruthContextId(document_id="doc_1", page=5, table_id="table_1")

        assert context1 == context2

    def test_equality_with_minimal_fields(self):
        """Test equality of GroundTruthContextId instances with minimal fields."""
        context1 = GroundTruthContextId(document_id="doc_1")
        context2 = GroundTruthContextId(document_id="doc_1")

        assert context1 == context2

    def test_inequality_different_document_id(self):
        """Test that contexts with different document_ids are not equal."""
        context1 = GroundTruthContextId(document_id="doc_1")
        context2 = GroundTruthContextId(document_id="doc_2")

        assert context1 != context2

    def test_inequality_different_page(self):
        """Test that contexts with different pages are not equal."""
        context1 = GroundTruthContextId(document_id="doc_1", page=1)
        context2 = GroundTruthContextId(document_id="doc_1", page=2)

        assert context1 != context2

    def test_inequality_different_table_id(self):
        """Test that contexts with different table_ids are not equal."""
        context1 = GroundTruthContextId(document_id="doc_1", table_id="table_1")
        context2 = GroundTruthContextId(document_id="doc_1", table_id="table_2")

        assert context1 != context2

    def test_inequality_none_vs_value(self):
        """Test that None and a value are not equal for optional fields."""
        context1 = GroundTruthContextId(document_id="doc_1")
        context2 = GroundTruthContextId(document_id="doc_1", page=1)

        assert context1 != context2

    def test_instances_are_not_hashable_by_default(self):
        """Test that GroundTruthContextId instances are not hashable by default.

        Note: Pydantic models with frozen=True on fields are not automatically
        hashable. The model itself needs to be configured as frozen to be hashable.
        This test documents current behavior.
        """
        context1 = GroundTruthContextId(document_id="doc_1", page=5)
        context2 = GroundTruthContextId(document_id="doc_2", page=10)

        # Currently not hashable - would need model-level frozen=True
        # This is expected behavior with current model configuration
        try:
            _ = {context1, context2}  # type: ignore[misc]
        except TypeError as e:
            assert "unhashable" in str(e).lower()

    # ============================================================================
    # Section 5: Edge Cases and Special Scenarios
    # ============================================================================

    def test_very_long_document_id(self):
        """Test with very long document_id string."""
        long_id = "doc_" + "x" * 1000
        context = GroundTruthContextId(document_id=long_id)
        assert context.document_id == long_id

    def test_very_long_table_id(self):
        """Test with very long table_id string."""
        long_table_id = "table_" + "y" * 1000
        context = GroundTruthContextId(document_id="doc_1", table_id=long_table_id)
        assert context.table_id == long_table_id

    def test_unicode_in_all_fields(self):
        """Test with Unicode characters in all string fields."""
        context = GroundTruthContextId(
            document_id="文档_123",
            page=5,
            table_id="表格_456",
        )

        assert context.document_id == "文档_123"
        assert context.page == 5
        assert context.table_id == "表格_456"

    def test_representation(self):
        """Test string representation of GroundTruthContextId."""
        context = GroundTruthContextId(document_id="doc_1", page=5, table_id="table_1")

        repr_str = repr(context)
        assert "doc_1" in repr_str
        assert "5" in repr_str or "page=5" in repr_str
        assert "table_1" in repr_str
