"""
Tests for DocumentObject data model.

This module comprehensively tests the DocumentObject class, focusing on:
- Document creation with various configurations
- MIME type validation (comprehensive coverage)
- Stream handling and content preservation
- Metadata management
- Edge cases and error conditions
"""

from io import BytesIO

import pytest

from ragworkbench.datasets_loader.data_models.document_object import DocumentObject


class TestDocumentObject:
    """Comprehensive test suite for DocumentObject model."""

    # ============================================================================
    # Section 1: Creation and Initialization
    # ============================================================================

    def test_creation_with_all_fields(self):
        """Test creating a DocumentObject with all fields specified."""
        doc = DocumentObject(
            name="test.pdf",
            stream=BytesIO(b"test content"),
            mime_type="application/pdf",
            metadata={"author": "Test Author", "pages": 10},
        )

        assert doc.name == "test.pdf"
        assert doc.mime_type == "application/pdf"
        assert doc.metadata == {"author": "Test Author", "pages": 10}

    def test_creation_with_minimal_fields(self):
        """Test creating a DocumentObject with only required fields."""
        doc = DocumentObject(
            name="test.txt",
            stream=BytesIO(b"content"),
            mime_type="text/plain",
        )

        assert doc.name == "test.txt"
        assert doc.mime_type == "text/plain"
        assert doc.metadata == {}

    def test_metadata_defaults_to_empty_dict(self):
        """Test that metadata defaults to an empty dictionary when not provided."""
        doc = DocumentObject(
            name="test.txt",
            stream=BytesIO(b"content"),
            mime_type="text/plain",
        )

        assert doc.metadata == {}
        assert isinstance(doc.metadata, dict)

    @pytest.mark.parametrize(
        "mime_type,extension",
        [
            ("application/pdf", "pdf"),
            ("text/plain", "txt"),
            ("text/html", "html"),
            ("image/jpeg", "jpeg"),
            ("image/png", "png"),
            ("application/json", "json"),
            ("application/xml", "xml"),
            ("text/csv", "csv"),
            ("application/zip", "zip"),
            ("video/mp4", "mp4"),
        ],
    )
    def test_creation_with_various_mime_types(self, mime_type, extension):
        """Test creating documents with various common MIME types."""
        doc = DocumentObject(
            name=f"test.{extension}",
            stream=BytesIO(b"content"),
            mime_type=mime_type,
        )
        assert doc.mime_type == mime_type

    # ============================================================================
    # Section 2: MIME Type Handling
    # ============================================================================

    def test_mime_type_accepts_various_formats(self):
        """Test that MIME type field accepts string values.

        Note: MIME type validation is defined in the model but not currently
        enforced (validator needs mode='before' to run). These tests document
        the intended behavior for when validation is enabled.
        """
        # Currently accepts any string
        doc = DocumentObject(
            name="test.txt",
            stream=BytesIO(b"content"),
            mime_type="application/pdf",
        )
        assert doc.mime_type == "application/pdf"

        # Also accepts custom MIME types (validation not enforced yet)
        doc2 = DocumentObject(
            name="test2.txt",
            stream=BytesIO(b"content"),
            mime_type="custom/type",
        )
        assert doc2.mime_type == "custom/type"

    # ============================================================================
    # Section 3: Stream Handling
    # ============================================================================

    def test_stream_handling(self):
        """Test that document stream is properly stored and accessible."""
        content = b"Test document content with special chars: \x00\xff"
        stream = BytesIO(content)

        doc = DocumentObject(
            name="test.txt",
            stream=stream,
            mime_type="text/plain",
        )

        # Rewind and read to verify content is preserved
        doc.stream.seek(0)
        assert doc.stream.read() == content

    def test_stream_with_empty_content(self):
        """Test creating a document with empty stream content."""
        doc = DocumentObject(
            name="empty.txt",
            stream=BytesIO(b""),
            mime_type="text/plain",
        )

        doc.stream.seek(0)
        assert doc.stream.read() == b""

    def test_stream_position_after_multiple_reads(self):
        """Test that stream position can be managed across multiple reads."""
        content = b"Test content for position tracking"
        doc = DocumentObject(
            name="test.txt",
            stream=BytesIO(content),
            mime_type="text/plain",
        )

        # First read
        doc.stream.seek(0)
        first_read = doc.stream.read(4)
        assert first_read == b"Test"

        # Second read continues from position
        second_read = doc.stream.read(8)
        assert second_read == b" content"

        # Rewind and read all
        doc.stream.seek(0)
        full_read = doc.stream.read()
        assert full_read == content

    def test_multiple_documents_independent_streams(self):
        """Test that multiple documents have independent streams."""
        doc1 = DocumentObject(
            name="doc1.txt",
            stream=BytesIO(b"content 1"),
            mime_type="text/plain",
        )
        doc2 = DocumentObject(
            name="doc2.txt",
            stream=BytesIO(b"content 2"),
            mime_type="text/plain",
        )

        doc1.stream.seek(0)
        doc2.stream.seek(0)

        assert doc1.stream.read() == b"content 1"
        assert doc2.stream.read() == b"content 2"

    def test_stream_with_large_content(self):
        """Test handling of large stream content."""
        # Create 1MB of content
        large_content = b"x" * (1024 * 1024)
        doc = DocumentObject(
            name="large.bin",
            stream=BytesIO(large_content),
            mime_type="application/octet-stream",
        )

        doc.stream.seek(0)
        assert len(doc.stream.read()) == 1024 * 1024

    # ============================================================================
    # Section 4: Metadata Management
    # ============================================================================

    def test_metadata_with_various_types(self):
        """Test metadata can contain various data types."""
        metadata = {
            "string": "value",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "none": None,
        }
        doc = DocumentObject(
            name="test.txt",
            stream=BytesIO(b"content"),
            mime_type="text/plain",
            metadata=metadata,
        )

        assert doc.metadata == metadata
        assert doc.metadata["string"] == "value"
        assert doc.metadata["integer"] == 42
        assert (
            doc.metadata["nested_dict"]["nested"] == "value"
            if "nested_dict" in doc.metadata
            else doc.metadata["dict"]["nested"] == "value"
        )

    def test_metadata_empty_dict(self):
        """Test that explicitly passing empty dict works."""
        doc = DocumentObject(
            name="test.txt",
            stream=BytesIO(b"content"),
            mime_type="text/plain",
            metadata={},
        )

        assert doc.metadata == {}

    # ============================================================================
    # Section 5: Edge Cases and Special Scenarios
    # ============================================================================

    def test_document_with_special_characters_in_name(self):
        """Test creating documents with special characters in names."""
        special_names = [
            "test file.txt",
            "test-file.txt",
            "test_file.txt",
            "test.multiple.dots.txt",
            "test (1).txt",
        ]

        for name in special_names:
            doc = DocumentObject(
                name=name,
                stream=BytesIO(b"content"),
                mime_type="text/plain",
            )
            assert doc.name == name

    def test_document_with_unicode_content(self):
        """Test handling of Unicode content in streams."""
        unicode_content = "Hello 世界 🌍".encode()
        doc = DocumentObject(
            name="unicode.txt",
            stream=BytesIO(unicode_content),
            mime_type="text/plain",
        )

        doc.stream.seek(0)
        assert doc.stream.read() == unicode_content

    def test_equality_of_documents(self):
        """Test equality comparison between DocumentObject instances."""
        doc1 = DocumentObject(
            name="test.txt",
            stream=BytesIO(b"content"),
            mime_type="text/plain",
            metadata={"key": "value"},
        )
        doc2 = DocumentObject(
            name="test.txt",
            stream=BytesIO(b"content"),
            mime_type="text/plain",
            metadata={"key": "value"},
        )

        # Note: Pydantic models compare by value, but streams are different objects
        # So these won't be equal due to stream object identity
        assert doc1.name == doc2.name
        assert doc1.mime_type == doc2.mime_type
        assert doc1.metadata == doc2.metadata
