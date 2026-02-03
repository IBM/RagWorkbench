"""
Tests for DocumentObject data model.

This module tests the DocumentObject class focusing on document creation
and metadata handling. Note: MIME type validation is defined but not currently
enforced due to missing mode='before' in the field_validator decorator.
"""

from io import BytesIO

from ragbench.datasets_loader.data_models.document_object import DocumentObject


class TestDocumentObject:
    """Test suite for DocumentObject model."""

    def test_creation_with_all_fields(self):
        """Test creating a DocumentObject with all fields."""
        doc = DocumentObject(
            name="test.pdf",
            stream=BytesIO(b"test content"),
            mime_type="application/pdf",
            metadata={"author": "Test Author", "pages": 10},
        )

        assert doc.name == "test.pdf"
        assert doc.mime_type == "application/pdf"
        assert doc.metadata == {"author": "Test Author", "pages": 10}

    def test_creation_with_common_mime_types(self):
        """Test creating documents with various common MIME types."""
        mime_types = [
            "application/pdf",
            "text/plain",
            "text/html",
            "image/jpeg",
            "application/json",
        ]

        for mime_type in mime_types:
            doc = DocumentObject(
                name=f"test.{mime_type.split('/')[-1]}",
                stream=BytesIO(b"content"),
                mime_type=mime_type,
            )
            assert doc.mime_type == mime_type

    def test_metadata_defaults_to_empty_dict(self):
        """Test that metadata defaults to an empty dictionary when not provided."""
        doc = DocumentObject(
            name="test.txt",
            stream=BytesIO(b"content"),
            mime_type="text/plain",
        )

        assert doc.metadata == {}
        assert isinstance(doc.metadata, dict)

    def test_stream_handling(self):
        """Test that document stream is properly stored and accessible."""
        content = b"Test document content with special chars"
        stream = BytesIO(content)

        doc = DocumentObject(
            name="test.txt",
            stream=stream,
            mime_type="text/plain",
        )

        # Rewind and read to verify content is preserved
        doc.stream.seek(0)
        assert doc.stream.read() == content

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
