"""
Tests for RagCorpus data model.

This module tests the RagCorpus class focusing on business logic including
indexing, iteration, and the export_to_folder() method.
"""

from io import BytesIO

import pytest
from pydantic import ValidationError

from ragbench.datasets_loader.data_models.document_object import DocumentObject
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus


class TestRagCorpus:
    """Test suite for RagCorpus model."""

    def test_minimum_documents_validation(self):
        """Test that RagCorpus requires at least one document."""
        with pytest.raises(ValidationError):
            RagCorpus(documents=[])

    def test_len_method(self, sample_rag_corpus):
        """Test that __len__ returns the correct number of documents."""
        assert len(sample_rag_corpus) == 5

    def test_getitem_indexing(self, sample_rag_corpus):
        """Test document access by index."""
        # Positive indexing
        first_doc = sample_rag_corpus[0]
        assert first_doc.name == "doc_0"
        assert isinstance(first_doc, DocumentObject)

        # Negative indexing
        last_doc = sample_rag_corpus[-1]
        assert last_doc.name == "doc_4"

    def test_getitem_out_of_range(self, sample_rag_corpus):
        """Test that indexing out of range raises IndexError."""
        with pytest.raises(IndexError):
            _ = sample_rag_corpus[10]

        with pytest.raises(IndexError):
            _ = sample_rag_corpus[-10]

    def test_export_to_folder_creates_directory(self, sample_rag_corpus, tmp_path):
        """Test that export_to_folder creates the output directory if it doesn't exist."""
        export_dir = tmp_path / "new_export_dir"
        assert not export_dir.exists()

        sample_rag_corpus.export_to_folder(export_dir)

        assert export_dir.exists()
        assert export_dir.is_dir()

    def test_export_to_folder_writes_files(self, sample_rag_corpus, tmp_path):
        """Test that export_to_folder writes all documents to disk."""
        export_dir = tmp_path / "exports"
        sample_rag_corpus.export_to_folder(export_dir)

        # Check that all documents were exported
        exported_files = list(export_dir.glob("*"))
        assert len(exported_files) == 5

        # Verify file names
        file_names = {f.name for f in exported_files}
        expected_names = {f"doc_{i}.pdf" for i in range(5)}
        assert file_names == expected_names

    def test_export_to_folder_file_content(self, tmp_path):
        """Test that exported files contain the correct content."""
        content = b"Test document content"
        doc = DocumentObject(
            name="test_doc",
            stream=BytesIO(content),
            mime_type="text/plain",
        )
        corpus = RagCorpus(documents=[doc])

        export_dir = tmp_path / "exports"
        corpus.export_to_folder(export_dir)

        # Read the exported file and verify content
        exported_file = export_dir / "test_doc.txt"
        assert exported_file.exists()
        assert exported_file.read_bytes() == content

    def test_export_to_folder_adds_extension(self, tmp_path):
        """Test that export_to_folder adds file extension based on MIME type."""
        doc = DocumentObject(
            name="document_without_extension",
            stream=BytesIO(b"content"),
            mime_type="application/pdf",
        )
        corpus = RagCorpus(documents=[doc])

        export_dir = tmp_path / "exports"
        corpus.export_to_folder(export_dir)

        # Check that extension was added
        exported_file = export_dir / "document_without_extension.pdf"
        assert exported_file.exists()

    def test_export_to_folder_preserves_existing_extension(self, tmp_path):
        """Test that export_to_folder doesn't duplicate extensions."""
        doc = DocumentObject(
            name="document.pdf",
            stream=BytesIO(b"content"),
            mime_type="application/pdf",
        )
        corpus = RagCorpus(documents=[doc])

        export_dir = tmp_path / "exports"
        corpus.export_to_folder(export_dir)

        # Check that extension wasn't duplicated
        exported_file = export_dir / "document.pdf"
        assert exported_file.exists()
        assert not (export_dir / "document.pdf.pdf").exists()

    def test_export_to_folder_overwrites_existing_files(self, tmp_path):
        """Test that export_to_folder overwrites existing files."""
        export_dir = tmp_path / "exports"
        export_dir.mkdir()

        # Create an existing file
        existing_file = export_dir / "doc.txt"
        existing_file.write_bytes(b"old content")

        # Export corpus with same filename
        doc = DocumentObject(
            name="doc",
            stream=BytesIO(b"new content"),
            mime_type="text/plain",
        )
        corpus = RagCorpus(documents=[doc])
        corpus.export_to_folder(export_dir)

        # Verify file was overwritten
        assert existing_file.read_bytes() == b"new content"
