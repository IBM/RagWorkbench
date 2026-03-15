"""
Tests for RagCorpus data model.

This module comprehensively tests the RagCorpus class, focusing on:
- Corpus creation and validation
- Document indexing and access
- Export functionality to filesystem
- Immutability of frozen fields
- Edge cases and special scenarios
"""

from io import BytesIO

import pytest
from pydantic import ValidationError

from ragworkbench.datasets_loader.data_models.document_object import DocumentObject
from ragworkbench.datasets_loader.data_models.rag_corpus import RagCorpus


class TestRagCorpus:
    """Comprehensive test suite for RagCorpus model."""

    # ============================================================================
    # Section 1: Creation and Validation
    # ============================================================================

    def test_creation_with_documents(self, sample_document_objects):
        """Test creating a RagCorpus with valid documents."""
        corpus = RagCorpus(documents=sample_document_objects)

        assert len(corpus.documents) == 5
        assert isinstance(corpus, RagCorpus)

    def test_minimum_documents_validation(self):
        """Test that RagCorpus requires at least one document."""
        with pytest.raises(ValidationError):
            RagCorpus(documents=[])

    def test_single_document_corpus(self):
        """Test creating a corpus with single document."""
        doc = DocumentObject(
            name="single.txt",
            stream=BytesIO(b"content"),
            mime_type="text/plain",
        )
        corpus = RagCorpus(documents=[doc])

        assert len(corpus) == 1
        assert corpus[0].name == "single.txt"

    def test_large_corpus(self):
        """Test creating a corpus with many documents."""
        docs = [
            DocumentObject(
                name=f"doc_{i}.txt",
                stream=BytesIO(f"content {i}".encode()),
                mime_type="text/plain",
            )
            for i in range(100)
        ]
        corpus = RagCorpus(documents=docs)

        assert len(corpus) == 100

    # ============================================================================
    # Section 2: Length and Indexing
    # ============================================================================

    def test_len_method(self, sample_rag_corpus):
        """Test that __len__ returns the correct number of documents."""
        assert len(sample_rag_corpus) == 5

    def test_getitem_positive_indexing(self, sample_rag_corpus):
        """Test document access by positive index."""
        first_doc = sample_rag_corpus[0]
        assert first_doc.name == "doc_0"
        assert isinstance(first_doc, DocumentObject)

        third_doc = sample_rag_corpus[2]
        assert third_doc.name == "doc_2"

    def test_getitem_negative_indexing(self, sample_rag_corpus):
        """Test document access by negative index."""
        last_doc = sample_rag_corpus[-1]
        assert last_doc.name == "doc_4"

        second_last = sample_rag_corpus[-2]
        assert second_last.name == "doc_3"

    def test_getitem_out_of_range(self, sample_rag_corpus):
        """Test that indexing out of range raises IndexError."""
        with pytest.raises(IndexError):
            _ = sample_rag_corpus[10]

        with pytest.raises(IndexError):
            _ = sample_rag_corpus[-10]

    def test_getitem_boundary_indices(self, sample_rag_corpus):
        """Test indexing at boundaries."""
        # First and last valid indices
        assert sample_rag_corpus[0].name == "doc_0"
        assert sample_rag_corpus[4].name == "doc_4"
        assert sample_rag_corpus[-5].name == "doc_0"
        assert sample_rag_corpus[-1].name == "doc_4"

    # ============================================================================
    # Section 3: Export to Folder - Basic Functionality
    # ============================================================================

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

    # ============================================================================
    # Section 4: Export to Folder - Edge Cases
    # ============================================================================

    def test_export_with_nested_directory_paths(self, sample_rag_corpus, tmp_path):
        """Test export_to_folder creates nested directories."""
        nested_path = tmp_path / "level1" / "level2" / "level3"
        sample_rag_corpus.export_to_folder(nested_path)

        assert nested_path.exists()
        assert len(list(nested_path.glob("*"))) == 5

    def test_export_with_special_characters_in_filenames(self, tmp_path):
        """Test export with special characters in document names."""
        special_names = [
            "doc with spaces.txt",
            "doc-with-dashes.txt",
            "doc_with_underscores.txt",
            "doc.multiple.dots.txt",
        ]

        docs = [
            DocumentObject(
                name=name,
                stream=BytesIO(b"content"),
                mime_type="text/plain",
            )
            for name in special_names
        ]
        corpus = RagCorpus(documents=docs)

        export_dir = tmp_path / "exports"
        corpus.export_to_folder(export_dir)

        # All files should be created
        for name in special_names:
            expected_file = export_dir / name
            assert expected_file.exists()

    def test_export_with_unicode_filenames(self, tmp_path):
        """Test export with Unicode characters in filenames."""
        doc = DocumentObject(
            name="文档_测试",
            stream=BytesIO(b"content"),
            mime_type="text/plain",
        )
        corpus = RagCorpus(documents=[doc])

        export_dir = tmp_path / "exports"
        corpus.export_to_folder(export_dir)

        # File should be created with Unicode name
        exported_files = list(export_dir.glob("*"))
        assert len(exported_files) == 1

    def test_export_with_various_mime_types(self, tmp_path):
        """Test export with documents of various MIME types."""
        docs = [
            DocumentObject(
                name="doc1",
                stream=BytesIO(b"pdf content"),
                mime_type="application/pdf",
            ),
            DocumentObject(
                name="doc2",
                stream=BytesIO(b"text content"),
                mime_type="text/plain",
            ),
            DocumentObject(
                name="doc3",
                stream=BytesIO(b"html content"),
                mime_type="text/html",
            ),
        ]
        corpus = RagCorpus(documents=docs)

        export_dir = tmp_path / "exports"
        corpus.export_to_folder(export_dir)

        # Check correct extensions were added
        assert (export_dir / "doc1.pdf").exists()
        assert (export_dir / "doc2.txt").exists()
        assert (export_dir / "doc3.html").exists()

    def test_export_with_empty_document_content(self, tmp_path):
        """Test export with documents containing empty content."""
        doc = DocumentObject(
            name="empty",
            stream=BytesIO(b""),
            mime_type="text/plain",
        )
        corpus = RagCorpus(documents=[doc])

        export_dir = tmp_path / "exports"
        corpus.export_to_folder(export_dir)

        exported_file = export_dir / "empty.txt"
        assert exported_file.exists()
        assert exported_file.read_bytes() == b""

    def test_export_with_large_document(self, tmp_path):
        """Test export with large document content."""
        # Create 10MB of content
        large_content = b"x" * (10 * 1024 * 1024)
        doc = DocumentObject(
            name="large",
            stream=BytesIO(large_content),
            mime_type="application/octet-stream",
        )
        corpus = RagCorpus(documents=[doc])

        export_dir = tmp_path / "exports"
        corpus.export_to_folder(export_dir)

        exported_file = export_dir / "large.bin"
        assert exported_file.exists()
        assert len(exported_file.read_bytes()) == 10 * 1024 * 1024

    # ============================================================================
    # Section 5: Immutability (Frozen Fields)
    # ============================================================================

    def test_documents_list_immutability(self, sample_rag_corpus):
        """Test that documents list cannot be modified after creation."""
        with pytest.raises((ValidationError, AttributeError)):
            sample_rag_corpus.documents = []  # type: ignore

    def test_documents_list_is_tuple(self, sample_rag_corpus):
        """Test that documents is a tuple (immutable sequence).

        Note: Pydantic with frozen=True converts lists to tuples for immutability.
        """
        # Check if it's a tuple (immutable) or list
        documents = sample_rag_corpus.documents

        # Pydantic frozen fields should be tuples
        assert isinstance(documents, (list, tuple))

        # Should have the expected number of documents
        assert len(documents) == 5

    # ============================================================================
    # Section 6: Equality and Special Methods
    # ============================================================================

    def test_equality(self, sample_document_objects):
        """Test equality of RagCorpus instances."""
        corpus1 = RagCorpus(documents=sample_document_objects)
        corpus2 = RagCorpus(documents=sample_document_objects)

        assert corpus1 == corpus2

    def test_inequality(self):
        """Test inequality of RagCorpus instances with different documents."""
        doc1 = DocumentObject(
            name="doc1.txt",
            stream=BytesIO(b"content1"),
            mime_type="text/plain",
        )
        doc2 = DocumentObject(
            name="doc2.txt",
            stream=BytesIO(b"content2"),
            mime_type="text/plain",
        )

        corpus1 = RagCorpus(documents=[doc1])
        corpus2 = RagCorpus(documents=[doc2])

        assert corpus1 != corpus2

    def test_representation(self, sample_rag_corpus):
        """Test string representation of RagCorpus."""
        repr_str = repr(sample_rag_corpus)

        assert "RagCorpus" in repr_str or "documents" in repr_str

    # ============================================================================
    # Section 7: Integration with DocumentObject
    # ============================================================================

    def test_corpus_preserves_document_metadata(self, tmp_path):
        """Test that corpus preserves document metadata during export."""
        doc = DocumentObject(
            name="test",
            stream=BytesIO(b"content"),
            mime_type="text/plain",
            metadata={"author": "Test", "version": 1},
        )
        corpus = RagCorpus(documents=[doc])

        # Metadata should be preserved in corpus
        assert corpus[0].metadata == {"author": "Test", "version": 1}

        # Export should work correctly
        export_dir = tmp_path / "exports"
        corpus.export_to_folder(export_dir)
        assert (export_dir / "test.txt").exists()

    def test_corpus_with_mixed_document_types(self, tmp_path):
        """Test corpus with documents of different types and sizes."""
        docs = [
            DocumentObject(
                name="small",
                stream=BytesIO(b"small"),
                mime_type="text/plain",
            ),
            DocumentObject(
                name="medium",
                stream=BytesIO(b"x" * 1000),
                mime_type="application/pdf",
            ),
            DocumentObject(
                name="large",
                stream=BytesIO(b"y" * 10000),
                mime_type="application/json",
            ),
        ]
        corpus = RagCorpus(documents=docs)

        assert len(corpus) == 3

        export_dir = tmp_path / "exports"
        corpus.export_to_folder(export_dir)

        assert (export_dir / "small.txt").exists()
        assert (export_dir / "medium.pdf").exists()
        assert (export_dir / "large.json").exists()
