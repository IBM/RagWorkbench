"""
Corpus and document-related pytest fixtures.

This module provides fixtures for creating sample documents,
document sets, and RagCorpus instances for testing.
"""

from io import BytesIO
from pathlib import Path

import pytest

from ragworkbench.datasets_loader.data_models.document_object import DocumentObject
from ragworkbench.datasets_loader.data_models.rag_corpus import RagCorpus


@pytest.fixture
def sample_document_objects() -> list[DocumentObject]:
    """
    Create a list of sample DocumentObject instances for testing.

    Returns:
        List of 5 DocumentObject instances with various MIME types.
    """
    documents = []
    for i in range(5):
        doc = DocumentObject(
            name=f"doc_{i}",
            stream=BytesIO(f"Content of document {i}".encode()),
            mime_type="application/pdf",
            metadata={"index": i, "category": f"category_{i % 2}"},
        )
        documents.append(doc)
    return documents


@pytest.fixture
def sample_rag_corpus(sample_document_objects: list[DocumentObject]) -> RagCorpus:
    """
    Create a sample RagCorpus instance.

    Args:
        sample_document_objects: Fixture providing document objects.

    Returns:
        RagCorpus instance with sample documents.
    """
    return RagCorpus(documents=sample_document_objects)


@pytest.fixture
def large_document_set() -> list[DocumentObject]:
    """
    Create a large set of documents for sampling tests.

    Returns:
        List of 20 DocumentObject instances.
    """
    documents = []
    for i in range(20):
        doc = DocumentObject(
            name=f"large_doc_{i}",
            stream=BytesIO(f"Content of large document {i}".encode()),
            mime_type="text/plain",
            metadata={"index": i},
        )
        documents.append(doc)
    return documents


@pytest.fixture
def temp_export_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory for file export tests.

    Args:
        tmp_path: pytest's built-in temporary directory fixture.

    Returns:
        Path to a temporary export directory.
    """
    export_dir = tmp_path / "exports"
    export_dir.mkdir(exist_ok=True)
    return export_dir
