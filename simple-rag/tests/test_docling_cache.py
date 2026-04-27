"""
Comprehensive tests for DoclingCache.

Tests DoclingCache-specific functionality that is NOT covered by abstract cache tests:
1. DoclingDocument serialization/deserialization
2. Document name-based hashing
"""

import json
import tempfile
from pathlib import Path

import pytest
from docling_core.types.doc.document import DoclingDocument
from simple_rag.docling_cache import DoclingCache

from ragworkbench.caching.abstract_file_system_cache import AbstractFileSystemCache

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache_instance(temp_cache_dir):
    """Create DoclingCache instance."""
    AbstractFileSystemCache.cache_path_to_contents.clear()
    return DoclingCache(cache_dir=temp_cache_dir)


@pytest.fixture
def sample_docling_doc():
    """Create a sample DoclingDocument."""
    return DoclingDocument(name="sample_doc.pdf")


# ============================================================================
# Minimal Test Suite - DoclingCache-Specific Functionality
# ============================================================================


class TestDoclingCacheSpecific:
    """Minimal tests for DoclingCache-specific functionality."""

    def test_docling_document_serialization_and_deserialization(
        self, temp_cache_dir, cache_instance, sample_docling_doc
    ):
        """Test DoclingDocument round-trip serialization (specific to DoclingCache)."""
        # Serialize
        json_str = cache_instance._content_to_json(sample_docling_doc)
        data = json.loads(json_str)

        # Verify it's valid JSON with DoclingDocument structure
        assert isinstance(data, dict)
        assert "name" in data
        assert data["name"] == "sample_doc.pdf"

        # Deserialize
        json_file = temp_cache_dir / "test_doc.json"
        json_file.write_text(json_str, encoding="utf-8")
        loaded_doc = cache_instance._read_content(json_file)

        # Verify type and equality
        assert isinstance(loaded_doc, DoclingDocument)
        assert loaded_doc.model_dump() == sample_docling_doc.model_dump()

    def test_document_name_based_hashing(self, cache_instance):
        """Test document name-based hash generation (specific to DoclingCache)."""
        # Same name produces same hash
        hash1 = cache_instance._get_parameters_hash("test_doc.pdf")
        hash2 = cache_instance._get_parameters_hash("test_doc.pdf")
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hash

        # Different names produce different hashes
        hash3 = cache_instance._get_parameters_hash("other_doc.pdf")
        assert hash1 != hash3

    def test_add_and_get_docling_document(self, cache_instance, sample_docling_doc):
        """Test basic cache operations with DoclingDocument."""
        doc_name = "test_doc.pdf"

        # Add and retrieve
        cache_instance.add(doc_name, sample_docling_doc)
        retrieved = cache_instance.get(doc_name)

        assert retrieved is not None
        assert isinstance(retrieved, DoclingDocument)
        assert retrieved.name == sample_docling_doc.name

        # Verify cache file created
        json_files = list(cache_instance.cache_path.glob("*.json"))
        assert len(json_files) == 1


# Made with Bob
