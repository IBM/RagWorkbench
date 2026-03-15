"""
Comprehensive tests for DataLoaderCache.

Test Categories:
1. Initialization & Configuration
2. Serialization (Document lists to JSON)
3. Deserialization (JSON to Document lists)
4. Cache Operations (add/get)
5. Error Handling & Edge Cases
6. Integration Tests
"""

import base64
import json
import tempfile
from io import BytesIO
from pathlib import Path

import pytest

from ragworkbench.caching.data_loader_cache import (
    DataLoaderCache,
    DataLoaderCacheError,
    InvalidCacheFileError,
    StreamSerializationError,
    _CacheKeys,
)
from ragworkbench.datasets_loader.data_models import (
    DocumentObject,
    GroundTruthContextId,
    RagBenchmark,
    RagBenchmarkEntry,
    RagCorpus,
)

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
    """Create DataLoaderCache instance."""
    return DataLoaderCache(
        cache_dir=temp_cache_dir,
        dataset_name="test_dataset",
        split="train",
    )


@pytest.fixture
def sample_document():
    """Create a sample DocumentObject with text content."""
    content = b"This is a test document with some content."
    stream = BytesIO(content)
    return DocumentObject(
        name="test_doc.txt",
        mime_type="text/plain",
        metadata={"author": "Test Author", "year": 2024},
        stream=stream,
    )


@pytest.fixture
def binary_document():
    """Create a document with binary content."""
    # Simulate PDF-like binary content
    content = b"\x25\x50\x44\x46\x2d\x31\x2e\x34" + b"\x00" * 100
    stream = BytesIO(content)
    return DocumentObject(
        name="test_doc.pdf",
        mime_type="application/pdf",
        metadata={"pages": 10},
        stream=stream,
    )


@pytest.fixture
def sample_corpus(sample_document):
    """Create a sample RagCorpus."""
    return RagCorpus(documents=[sample_document])


@pytest.fixture
def multi_doc_corpus(sample_document, binary_document):
    """Create a corpus with multiple documents."""
    # Create additional documents
    doc3 = DocumentObject(
        name="doc3.txt",
        mime_type="text/plain",
        metadata={"id": 3},
        stream=BytesIO(b"Document 3 content"),
    )
    return RagCorpus(documents=[sample_document, binary_document, doc3])


@pytest.fixture
def sample_benchmark():
    """Create a sample RagBenchmark."""
    entries = [
        RagBenchmarkEntry(
            question_id="q1",
            question="What is the capital of France?",
            ground_truth_answers=["Paris"],
            ground_truths_context_ids=[
                GroundTruthContextId(document_id="doc1", page=1)
            ],
            is_answerable=True,
        ),
        RagBenchmarkEntry(
            question_id="q2",
            question="Who invented the telephone?",
            ground_truth_answers=["Alexander Graham Bell", "Bell"],
            ground_truths_context_ids=[
                GroundTruthContextId(document_id="doc2", page=5)
            ],
            is_answerable=True,
        ),
    ]
    return RagBenchmark(benchmark_entries=entries)


# ============================================================================
# Helper Functions
# ============================================================================


def create_document(
    name: str, content: bytes, mime_type: str, metadata: dict | None = None
) -> DocumentObject:
    """Helper to create DocumentObject with content."""
    stream = BytesIO(content)
    return DocumentObject(
        name=name,
        mime_type=mime_type,
        metadata=metadata or {},
        stream=stream,
    )


def create_corpus(num_docs: int) -> RagCorpus:
    """Helper to create RagCorpus with N documents."""
    documents = []
    for i in range(num_docs):
        doc = create_document(
            name=f"doc_{i}.txt",
            content=f"Content for document {i}".encode(),
            mime_type="text/plain",
            metadata={"index": i},
        )
        documents.append(doc)
    return RagCorpus(documents=documents)


def assert_streams_equal(stream1: BytesIO, stream2: BytesIO):
    """Assert two streams have identical content."""
    pos1, pos2 = stream1.tell(), stream2.tell()
    stream1.seek(0)
    stream2.seek(0)
    content1 = stream1.read()
    content2 = stream2.read()
    stream1.seek(pos1)
    stream2.seek(pos2)
    assert content1 == content2, "Stream contents do not match"


# ============================================================================
# Category 1: Initialization & Configuration Tests
# ============================================================================


class TestInitialization:
    """Tests for cache initialization and configuration."""

    def test_initialization_creates_cache_directory(self, temp_cache_dir):
        """Test that initialization creates the cache directory."""
        cache = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split="train",
        )

        assert cache.cache_path.exists()
        assert cache.cache_path.is_dir()

    def test_initialization_with_dataset_config(self, temp_cache_dir):
        """Test that config dict is properly hashed and used in path."""
        cache = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split="train",
        )

        # Path should include data_loader and a hash
        assert "data_loader" in str(cache.cache_path)
        # Should have created a subdirectory with hash
        assert cache.cache_path.parent.name == "data_loader"

    def test_cache_path_includes_dataset_hash(self, temp_cache_dir):
        """Test that different configs create different cache paths."""
        cache1 = DataLoaderCache(temp_cache_dir, dataset_name="dataset1", split="train")
        cache2 = DataLoaderCache(temp_cache_dir, dataset_name="dataset2", split="train")

        assert cache1.cache_path != cache2.cache_path

    def test_same_config_reuses_same_path(self, temp_cache_dir):
        """Test that same config uses same cache path."""
        cache1 = DataLoaderCache(
            temp_cache_dir, dataset_name="test_dataset", split="train"
        )
        cache2 = DataLoaderCache(
            temp_cache_dir, dataset_name="test_dataset", split="train"
        )

        assert cache1.cache_path == cache2.cache_path

    def test_initialization_with_none_split(self, temp_cache_dir):
        """Test that split=None works correctly."""
        cache = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split=None,
        )

        assert cache.cache_path.exists()
        assert cache.cache_path.is_dir()

    def test_different_splits_create_different_paths(self, temp_cache_dir):
        """Test that different splits create different cache paths."""
        cache_train = DataLoaderCache(
            temp_cache_dir, dataset_name="test_dataset", split="train"
        )
        cache_test = DataLoaderCache(
            temp_cache_dir, dataset_name="test_dataset", split="test"
        )
        cache_none = DataLoaderCache(
            temp_cache_dir, dataset_name="test_dataset", split=None
        )

        assert cache_train.cache_path != cache_test.cache_path
        assert cache_train.cache_path != cache_none.cache_path
        assert cache_test.cache_path != cache_none.cache_path


# ============================================================================
# Category 2: Serialization Tests (Document lists → JSON)
# ============================================================================


class TestSerialization:
    """Tests for document list serialization to JSON."""

    def test_save_documents_basic(self, sample_corpus):
        """Test serialization of simple document list with one document."""
        json_str = DataLoaderCache._save_documents_to_json(sample_corpus.documents)

        # Parse JSON
        data = json.loads(json_str)

        # Verify structure
        assert _CacheKeys.DOCUMENTS in data
        assert len(data[_CacheKeys.DOCUMENTS]) == 1

        # Verify document fields
        doc = data[_CacheKeys.DOCUMENTS][0]
        assert "name" in doc
        assert "mime_type" in doc
        assert "metadata" in doc
        assert "stream" in doc

    def test_save_documents_multiple_documents(self, multi_doc_corpus):
        """Test serialization of document list with multiple documents."""
        json_str = DataLoaderCache._save_documents_to_json(multi_doc_corpus.documents)
        data = json.loads(json_str)

        assert len(data[_CacheKeys.DOCUMENTS]) == 3

    def test_save_documents_preserves_metadata(self, sample_document):
        """Test that document metadata is preserved."""
        json_str = DataLoaderCache._save_documents_to_json([sample_document])
        data = json.loads(json_str)

        doc = data[_CacheKeys.DOCUMENTS][0]
        assert doc["metadata"] == sample_document.metadata
        assert doc["metadata"]["author"] == "Test Author"
        assert doc["metadata"]["year"] == 2024

    def test_save_documents_handles_binary_streams(self, binary_document):
        """Test serialization with binary content."""
        json_str = DataLoaderCache._save_documents_to_json([binary_document])
        data = json.loads(json_str)

        # Verify base64 encoding
        doc = data[_CacheKeys.DOCUMENTS][0]
        stream_b64 = doc["stream"]

        # Should be valid base64
        decoded = base64.b64decode(stream_b64)
        assert decoded.startswith(b"\x25\x50\x44\x46")  # PDF header

    def test_save_documents_handles_text_streams(self, sample_document):
        """Test serialization with text content."""
        json_str = DataLoaderCache._save_documents_to_json([sample_document])
        data = json.loads(json_str)

        doc = data[_CacheKeys.DOCUMENTS][0]
        decoded = base64.b64decode(doc["stream"])
        assert b"This is a test document" in decoded

    def test_save_documents_preserves_stream_position(self, sample_document):
        """Test that stream position is restored after serialization."""
        # Set stream to middle position
        sample_document.stream.seek(10)
        original_position = sample_document.stream.tell()

        DataLoaderCache._save_documents_to_json([sample_document])

        # Position should be restored
        assert sample_document.stream.tell() == original_position

    def test_save_documents_handles_special_characters(self):
        """Test document names and content with special characters."""
        doc = create_document(
            name="test_文档_🎉.txt",
            content="Content with unicode: 你好世界 🌍".encode(),
            mime_type="text/plain",
        )

        json_str = DataLoaderCache._save_documents_to_json([doc])
        data = json.loads(json_str)

        assert data[_CacheKeys.DOCUMENTS][0]["name"] == "test_文档_🎉.txt"


# ============================================================================
# Category 3: Deserialization Tests (JSON → Document lists)
# ============================================================================


class TestDeserialization:
    """Tests for loading document lists from JSON."""

    def test_load_documents_basic(self, temp_cache_dir, sample_document):
        """Test loading simple valid JSON file."""
        # Create JSON file
        json_str = DataLoaderCache._save_documents_to_json([sample_document])

        json_file = temp_cache_dir / "test_corpus.json"
        json_file.write_text(json_str, encoding="utf-8")

        # Load it back
        loaded_documents = DataLoaderCache._load_documents_from_json(json_file)

        assert len(loaded_documents) == 1
        assert loaded_documents[0].name == sample_document.name
        assert loaded_documents[0].mime_type == sample_document.mime_type

    def test_load_documents_multiple_documents(self, temp_cache_dir, multi_doc_corpus):
        """Test loading JSON with multiple documents."""
        json_str = DataLoaderCache._save_documents_to_json(multi_doc_corpus.documents)
        json_file = temp_cache_dir / "test_corpus.json"
        json_file.write_text(json_str, encoding="utf-8")

        loaded_documents = DataLoaderCache._load_documents_from_json(json_file)

        assert len(loaded_documents) == 3

    def test_load_documents_preserves_metadata(self, temp_cache_dir, sample_document):
        """Test that metadata is correctly restored."""
        json_str = DataLoaderCache._save_documents_to_json([sample_document])

        json_file = temp_cache_dir / "test_corpus.json"
        json_file.write_text(json_str, encoding="utf-8")

        loaded_documents = DataLoaderCache._load_documents_from_json(json_file)
        loaded_doc = loaded_documents[0]

        assert loaded_doc.metadata == sample_document.metadata

    def test_load_documents_reconstructs_streams(self, temp_cache_dir, sample_document):
        """Test that streams are readable and contain correct content."""
        json_str = DataLoaderCache._save_documents_to_json([sample_document])

        json_file = temp_cache_dir / "test_corpus.json"
        json_file.write_text(json_str, encoding="utf-8")

        loaded_documents = DataLoaderCache._load_documents_from_json(json_file)
        loaded_stream = loaded_documents[0].stream

        # Read content
        loaded_stream.seek(0)
        content = loaded_stream.read()

        sample_document.stream.seek(0)
        original_content = sample_document.stream.read()

        assert content == original_content

    def test_load_documents_invalid_json_raises_error(self, temp_cache_dir):
        """Test that malformed JSON raises InvalidCacheFileError."""
        json_file = temp_cache_dir / "invalid.json"
        json_file.write_text("{ invalid json }", encoding="utf-8")

        with pytest.raises(InvalidCacheFileError, match="Invalid JSON"):
            DataLoaderCache._load_documents_from_json(json_file)

    def test_load_documents_missing_documents_field_raises_error(self, temp_cache_dir):
        """Test that JSON without 'documents' key raises error."""
        json_file = temp_cache_dir / "missing_field.json"
        json_file.write_text('{"other_field": []}', encoding="utf-8")

        with pytest.raises(InvalidCacheFileError, match="missing required.*documents"):
            DataLoaderCache._load_documents_from_json(json_file)

    def test_load_documents_empty_documents_raises_error(self, temp_cache_dir):
        """Test that JSON with empty documents array raises error."""
        json_file = temp_cache_dir / "empty_docs.json"
        json_file.write_text('{"documents": []}', encoding="utf-8")

        with pytest.raises(InvalidCacheFileError, match="contains no documents"):
            DataLoaderCache._load_documents_from_json(json_file)

    def test_load_documents_missing_required_field_raises_error(self, temp_cache_dir):
        """Test that document missing required field raises error."""
        json_file = temp_cache_dir / "missing_name.json"
        json_data = {
            "documents": [
                {
                    # Missing "name" field
                    "mime_type": "text/plain",
                    "metadata": {},
                    "stream": "dGVzdA==",
                }
            ]
        }
        json_file.write_text(json.dumps(json_data), encoding="utf-8")

        with pytest.raises(
            InvalidCacheFileError, match="missing required field 'name'"
        ):
            DataLoaderCache._load_documents_from_json(json_file)

    def test_load_documents_invalid_base64_raises_error(self, temp_cache_dir):
        """Test that corrupt base64 raises StreamSerializationError."""
        json_file = temp_cache_dir / "invalid_base64.json"
        json_data = {
            "documents": [
                {
                    "name": "test.txt",
                    "mime_type": "text/plain",
                    "metadata": {},
                    "stream": "not-valid-base64!!!",
                }
            ]
        }
        json_file.write_text(json.dumps(json_data), encoding="utf-8")

        with pytest.raises(StreamSerializationError, match="Failed to decode stream"):
            DataLoaderCache._load_documents_from_json(json_file)


# ============================================================================
# Category 4: Cache Operations Tests
# ============================================================================


class TestCacheOperations:
    """Tests for cache add/get operations."""

    def test_add_and_get_documents_and_benchmark(
        self, cache_instance, sample_corpus, sample_benchmark
    ):
        """Test adding and retrieving both documents and benchmark."""
        cache_instance.add(sample_benchmark, sample_corpus.documents)

        loaded_documents, loaded_benchmark = cache_instance.get()

        assert loaded_documents is not None
        assert loaded_benchmark is not None
        assert len(loaded_documents) == len(sample_corpus.documents)
        assert len(loaded_benchmark.benchmark_entries) == len(
            sample_benchmark.benchmark_entries
        )

    def test_add_creates_two_cache_files(
        self, cache_instance, sample_corpus, sample_benchmark
    ):
        """Test that two JSON files are created (documents + benchmark)."""
        cache_instance.add(sample_benchmark, sample_corpus.documents)

        json_files = list(cache_instance.cache_path.glob("*.json"))
        assert len(json_files) == 2

        # Check filenames contain proper keys
        filenames = [f.name for f in json_files]
        assert any(_CacheKeys.DOCUMENTS in name for name in filenames)
        assert any(_CacheKeys.RAG_BENCHMARK in name for name in filenames)

    def test_get_returns_none_when_cache_empty(self, cache_instance):
        """Test that get returns (None, None) on empty cache."""
        documents, benchmark = cache_instance.get()

        assert documents is None
        assert benchmark is None

    def test_cache_persistence_across_instances(
        self, temp_cache_dir, sample_corpus, sample_benchmark
    ):
        """Test that data persists across different instances."""
        # Add data with first instance
        cache1 = DataLoaderCache(
            temp_cache_dir, dataset_name="test_dataset", split="train"
        )
        cache1.add(sample_benchmark, sample_corpus.documents)

        # Create new instance and retrieve
        cache2 = DataLoaderCache(
            temp_cache_dir, dataset_name="test_dataset", split="train"
        )
        loaded_documents, loaded_benchmark = cache2.get()

        assert loaded_documents is not None
        assert loaded_benchmark is not None

    def test_cache_hit_and_miss_tracking(
        self, cache_instance, sample_corpus, sample_benchmark
    ):
        """Test that cache statistics are tracked correctly."""
        # Initial state
        stats = cache_instance.get_cache_stats()
        assert stats["cache_hit"] == 0
        assert stats["cache_miss"] == 0

        # Add items
        cache_instance.add(sample_benchmark, sample_corpus.documents)

        # Get items (should be hits)
        cache_instance.get()
        cache_instance.get()

        stats = cache_instance.get_cache_stats()
        assert stats["cache_hit"] == 4  # 2 gets × 2 items each
        assert stats["total_entries"] == 2

    def test_multiple_add_overwrites_previous(
        self, cache_instance, sample_corpus, sample_benchmark
    ):
        """Test that adding again overwrites previous data."""
        # Add first time
        cache_instance.add(sample_benchmark, sample_corpus.documents)

        # Create new document list with different content
        new_doc = create_document(
            name="new_doc.txt",
            content=b"New content",
            mime_type="text/plain",
        )

        # Add again
        cache_instance.add(sample_benchmark, [new_doc])

        # Get should return latest
        loaded_documents, _ = cache_instance.get()
        assert loaded_documents[0].name == "new_doc.txt"

    def test_deepcopy_on_get(self, cache_instance, sample_corpus, sample_benchmark):
        """Test that get returns a deep copy."""
        cache_instance.add(sample_benchmark, sample_corpus.documents)

        # Get and modify
        documents1, _ = cache_instance.get()
        documents1[0].metadata["modified"] = True

        # Get again
        documents2, _ = cache_instance.get()

        # Original should be unchanged
        assert "modified" not in documents2[0].metadata


# ============================================================================
# Category 5: Content Type Detection Tests
# ============================================================================


class TestContentTypeDetection:
    """Tests for _read_content file type detection."""

    def test_read_content_detects_documents_file(
        self, temp_cache_dir, cache_instance, sample_corpus
    ):
        """Test that file with 'documents' in name is detected."""
        json_str = DataLoaderCache._save_documents_to_json(sample_corpus.documents)
        documents_file = temp_cache_dir / "documents_test.json"
        documents_file.write_text(json_str, encoding="utf-8")

        result = cache_instance._read_content(documents_file)

        assert isinstance(result, list)
        assert all(isinstance(doc, DocumentObject) for doc in result)

    def test_read_content_detects_benchmark_file(
        self, temp_cache_dir, cache_instance, sample_benchmark
    ):
        """Test that file with 'rag_benchmark' in name is detected."""
        benchmark_file = temp_cache_dir / "rag_benchmark_test.json"
        benchmark_file.write_text(
            sample_benchmark.model_dump_json(indent=2), encoding="utf-8"
        )

        result = cache_instance._read_content(benchmark_file)

        assert isinstance(result, RagBenchmark)

    def test_read_content_unknown_file_raises_error(
        self, temp_cache_dir, cache_instance
    ):
        """Test that file with neither keyword raises error."""
        unknown_file = temp_cache_dir / "unknown_file.json"
        unknown_file.write_text("{}", encoding="utf-8")

        with pytest.raises(
            InvalidCacheFileError, match="Cannot determine cache file type"
        ):
            cache_instance._read_content(unknown_file)

    def test_read_content_case_insensitive(
        self, temp_cache_dir, cache_instance, sample_corpus
    ):
        """Test that file detection is case-insensitive."""
        json_str = DataLoaderCache._save_documents_to_json(sample_corpus.documents)
        documents_file = temp_cache_dir / "DOCUMENTS_test.json"
        documents_file.write_text(json_str, encoding="utf-8")

        result = cache_instance._read_content(documents_file)

        assert isinstance(result, list)
        assert all(isinstance(doc, DocumentObject) for doc in result)


# ============================================================================
# Category 6: Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_content_to_json_unsupported_type_raises_error(self, cache_instance):
        """Test that passing invalid object type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type"):
            cache_instance._content_to_json("invalid_object")

    def test_content_to_json_none_raises_error(self, cache_instance):
        """Test that passing None raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported type"):
            cache_instance._content_to_json(None)


# ============================================================================
# Category 7: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for round-trip serialization."""

    def test_round_trip_serialization_text_documents(
        self, cache_instance, sample_document
    ):
        """Test complete round-trip with text documents."""
        original_documents = [sample_document]
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1",
                    question="Test question?",
                    ground_truth_answers=["Answer"],
                    is_answerable=True,
                )
            ]
        )

        # Add to cache
        cache_instance.add(benchmark, original_documents)

        # Retrieve from cache
        loaded_documents, loaded_benchmark = cache_instance.get()

        # Verify documents
        assert len(loaded_documents) == len(original_documents)
        assert_streams_equal(loaded_documents[0].stream, original_documents[0].stream)

        # Verify benchmark
        assert len(loaded_benchmark.benchmark_entries) == 1

    def test_round_trip_serialization_binary_documents(
        self, cache_instance, binary_document
    ):
        """Test round-trip with binary documents."""
        original_documents = [binary_document]
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1",
                    question="Test?",
                    is_answerable=True,
                )
            ]
        )

        cache_instance.add(benchmark, original_documents)
        loaded_documents, _ = cache_instance.get()

        assert_streams_equal(loaded_documents[0].stream, original_documents[0].stream)

    def test_round_trip_serialization_mixed_documents(
        self, cache_instance, multi_doc_corpus, sample_benchmark
    ):
        """Test round-trip with mix of text and binary documents."""
        cache_instance.add(sample_benchmark, multi_doc_corpus.documents)
        loaded_documents, _ = cache_instance.get()

        assert len(loaded_documents) == len(multi_doc_corpus.documents)

        # Verify each document
        for original, loaded in zip(
            multi_doc_corpus.documents, loaded_documents, strict=True
        ):
            assert original.name == loaded.name
            assert original.mime_type == loaded.mime_type
            assert_streams_equal(loaded.stream, original.stream)

    def test_round_trip_with_complex_metadata(self, cache_instance):
        """Test that complex nested metadata is preserved."""
        doc = create_document(
            name="complex.txt",
            content=b"Content",
            mime_type="text/plain",
            metadata={
                "nested": {"level1": {"level2": "value"}},
                "list": [1, 2, 3],
                "mixed": {"a": [1, 2], "b": {"c": "d"}},
            },
        )
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1", question="Test?", is_answerable=True
                )
            ]
        )

        cache_instance.add(benchmark, [doc])
        loaded_documents, _ = cache_instance.get()

        assert loaded_documents[0].metadata == doc.metadata

    def test_round_trip_preserves_stream_content(self, cache_instance):
        """Test that stream content is byte-for-byte identical."""
        # Create document with specific binary pattern
        content = bytes(range(256))  # All byte values
        doc = create_document(
            name="binary.bin",
            content=content,
            mime_type="application/octet-stream",
        )
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1", question="Test?", is_answerable=True
                )
            ]
        )

        cache_instance.add(benchmark, [doc])
        loaded_documents, _ = cache_instance.get()

        loaded_documents[0].stream.seek(0)
        loaded_content = loaded_documents[0].stream.read()

        assert loaded_content == content

    def test_large_corpus_performance(self, cache_instance):
        """Test with large document list (100+ documents)."""
        large_corpus = create_corpus(100)
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1", question="Test?", is_answerable=True
                )
            ]
        )

        # Should complete without errors
        cache_instance.add(benchmark, large_corpus.documents)
        loaded_documents, _ = cache_instance.get()

        assert len(loaded_documents) == 100

    def test_integration_with_benchmark_entries(self, cache_instance, sample_corpus):
        """Test that all benchmark fields are preserved."""
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1",
                    question="What is X?",
                    ground_truth_answers=["Answer 1", "Answer 2"],
                    ground_truths_context_ids=[
                        GroundTruthContextId(document_id="doc1", page=5, table_id="t1")
                    ],
                    is_answerable=True,
                    additional_information={
                        "category": "science",
                        "difficulty": "hard",
                    },
                )
            ]
        )

        cache_instance.add(benchmark, sample_corpus.documents)
        _, loaded_benchmark = cache_instance.get()

        entry = loaded_benchmark.benchmark_entries[0]
        assert entry.question_id == "q1"
        assert entry.question == "What is X?"
        assert entry.ground_truth_answers == ["Answer 1", "Answer 2"]
        assert len(entry.ground_truths_context_ids) == 1
        assert entry.ground_truths_context_ids[0].page == 5
        assert entry.additional_information["category"] == "science"


# ============================================================================
# Category 8: Exception Classes Tests
# ============================================================================


class TestExceptionClasses:
    """Tests for custom exception classes."""

    def test_data_loader_cache_error_is_exception(self):
        """Test that DataLoaderCacheError is an Exception."""
        assert issubclass(DataLoaderCacheError, Exception)

    def test_invalid_cache_file_error_is_data_loader_cache_error(self):
        """Test exception hierarchy."""
        assert issubclass(InvalidCacheFileError, DataLoaderCacheError)

    def test_stream_serialization_error_is_data_loader_cache_error(self):
        """Test exception hierarchy."""
        assert issubclass(StreamSerializationError, DataLoaderCacheError)

    def test_exceptions_can_be_raised_and_caught(self):
        """Test that exceptions work as expected."""
        with pytest.raises(InvalidCacheFileError):
            raise InvalidCacheFileError("Test error")

        with pytest.raises(DataLoaderCacheError):
            raise StreamSerializationError("Test error")


# ============================================================================
# Category 9: Cache Reuse with Different Sampling Tests
# ============================================================================


class TestCacheReuseWithDifferentSampling:
    """Test that cache is reused when loading same dataset with different sampling."""

    def test_cache_reuse_with_different_sampling_parameters(
        self, temp_cache_dir, sample_corpus, sample_benchmark
    ):
        """
        Test that the same dataset with different sampling configs reuses the cache.

        This test verifies the key behavior: DataLoaderCache stores the FULL
        unsampled dataset, independent of sampling parameters. When the same
        dataset is loaded with different sampling configurations, the cache
        should be reused.

        Scenario:
        1. Create cache instance #1 for dataset "test_dataset", split "train"
        2. Add full dataset (benchmark + documents) to cache
        3. Create cache instance #2 with SAME dataset/split
        4. Retrieve data from second instance
        5. Verify cache HIT occurred (data was reused, not reloaded)

        This simulates what happens in RagDataLoader when:
        - First call: DataSamplingParams(question_limit=5, seed=42)
        - Second call: DataSamplingParams(question_limit=10, seed=99)
        Both should use the same cached full dataset.
        """
        # Step 1: First cache instance (simulates first sampling config)
        cache1 = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split="train",
        )

        # Step 2: Add full dataset to cache
        cache1.add(sample_benchmark, sample_corpus.documents)

        # Step 3: Verify initial state
        stats1 = cache1.get_cache_stats()
        assert stats1["total_entries"] == 2  # documents + benchmark

        # Step 4: Second cache instance (simulates second sampling config)
        # This should reuse the same cache directory because dataset_name and split match
        cache2 = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",  # Same dataset
            split="train",  # Same split
        )

        # Step 5: Retrieve from cache
        loaded_documents, loaded_benchmark = cache2.get()

        # Step 6: Verify cache HIT occurred
        assert loaded_documents is not None, "Documents should be loaded from cache"
        assert loaded_benchmark is not None, "Benchmark should be loaded from cache"

        stats2 = cache2.get_cache_stats()
        assert (
            stats2["cache_hit"] == 2
        ), "Should have 2 cache hits (documents + benchmark)"
        assert stats2["cache_miss"] == 0, "Should have no cache misses"

        # Step 7: Verify data integrity
        assert len(loaded_documents) == len(
            sample_corpus.documents
        ), "All documents should be retrieved"
        assert len(loaded_benchmark.benchmark_entries) == len(
            sample_benchmark.benchmark_entries
        ), "All benchmark entries should be retrieved"

        # Step 8: Verify content matches original
        assert loaded_documents[0].name == sample_corpus.documents[0].name
        assert loaded_documents[0].mime_type == sample_corpus.documents[0].mime_type
        assert (
            loaded_benchmark.benchmark_entries[0].question_id
            == sample_benchmark.benchmark_entries[0].question_id
        )

    def test_different_splits_create_different_caches(
        self, temp_cache_dir, sample_corpus, sample_benchmark
    ):
        """
        Test that different splits create separate cache directories.

        When the split parameter differs, a different cache should be used,
        resulting in a cache MISS on the second load.
        """
        # Cache for train split
        cache_train = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split="train",
        )
        cache_train.add(sample_benchmark, sample_corpus.documents)

        # Cache for test split (different split, same dataset)
        cache_test = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split="test",
        )

        # Attempt to retrieve from test split cache
        documents, benchmark = cache_test.get()

        # Should be cache MISS because split is different
        assert documents is None, "Should not find documents in different split cache"
        assert benchmark is None, "Should not find benchmark in different split cache"

        stats = cache_test.get_cache_stats()
        assert stats["cache_miss"] == 2, "Should have 2 cache misses"
        assert stats["cache_hit"] == 0, "Should have no cache hits"

    def test_different_datasets_create_different_caches(
        self, temp_cache_dir, sample_corpus, sample_benchmark
    ):
        """
        Test that different datasets create separate cache directories.

        When the dataset_name differs, a different cache should be used,
        resulting in a cache MISS on the second load.
        """
        # Cache for dataset1
        cache1 = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="dataset1",
            split="train",
        )
        cache1.add(sample_benchmark, sample_corpus.documents)

        # Cache for dataset2 (different dataset, same split)
        cache2 = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="dataset2",
            split="train",
        )

        # Attempt to retrieve from dataset2 cache
        documents, benchmark = cache2.get()

        # Should be cache MISS because dataset is different
        assert documents is None, "Should not find documents in different dataset cache"
        assert benchmark is None, "Should not find benchmark in different dataset cache"

        stats = cache2.get_cache_stats()
        assert stats["cache_miss"] == 2, "Should have 2 cache misses"
        assert stats["cache_hit"] == 0, "Should have no cache hits"

    def test_multiple_cache_instances_share_same_cache(
        self, temp_cache_dir, sample_corpus, sample_benchmark
    ):
        """
        Test that multiple cache instances with same config share the same cache.

        This verifies the class-level cache mechanism that prevents reloading
        the same cache directory multiple times.
        """
        # First instance
        cache1 = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split="train",
        )
        cache1.add(sample_benchmark, sample_corpus.documents)

        # Second instance (same config)
        cache2 = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split="train",
        )

        # Third instance (same config)
        cache3 = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split="train",
        )

        # All should retrieve the same data
        docs1, bench1 = cache1.get()
        docs2, bench2 = cache2.get()
        docs3, bench3 = cache3.get()

        assert docs1 is not None and docs2 is not None and docs3 is not None
        assert bench1 is not None and bench2 is not None and bench3 is not None

        # Verify they all have cache hits
        assert cache1.get_cache_stats()["cache_hit"] == 2
        assert cache2.get_cache_stats()["cache_hit"] == 2
        assert cache3.get_cache_stats()["cache_hit"] == 2

    def test_cache_reuse_with_none_split(
        self, temp_cache_dir, sample_corpus, sample_benchmark
    ):
        """
        Test cache reuse when split is None.

        Datasets without splits should also benefit from cache reuse.
        """
        # First instance with split=None
        cache1 = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split=None,
        )
        cache1.add(sample_benchmark, sample_corpus.documents)

        # Second instance with split=None
        cache2 = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_name="test_dataset",
            split=None,
        )

        # Should retrieve from cache
        documents, benchmark = cache2.get()

        assert documents is not None
        assert benchmark is not None

        stats = cache2.get_cache_stats()
        assert stats["cache_hit"] == 2
        assert stats["cache_miss"] == 0
