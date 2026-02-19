"""
Comprehensive tests for DataLoaderCache.

Test Categories:
1. Initialization & Configuration
2. Serialization (RagCorpus to JSON)
3. Deserialization (JSON to RagCorpus)
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

from ragbench.caching.data_loader_cache import (
    DataLoaderCache,
    DataLoaderCacheError,
    InvalidCacheFileError,
    StreamSerializationError,
    _CacheKeys,
)
from ragbench.datasets_loader.data_models import (
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
def dataset_config():
    """Sample dataset configuration."""
    return {"name": "test_dataset", "version": "1.0", "split": "train"}


@pytest.fixture
def cache_instance(temp_cache_dir, dataset_config):
    """Create DataLoaderCache instance."""
    return DataLoaderCache(
        cache_dir=temp_cache_dir,
        dataset_config_dict=dataset_config,
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
            ground_truth_context_ids=[GroundTruthContextId(document_id="doc1", page=1)],
            is_answerable=True,
        ),
        RagBenchmarkEntry(
            question_id="q2",
            question="Who invented the telephone?",
            ground_truth_answers=["Alexander Graham Bell", "Bell"],
            ground_truth_context_ids=[GroundTruthContextId(document_id="doc2", page=5)],
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

    def test_initialization_creates_cache_directory(
        self, temp_cache_dir, dataset_config
    ):
        """Test that initialization creates the cache directory."""
        cache = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_config_dict=dataset_config,
        )

        assert cache.cache_path.exists()
        assert cache.cache_path.is_dir()

    def test_initialization_with_dataset_config(self, temp_cache_dir, dataset_config):
        """Test that config dict is properly hashed and used in path."""
        cache = DataLoaderCache(
            cache_dir=temp_cache_dir,
            dataset_config_dict=dataset_config,
        )

        # Path should include data_loader and a hash
        assert "data_loader" in str(cache.cache_path)
        # Should have created a subdirectory with hash
        assert cache.cache_path.parent.name == "data_loader"

    def test_cache_path_includes_dataset_hash(self, temp_cache_dir):
        """Test that different configs create different cache paths."""
        config1 = {"name": "dataset1"}
        config2 = {"name": "dataset2"}

        cache1 = DataLoaderCache(temp_cache_dir, config1)
        cache2 = DataLoaderCache(temp_cache_dir, config2)

        assert cache1.cache_path != cache2.cache_path

    def test_same_config_reuses_same_path(self, temp_cache_dir, dataset_config):
        """Test that same config uses same cache path."""
        cache1 = DataLoaderCache(temp_cache_dir, dataset_config)
        cache2 = DataLoaderCache(temp_cache_dir, dataset_config)

        assert cache1.cache_path == cache2.cache_path


# ============================================================================
# Category 2: Serialization Tests (RagCorpus → JSON)
# ============================================================================


class TestSerialization:
    """Tests for RagCorpus serialization to JSON."""

    def test_save_rag_corpus_basic(self, sample_corpus):
        """Test serialization of simple corpus with one document."""
        json_str = DataLoaderCache._save_rag_corpus_to_json(sample_corpus)

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

    def test_save_rag_corpus_multiple_documents(self, multi_doc_corpus):
        """Test serialization of corpus with multiple documents."""
        json_str = DataLoaderCache._save_rag_corpus_to_json(multi_doc_corpus)
        data = json.loads(json_str)

        assert len(data[_CacheKeys.DOCUMENTS]) == 3

    def test_save_rag_corpus_preserves_metadata(self, sample_document):
        """Test that document metadata is preserved."""
        corpus = RagCorpus(documents=[sample_document])
        json_str = DataLoaderCache._save_rag_corpus_to_json(corpus)
        data = json.loads(json_str)

        doc = data[_CacheKeys.DOCUMENTS][0]
        assert doc["metadata"] == sample_document.metadata
        assert doc["metadata"]["author"] == "Test Author"
        assert doc["metadata"]["year"] == 2024

    def test_save_rag_corpus_handles_binary_streams(self, binary_document):
        """Test serialization with binary content."""
        corpus = RagCorpus(documents=[binary_document])
        json_str = DataLoaderCache._save_rag_corpus_to_json(corpus)
        data = json.loads(json_str)

        # Verify base64 encoding
        doc = data[_CacheKeys.DOCUMENTS][0]
        stream_b64 = doc["stream"]

        # Should be valid base64
        decoded = base64.b64decode(stream_b64)
        assert decoded.startswith(b"\x25\x50\x44\x46")  # PDF header

    def test_save_rag_corpus_handles_text_streams(self, sample_document):
        """Test serialization with text content."""
        corpus = RagCorpus(documents=[sample_document])
        json_str = DataLoaderCache._save_rag_corpus_to_json(corpus)
        data = json.loads(json_str)

        doc = data[_CacheKeys.DOCUMENTS][0]
        decoded = base64.b64decode(doc["stream"])
        assert b"This is a test document" in decoded

    def test_save_rag_corpus_preserves_stream_position(self, sample_document):
        """Test that stream position is restored after serialization."""
        # Set stream to middle position
        sample_document.stream.seek(10)
        original_position = sample_document.stream.tell()

        corpus = RagCorpus(documents=[sample_document])
        DataLoaderCache._save_rag_corpus_to_json(corpus)

        # Position should be restored
        assert sample_document.stream.tell() == original_position

    def test_save_rag_corpus_handles_special_characters(self):
        """Test document names and content with special characters."""
        doc = create_document(
            name="test_文档_🎉.txt",
            content="Content with unicode: 你好世界 🌍".encode(),
            mime_type="text/plain",
        )
        corpus = RagCorpus(documents=[doc])

        json_str = DataLoaderCache._save_rag_corpus_to_json(corpus)
        data = json.loads(json_str)

        assert data[_CacheKeys.DOCUMENTS][0]["name"] == "test_文档_🎉.txt"


# ============================================================================
# Category 3: Deserialization Tests (JSON → RagCorpus)
# ============================================================================


class TestDeserialization:
    """Tests for loading RagCorpus from JSON."""

    def test_load_rag_corpus_basic(self, temp_cache_dir, sample_document):
        """Test loading simple valid JSON file."""
        # Create JSON file
        corpus = RagCorpus(documents=[sample_document])
        json_str = DataLoaderCache._save_rag_corpus_to_json(corpus)

        json_file = temp_cache_dir / "test_corpus.json"
        json_file.write_text(json_str, encoding="utf-8")

        # Load it back
        loaded_corpus = DataLoaderCache._load_rag_corpus_from_json(json_file)

        assert len(loaded_corpus.documents) == 1
        assert loaded_corpus.documents[0].name == sample_document.name
        assert loaded_corpus.documents[0].mime_type == sample_document.mime_type

    def test_load_rag_corpus_multiple_documents(self, temp_cache_dir, multi_doc_corpus):
        """Test loading JSON with multiple documents."""
        json_str = DataLoaderCache._save_rag_corpus_to_json(multi_doc_corpus)
        json_file = temp_cache_dir / "test_corpus.json"
        json_file.write_text(json_str, encoding="utf-8")

        loaded_corpus = DataLoaderCache._load_rag_corpus_from_json(json_file)

        assert len(loaded_corpus.documents) == 3

    def test_load_rag_corpus_preserves_metadata(self, temp_cache_dir, sample_document):
        """Test that metadata is correctly restored."""
        corpus = RagCorpus(documents=[sample_document])
        json_str = DataLoaderCache._save_rag_corpus_to_json(corpus)

        json_file = temp_cache_dir / "test_corpus.json"
        json_file.write_text(json_str, encoding="utf-8")

        loaded_corpus = DataLoaderCache._load_rag_corpus_from_json(json_file)
        loaded_doc = loaded_corpus.documents[0]

        assert loaded_doc.metadata == sample_document.metadata

    def test_load_rag_corpus_reconstructs_streams(
        self, temp_cache_dir, sample_document
    ):
        """Test that streams are readable and contain correct content."""
        corpus = RagCorpus(documents=[sample_document])
        json_str = DataLoaderCache._save_rag_corpus_to_json(corpus)

        json_file = temp_cache_dir / "test_corpus.json"
        json_file.write_text(json_str, encoding="utf-8")

        loaded_corpus = DataLoaderCache._load_rag_corpus_from_json(json_file)
        loaded_stream = loaded_corpus.documents[0].stream

        # Read content
        loaded_stream.seek(0)
        content = loaded_stream.read()

        sample_document.stream.seek(0)
        original_content = sample_document.stream.read()

        assert content == original_content

    def test_load_rag_corpus_invalid_json_raises_error(self, temp_cache_dir):
        """Test that malformed JSON raises InvalidCacheFileError."""
        json_file = temp_cache_dir / "invalid.json"
        json_file.write_text("{ invalid json }", encoding="utf-8")

        with pytest.raises(InvalidCacheFileError, match="Invalid JSON"):
            DataLoaderCache._load_rag_corpus_from_json(json_file)

    def test_load_rag_corpus_missing_documents_field_raises_error(self, temp_cache_dir):
        """Test that JSON without 'documents' key raises error."""
        json_file = temp_cache_dir / "missing_field.json"
        json_file.write_text('{"other_field": []}', encoding="utf-8")

        with pytest.raises(InvalidCacheFileError, match="missing required.*documents"):
            DataLoaderCache._load_rag_corpus_from_json(json_file)

    def test_load_rag_corpus_empty_documents_raises_error(self, temp_cache_dir):
        """Test that JSON with empty documents array raises error."""
        json_file = temp_cache_dir / "empty_docs.json"
        json_file.write_text('{"documents": []}', encoding="utf-8")

        with pytest.raises(InvalidCacheFileError, match="contains no documents"):
            DataLoaderCache._load_rag_corpus_from_json(json_file)

    def test_load_rag_corpus_missing_required_field_raises_error(self, temp_cache_dir):
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
            DataLoaderCache._load_rag_corpus_from_json(json_file)

    def test_load_rag_corpus_invalid_base64_raises_error(self, temp_cache_dir):
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
            DataLoaderCache._load_rag_corpus_from_json(json_file)


# ============================================================================
# Category 4: Cache Operations Tests
# ============================================================================


class TestCacheOperations:
    """Tests for cache add/get operations."""

    def test_add_and_get_corpus_and_benchmark(
        self, cache_instance, sample_corpus, sample_benchmark
    ):
        """Test adding and retrieving both corpus and benchmark."""
        cache_instance.add(sample_benchmark, sample_corpus)

        loaded_corpus, loaded_benchmark = cache_instance.get()

        assert loaded_corpus is not None
        assert loaded_benchmark is not None
        assert len(loaded_corpus.documents) == len(sample_corpus.documents)
        assert len(loaded_benchmark.benchmark_entries) == len(
            sample_benchmark.benchmark_entries
        )

    def test_add_creates_two_cache_files(
        self, cache_instance, sample_corpus, sample_benchmark
    ):
        """Test that two JSON files are created (corpus + benchmark)."""
        cache_instance.add(sample_benchmark, sample_corpus)

        json_files = list(cache_instance.cache_path.glob("*.json"))
        assert len(json_files) == 2

        # Check filenames contain proper keys
        filenames = [f.name for f in json_files]
        assert any(_CacheKeys.RAG_CORPUS in name for name in filenames)
        assert any(_CacheKeys.RAG_BENCHMARK in name for name in filenames)

    def test_get_returns_none_when_cache_empty(self, cache_instance):
        """Test that get returns (None, None) on empty cache."""
        corpus, benchmark = cache_instance.get()

        assert corpus is None
        assert benchmark is None

    def test_cache_persistence_across_instances(
        self, temp_cache_dir, dataset_config, sample_corpus, sample_benchmark
    ):
        """Test that data persists across different instances."""
        # Add data with first instance
        cache1 = DataLoaderCache(temp_cache_dir, dataset_config)
        cache1.add(sample_benchmark, sample_corpus)

        # Create new instance and retrieve
        cache2 = DataLoaderCache(temp_cache_dir, dataset_config)
        loaded_corpus, loaded_benchmark = cache2.get()

        assert loaded_corpus is not None
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
        cache_instance.add(sample_benchmark, sample_corpus)

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
        cache_instance.add(sample_benchmark, sample_corpus)

        # Create new corpus with different content
        new_doc = create_document(
            name="new_doc.txt",
            content=b"New content",
            mime_type="text/plain",
        )
        new_corpus = RagCorpus(documents=[new_doc])

        # Add again
        cache_instance.add(sample_benchmark, new_corpus)

        # Get should return latest
        loaded_corpus, _ = cache_instance.get()
        assert loaded_corpus.documents[0].name == "new_doc.txt"

    def test_deepcopy_on_get(self, cache_instance, sample_corpus, sample_benchmark):
        """Test that get returns a deep copy."""
        cache_instance.add(sample_benchmark, sample_corpus)

        # Get and modify
        corpus1, _ = cache_instance.get()
        corpus1.documents[0].metadata["modified"] = True

        # Get again
        corpus2, _ = cache_instance.get()

        # Original should be unchanged
        assert "modified" not in corpus2.documents[0].metadata

    def test_add_with_wrong_argument_count_raises_error(
        self, cache_instance, sample_corpus
    ):
        """Test that add with wrong number of arguments raises ValueError."""
        with pytest.raises(ValueError, match="requires exactly 2 arguments"):
            cache_instance.add(sample_corpus)  # Only one argument


# ============================================================================
# Category 5: Content Type Detection Tests
# ============================================================================


class TestContentTypeDetection:
    """Tests for _read_content file type detection."""

    def test_read_content_detects_corpus_file(
        self, temp_cache_dir, cache_instance, sample_corpus
    ):
        """Test that file with 'rag_corpus' in name is detected."""
        json_str = DataLoaderCache._save_rag_corpus_to_json(sample_corpus)
        corpus_file = temp_cache_dir / "rag_corpus_test.json"
        corpus_file.write_text(json_str, encoding="utf-8")

        result = cache_instance._read_content(corpus_file)

        assert isinstance(result, RagCorpus)

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
        json_str = DataLoaderCache._save_rag_corpus_to_json(sample_corpus)
        corpus_file = temp_cache_dir / "RAG_CORPUS_test.json"
        corpus_file.write_text(json_str, encoding="utf-8")

        result = cache_instance._read_content(corpus_file)

        assert isinstance(result, RagCorpus)


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
        original_corpus = RagCorpus(documents=[sample_document])
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
        cache_instance.add(benchmark, original_corpus)

        # Retrieve from cache
        loaded_corpus, loaded_benchmark = cache_instance.get()

        # Verify corpus
        assert len(loaded_corpus.documents) == len(original_corpus.documents)
        assert_streams_equal(
            loaded_corpus.documents[0].stream, original_corpus.documents[0].stream
        )

        # Verify benchmark
        assert len(loaded_benchmark.benchmark_entries) == 1

    def test_round_trip_serialization_binary_documents(
        self, cache_instance, binary_document
    ):
        """Test round-trip with binary documents."""
        original_corpus = RagCorpus(documents=[binary_document])
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1",
                    question="Test?",
                    is_answerable=True,
                )
            ]
        )

        cache_instance.add(benchmark, original_corpus)
        loaded_corpus, _ = cache_instance.get()

        assert_streams_equal(
            loaded_corpus.documents[0].stream, original_corpus.documents[0].stream
        )

    def test_round_trip_serialization_mixed_documents(
        self, cache_instance, multi_doc_corpus, sample_benchmark
    ):
        """Test round-trip with mix of text and binary documents."""
        cache_instance.add(sample_benchmark, multi_doc_corpus)
        loaded_corpus, _ = cache_instance.get()

        assert len(loaded_corpus.documents) == len(multi_doc_corpus.documents)

        # Verify each document
        for original, loaded in zip(
            multi_doc_corpus.documents, loaded_corpus.documents, strict=True
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
        corpus = RagCorpus(documents=[doc])
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1", question="Test?", is_answerable=True
                )
            ]
        )

        cache_instance.add(benchmark, corpus)
        loaded_corpus, _ = cache_instance.get()

        assert loaded_corpus.documents[0].metadata == doc.metadata

    def test_round_trip_preserves_stream_content(self, cache_instance):
        """Test that stream content is byte-for-byte identical."""
        # Create document with specific binary pattern
        content = bytes(range(256))  # All byte values
        doc = create_document(
            name="binary.bin",
            content=content,
            mime_type="application/octet-stream",
        )
        corpus = RagCorpus(documents=[doc])
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1", question="Test?", is_answerable=True
                )
            ]
        )

        cache_instance.add(benchmark, corpus)
        loaded_corpus, _ = cache_instance.get()

        loaded_corpus.documents[0].stream.seek(0)
        loaded_content = loaded_corpus.documents[0].stream.read()

        assert loaded_content == content

    def test_large_corpus_performance(self, cache_instance):
        """Test with large corpus (100+ documents)."""
        large_corpus = create_corpus(100)
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1", question="Test?", is_answerable=True
                )
            ]
        )

        # Should complete without errors
        cache_instance.add(benchmark, large_corpus)
        loaded_corpus, _ = cache_instance.get()

        assert len(loaded_corpus.documents) == 100

    def test_integration_with_benchmark_entries(self, cache_instance, sample_corpus):
        """Test that all benchmark fields are preserved."""
        benchmark = RagBenchmark(
            benchmark_entries=[
                RagBenchmarkEntry(
                    question_id="q1",
                    question="What is X?",
                    ground_truth_answers=["Answer 1", "Answer 2"],
                    ground_truth_context_ids=[
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

        cache_instance.add(benchmark, sample_corpus)
        _, loaded_benchmark = cache_instance.get()

        entry = loaded_benchmark.benchmark_entries[0]
        assert entry.question_id == "q1"
        assert entry.question == "What is X?"
        assert entry.ground_truth_answers == ["Answer 1", "Answer 2"]
        assert len(entry.ground_truth_context_ids) == 1
        assert entry.ground_truth_context_ids[0].page == 5
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
