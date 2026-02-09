"""
Tests for DataLoaderCache.

This module comprehensively tests the DataLoaderCache class, focusing on:
- Cache initialization and directory creation
- Adding and retrieving cached items (RagCorpus and RagBenchmark)
- Serialization and deserialization of RagCorpus with binary streams
- Serialization and deserialization of RagBenchmark
- Cache hit/miss tracking
- Hash generation functions
- File system operations
- Edge cases and error handling
"""

import base64
import json
from io import BytesIO
from pathlib import Path

import pytest
import yaml

from ragbench.datasets_loader.data_loader_caching.data_loader_cache import (
    DataLoaderCache,
)
from ragbench.datasets_loader.data_models import (
    DocumentObject,
    GroundTruthContextId,
    RagBenchmark,
    RagBenchmarkEntry,
    RagCorpus,
)


class TestDataLoaderCache:
    """Comprehensive test suite for DataLoaderCache."""

    # ============================================================================
    # Section 1: Initialization and Setup
    # ============================================================================

    def test_initialization_creates_cache_directory(self, tmp_path):
        """Test that cache initialization creates the cache directory."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset", "version": "1.0"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        assert cache.cache_path.exists()
        assert cache.cache_path.is_dir()

    def test_initialization_creates_nested_directory_structure(self, tmp_path):
        """Test that cache creates nested directory based on config hash."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset", "version": "1.0"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        # Should create: cache_dir / "data_loader" / <hash>
        assert cache.cache_path.parent.name == "data_loader"
        assert cache.cache_path.exists()

    def test_initialization_creates_yaml_config_file(self, tmp_path):
        """Test that cache initialization creates a YAML config file."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset", "version": "1.0"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        yaml_file = cache.cache_path / "data_loader_cache.yaml"
        assert yaml_file.exists()

        # Verify YAML content
        yaml_content = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        assert yaml_content == {"dataset": config}

    def test_initialization_does_not_overwrite_existing_yaml(self, tmp_path):
        """Test that existing YAML config is not overwritten."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset", "version": "1.0"}

        # Create cache first time
        cache1 = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        yaml_file = cache1.cache_path / "data_loader_cache.yaml"
        original_content = yaml_file.read_text(encoding="utf-8")

        # Create cache second time with same config
        _ = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        # YAML should not be modified
        assert yaml_file.read_text(encoding="utf-8") == original_content

    def test_initialization_with_string_path(self, tmp_path):
        """Test that cache accepts string path as well as Path object."""
        cache_dir_str = str(tmp_path / "cache")
        config = {"dataset_name": "test_dataset"}

        cache = DataLoaderCache(cache_dir=cache_dir_str, dataset_config_dict=config)

        assert cache.cache_path.exists()
        assert isinstance(cache.cache_path, Path)

    def test_initialization_counters(self, tmp_path):
        """Test that cache initializes counters correctly."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        assert cache.cache_hit == 0
        assert cache.cache_miss == 0
        assert cache.read_files == 0

    # ============================================================================
    # Section 2: Adding Items to Cache
    # ============================================================================

    def test_add_rag_corpus_and_benchmark(
        self, tmp_path, sample_rag_corpus, sample_rag_benchmark
    ):
        """Test adding RagCorpus and RagBenchmark to cache."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache.add(rag_benchmark=sample_rag_benchmark, rag_corpus=sample_rag_corpus)

        # Check that files were created
        corpus_file = cache.cache_path / "rag_corpus.json"
        benchmark_file = cache.cache_path / "rag_benchmark.json"

        assert corpus_file.exists()
        assert benchmark_file.exists()

    def test_add_updates_cache_dict(
        self, tmp_path, sample_rag_corpus, sample_rag_benchmark
    ):
        """Test that add() updates the internal cache dictionary."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache.add(rag_benchmark=sample_rag_benchmark, rag_corpus=sample_rag_corpus)

        assert "rag_corpus" in cache.cache_dict
        assert "rag_benchmark" in cache.cache_dict
        cached_corpus = cache.cache_dict.get("rag_corpus")
        cached_benchmark = cache.cache_dict.get("rag_benchmark")
        assert isinstance(cached_corpus, RagCorpus)
        assert isinstance(cached_benchmark, RagBenchmark)

    def test_add_creates_deep_copy(
        self, tmp_path, sample_rag_corpus, sample_rag_benchmark
    ):
        """Test that add() creates a deep copy of cached items."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache.add(rag_benchmark=sample_rag_benchmark, rag_corpus=sample_rag_corpus)

        # Cached items should be different objects
        cached_corpus = cache.cache_dict.get("rag_corpus")
        cached_benchmark = cache.cache_dict.get("rag_benchmark")
        assert cached_corpus is not sample_rag_corpus
        assert cached_benchmark is not sample_rag_benchmark

    # ============================================================================
    # Section 3: Retrieving Items from Cache
    # ============================================================================

    def test_get_returns_cached_items(
        self, tmp_path, sample_rag_corpus, sample_rag_benchmark
    ):
        """Test that get() returns previously cached items."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache.add(rag_benchmark=sample_rag_benchmark, rag_corpus=sample_rag_corpus)

        cached_corpus, cached_benchmark = cache.get()

        assert cached_corpus is not None
        assert cached_benchmark is not None
        assert isinstance(cached_corpus, RagCorpus)
        assert isinstance(cached_benchmark, RagBenchmark)

    def test_get_returns_none_when_cache_empty(self, tmp_path):
        """Test that get() returns None when cache is empty."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cached_corpus, cached_benchmark = cache.get()

        assert cached_corpus is None
        assert cached_benchmark is None

    def test_get_returns_deep_copy(
        self, tmp_path, sample_rag_corpus, sample_rag_benchmark
    ):
        """Test that get() returns a deep copy of cached items."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache.add(rag_benchmark=sample_rag_benchmark, rag_corpus=sample_rag_corpus)

        cached_corpus1, cached_benchmark1 = cache.get()
        cached_corpus2, cached_benchmark2 = cache.get()

        # Should be different objects
        assert cached_corpus1 is not cached_corpus2
        assert cached_benchmark1 is not cached_benchmark2

    def test_cache_hit_tracking(
        self, tmp_path, sample_rag_corpus, sample_rag_benchmark
    ):
        """Test that cache hit counter is incremented correctly."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache.add(rag_benchmark=sample_rag_benchmark, rag_corpus=sample_rag_corpus)

        assert cache.cache_hit == 0

        cache.get()
        # Two hits: one for corpus, one for benchmark
        assert cache.cache_hit == 2

        cache.get()
        assert cache.cache_hit == 4

    def test_cache_miss_tracking(self, tmp_path):
        """Test that cache miss counter is incremented correctly."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        assert cache.cache_miss == 0

        cache.get()
        # Two misses: one for corpus, one for benchmark
        assert cache.cache_miss == 2

        cache.get()
        assert cache.cache_miss == 4

    # ============================================================================
    # Section 4: RagCorpus Serialization and Deserialization
    # ============================================================================

    def test_save_rag_corpus_to_json(self, tmp_path):
        """Test serialization of RagCorpus to JSON."""
        doc = DocumentObject(
            name="test_doc",
            stream=BytesIO(b"test content"),
            mime_type="text/plain",
            metadata={"key": "value"},
        )
        corpus = RagCorpus(documents=[doc])

        json_str = DataLoaderCache._save_rag_corpus_to_json(corpus)
        json_data = json.loads(json_str)

        assert "documents" in json_data
        assert len(json_data["documents"]) == 1
        assert json_data["documents"][0]["name"] == "test_doc"
        assert json_data["documents"][0]["mime_type"] == "text/plain"
        assert json_data["documents"][0]["metadata"] == {"key": "value"}
        assert "stream" in json_data["documents"][0]

    def test_save_rag_corpus_preserves_stream_position(self, tmp_path):
        """Test that serialization preserves stream position."""
        stream = BytesIO(b"test content")
        stream.seek(5)  # Move to position 5
        original_position = stream.tell()

        doc = DocumentObject(
            name="test_doc",
            stream=stream,
            mime_type="text/plain",
        )
        corpus = RagCorpus(documents=[doc])

        DataLoaderCache._save_rag_corpus_to_json(corpus)

        # Stream position should be restored
        assert stream.tell() == original_position

    def test_save_rag_corpus_encodes_binary_correctly(self, tmp_path):
        """Test that binary content is base64 encoded correctly."""
        binary_content = b"\x00\x01\x02\x03\xff\xfe\xfd"
        doc = DocumentObject(
            name="binary_doc",
            stream=BytesIO(binary_content),
            mime_type="application/octet-stream",
        )
        corpus = RagCorpus(documents=[doc])

        json_str = DataLoaderCache._save_rag_corpus_to_json(corpus)
        json_data = json.loads(json_str)

        # Decode and verify
        encoded_stream = json_data["documents"][0]["stream"]
        decoded_content = base64.b64decode(encoded_stream)
        assert decoded_content == binary_content

    def test_load_rag_corpus_from_json(self, tmp_path):
        """Test deserialization of RagCorpus from JSON file."""
        # Create a JSON file
        json_data = {
            "documents": [
                {
                    "name": "test_doc",
                    "mime_type": "text/plain",
                    "metadata": {"key": "value"},
                    "stream": base64.b64encode(b"test content").decode("utf-8"),
                }
            ]
        }
        json_file = tmp_path / "test_corpus.json"
        json_file.write_text(json.dumps(json_data), encoding="utf-8")

        corpus = DataLoaderCache._load_rag_corpus_from_json(json_file)

        assert isinstance(corpus, RagCorpus)
        assert len(corpus.documents) == 1
        assert corpus.documents[0].name == "test_doc"
        assert corpus.documents[0].mime_type == "text/plain"
        assert corpus.documents[0].metadata == {"key": "value"}

        # Verify stream content
        corpus.documents[0].stream.seek(0)
        assert corpus.documents[0].stream.read() == b"test content"

    def test_roundtrip_rag_corpus_serialization(self, tmp_path, sample_rag_corpus):
        """Test that RagCorpus can be serialized and deserialized correctly."""
        # Serialize
        json_str = DataLoaderCache._save_rag_corpus_to_json(sample_rag_corpus)
        json_file = tmp_path / "corpus.json"
        json_file.write_text(json_str, encoding="utf-8")

        # Deserialize
        loaded_corpus = DataLoaderCache._load_rag_corpus_from_json(json_file)

        # Verify
        assert len(loaded_corpus.documents) == len(sample_rag_corpus.documents)
        for orig_doc, loaded_doc in zip(
            sample_rag_corpus.documents, loaded_corpus.documents, strict=True
        ):
            assert loaded_doc.name == orig_doc.name
            assert loaded_doc.mime_type == orig_doc.mime_type
            assert loaded_doc.metadata == orig_doc.metadata

            # Verify stream content
            orig_doc.stream.seek(0)
            loaded_doc.stream.seek(0)
            assert loaded_doc.stream.read() == orig_doc.stream.read()

    # ============================================================================
    # Section 5: RagBenchmark Serialization
    # ============================================================================

    def test_content_to_json_for_benchmark(self, sample_rag_benchmark):
        """Test serialization of RagBenchmark to JSON."""
        cache_dir = Path("/tmp/cache")
        config = {"dataset_name": "test"}
        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        json_str = cache._content_to_json(sample_rag_benchmark)
        json_data = json.loads(json_str)

        assert "benchmark_entries" in json_data
        assert len(json_data["benchmark_entries"]) == len(
            sample_rag_benchmark.benchmark_entries
        )

    def test_content_to_json_for_corpus(self, sample_rag_corpus):
        """Test serialization of RagCorpus to JSON."""
        cache_dir = Path("/tmp/cache")
        config = {"dataset_name": "test"}
        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        json_str = cache._content_to_json(sample_rag_corpus)
        json_data = json.loads(json_str)

        assert "documents" in json_data

    def test_read_content_for_corpus(self, tmp_path):
        """Test reading RagCorpus from file."""
        # Create a corpus JSON file
        json_data = {
            "documents": [
                {
                    "name": "doc1",
                    "mime_type": "text/plain",
                    "metadata": {},
                    "stream": base64.b64encode(b"content").decode("utf-8"),
                }
            ]
        }
        corpus_file = tmp_path / "rag_corpus_test.json"
        corpus_file.write_text(json.dumps(json_data), encoding="utf-8")

        result = DataLoaderCache._read_content(corpus_file)

        assert isinstance(result, RagCorpus)
        assert len(result.documents) == 1

    def test_read_content_for_benchmark(self, tmp_path, sample_rag_benchmark):
        """Test reading RagBenchmark from file."""
        benchmark_file = tmp_path / "rag_benchmark_test.json"
        benchmark_file.write_text(
            sample_rag_benchmark.model_dump_json(), encoding="utf-8"
        )

        result = DataLoaderCache._read_content(benchmark_file)

        assert isinstance(result, RagBenchmark)
        assert len(result.benchmark_entries) == len(
            sample_rag_benchmark.benchmark_entries
        )

    def test_read_content_raises_for_unexpected_file(self, tmp_path):
        """Test that _read_content raises exception for unexpected files."""
        unexpected_file = tmp_path / "unexpected.json"
        unexpected_file.write_text("{}", encoding="utf-8")

        with pytest.raises(Exception, match="Got an unexpected file"):
            DataLoaderCache._read_content(unexpected_file)

    # ============================================================================
    # Section 6: Cache Persistence and Loading
    # ============================================================================

    def test_cache_persists_across_instances(
        self, tmp_path, sample_rag_corpus, sample_rag_benchmark
    ):
        """Test that cache persists and can be loaded by new instances."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        # Create cache and add items
        cache1 = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache1.add(rag_benchmark=sample_rag_benchmark, rag_corpus=sample_rag_corpus)

        # Create new cache instance with same config
        cache2 = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        # Should load existing cache files
        assert cache2.read_files == 2
        cached_corpus, cached_benchmark = cache2.get()

        assert cached_corpus is not None
        assert cached_benchmark is not None

    def test_cache_loads_existing_files_on_init(
        self, tmp_path, sample_rag_corpus, sample_rag_benchmark
    ):
        """Test that cache loads existing JSON files during initialization."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        # Create cache and add items
        cache1 = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache1.add(rag_benchmark=sample_rag_benchmark, rag_corpus=sample_rag_corpus)

        # Clear the class-level cache
        DataLoaderCache.cache_path_to_contents.clear()

        # Create new instance - should load from files
        cache2 = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        assert cache2.read_files == 2
        assert "rag_corpus" in cache2.cache_dict
        assert "rag_benchmark" in cache2.cache_dict

    def test_cache_reuses_loaded_content(self, tmp_path):
        """Test that cache reuses already loaded content from class variable."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test_dataset"}

        # Create first cache instance
        _ = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        # Create second instance with same path
        cache2 = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        # Should not read files again
        assert cache2.read_files == 0

    # ============================================================================
    # Section 7: Hash Functions
    # ============================================================================

    def test_get_hash_from_buffer(self):
        """Test hash generation from bytes."""
        data = b"test data"
        hash_result = DataLoaderCache.get_hash_from_buffer(data)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32  # MD5 hash is 32 hex characters

    def test_get_hash_from_buffer_consistency(self):
        """Test that same data produces same hash."""
        data = b"test data"
        hash1 = DataLoaderCache.get_hash_from_buffer(data)
        hash2 = DataLoaderCache.get_hash_from_buffer(data)

        assert hash1 == hash2

    def test_get_hash_from_buffer_different_data(self):
        """Test that different data produces different hashes."""
        hash1 = DataLoaderCache.get_hash_from_buffer(b"data1")
        hash2 = DataLoaderCache.get_hash_from_buffer(b"data2")

        assert hash1 != hash2

    def test_get_hash_string(self):
        """Test hash generation from string."""
        text = "test string"
        hash_result = DataLoaderCache.get_hash_string(text)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_get_hash_string_consistency(self):
        """Test that same string produces same hash."""
        text = "test string"
        hash1 = DataLoaderCache.get_hash_string(text)
        hash2 = DataLoaderCache.get_hash_string(text)

        assert hash1 == hash2

    def test_get_hash_dict(self):
        """Test hash generation from dictionary."""
        test_dict = {"key1": "value1", "key2": "value2"}
        hash_result = DataLoaderCache.get_hash_dict(test_dict)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_get_hash_dict_order_independent(self):
        """Test that dictionary hash is order-independent."""
        dict1 = {"key1": "value1", "key2": "value2"}
        dict2 = {"key2": "value2", "key1": "value1"}

        hash1 = DataLoaderCache.get_hash_dict(dict1)
        hash2 = DataLoaderCache.get_hash_dict(dict2)

        # Should be the same because items are sorted
        assert hash1 == hash2

    def test_get_hash_dict_with_nested_structures(self):
        """Test hash generation with nested dictionaries and lists."""
        test_dict = {
            "key1": {"nested": "value"},
            "key2": [1, 2, 3],
            "key3": "simple",
        }
        hash_result = DataLoaderCache.get_hash_dict(test_dict)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 32

    def test_serialize_dict_to_json(self):
        """Test dictionary serialization to JSON string."""
        test_dict = {"key1": "value1", "key2": "value2"}
        json_str = DataLoaderCache._serialize_dict_to_json(test_dict)

        assert isinstance(json_str, str)
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)  # Sorted items as list

    def test_serialize_dict_with_non_serializable_objects(self):
        """Test serialization with objects that need fallback serializer."""
        test_dict = {"key1": "value1", "key2": Path("/some/path")}
        json_str = DataLoaderCache._serialize_dict_to_json(test_dict)

        assert isinstance(json_str, str)
        # Path should be converted to string
        assert "/some/path" in json_str

    # ============================================================================
    # Section 8: Cache File Path Management
    # ============================================================================

    def test_get_parameters_hash(self):
        """Test that _get_parameters_hash returns the key as-is."""
        key = "test_key"
        result = DataLoaderCache._get_parameters_hash(key)

        assert result == key

    def test_format_cache_file_path(self, tmp_path):
        """Test cache file path formatting."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        file_path = cache._format_cache_file_path("test_key")

        assert file_path.suffix == ".json"
        assert file_path.stem == "test_key"
        assert file_path.parent == cache.cache_path

    # ============================================================================
    # Section 9: Edge Cases and Error Handling
    # ============================================================================

    def test_cache_with_empty_config(self, tmp_path):
        """Test cache creation with empty config dictionary."""
        cache_dir = tmp_path / "cache"
        config = {}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        assert cache.cache_path.exists()

    def test_cache_with_complex_config(self, tmp_path):
        """Test cache with complex nested configuration."""
        cache_dir = tmp_path / "cache"
        config = {
            "dataset_name": "test",
            "version": "1.0",
            "params": {"nested": {"deep": "value"}},
            "list": [1, 2, 3],
        }

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        assert cache.cache_path.exists()

    def test_cache_with_unicode_in_config(self, tmp_path):
        """Test cache with Unicode characters in configuration."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "测试数据集", "description": "Тест"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        assert cache.cache_path.exists()

    def test_multiple_caches_with_different_configs(self, tmp_path):
        """Test that different configs create different cache directories."""
        cache_dir = tmp_path / "cache"
        config1 = {"dataset_name": "dataset1"}
        config2 = {"dataset_name": "dataset2"}

        cache1 = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config1)
        cache2 = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config2)

        # Should have different cache paths
        assert cache1.cache_path != cache2.cache_path

    def test_cache_with_large_corpus(self, tmp_path):
        """Test caching with a large corpus."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "large_test"}

        # Create large corpus
        docs = [
            DocumentObject(
                name=f"doc_{i}",
                stream=BytesIO((f"Content {i}" * 100).encode()),
                mime_type="text/plain",
            )
            for i in range(50)
        ]
        large_corpus = RagCorpus(documents=docs)

        # Create simple benchmark
        entries = [
            RagBenchmarkEntry(
                question_id="q1",
                question="Test question?",
                ground_truth_answers=["answer"],
                ground_truth_context_ids=[GroundTruthContextId(document_id="doc_0")],
            )
        ]
        benchmark = RagBenchmark(benchmark_entries=entries)

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache.add(rag_benchmark=benchmark, rag_corpus=large_corpus)

        # Verify it can be retrieved
        cached_corpus, cached_benchmark = cache.get()
        assert cached_corpus is not None
        assert len(cached_corpus.documents) == 50

    def test_cache_with_binary_document_content(self, tmp_path):
        """Test caching documents with binary content."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "binary_test"}

        # Create document with binary content
        binary_content = bytes(range(256))  # All byte values
        doc = DocumentObject(
            name="binary_doc",
            stream=BytesIO(binary_content),
            mime_type="application/octet-stream",
        )
        corpus = RagCorpus(documents=[doc])

        entries = [
            RagBenchmarkEntry(
                question_id="q1",
                question="Test?",
                ground_truth_answers=["answer"],
                ground_truth_context_ids=[
                    GroundTruthContextId(document_id="binary_doc")
                ],
            )
        ]
        benchmark = RagBenchmark(benchmark_entries=entries)

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache.add(rag_benchmark=benchmark, rag_corpus=corpus)

        # Retrieve and verify
        cached_corpus, _ = cache.get()
        assert cached_corpus is not None
        cached_corpus.documents[0].stream.seek(0)
        retrieved_content = cached_corpus.documents[0].stream.read()

        assert retrieved_content == binary_content

    def test_add_with_key_method(self, tmp_path, sample_rag_corpus):
        """Test the _add_with_key internal method."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache._add_with_key("custom_key", sample_rag_corpus)

        assert "custom_key" in cache.cache_dict
        assert (cache.cache_path / "custom_key.json").exists()

    def test_get_with_key_method(self, tmp_path, sample_rag_corpus):
        """Test the _get_with_key internal method."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test"}

        cache = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache._add_with_key("custom_key", sample_rag_corpus)

        result, key = cache._get_with_key("custom_key")

        assert result is not None
        assert key == "custom_key"
        assert isinstance(result, RagCorpus)

    def test_class_level_cache_sharing(self, tmp_path):
        """Test that cache_path_to_contents is shared across instances."""
        cache_dir = tmp_path / "cache"
        config = {"dataset_name": "test"}

        cache1 = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)
        cache2 = DataLoaderCache(cache_dir=cache_dir, dataset_config_dict=config)

        # Both should reference the same class-level dictionary
        assert cache1.cache_path_to_contents is cache2.cache_path_to_contents
