"""
Comprehensive tests for GenerationCache.

Test Categories:
1. Initialization & Configuration
2. Query Hash Generation
3. Serialization (generation results to JSON)
4. Deserialization (JSON to generation results)
5. Cache Operations (add/get)
6. Data Integrity
7. Persistence
8. Edge Cases & Error Handling
9. Integration Tests
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from ragbench.caching.abstract_file_system_cache import AbstractFileSystemCache
from ragbench.caching.generation_cache import GenerationCache

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
    """Create GenerationCache instance without config."""
    # Clear class-level cache before each test
    AbstractFileSystemCache.cache_path_to_contents.clear()
    return GenerationCache(
        cache_dir=temp_cache_dir,
        config_dict={},
    )


@pytest.fixture
def cache_with_config(temp_cache_dir):
    """Create GenerationCache instance with config dict."""
    AbstractFileSystemCache.cache_path_to_contents.clear()
    config = {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
    }
    return GenerationCache(
        cache_dir=temp_cache_dir,
        config_dict=config,
    )


@pytest.fixture
def sample_string_query():
    """Simple string query."""
    return "What is the capital of France?"


@pytest.fixture
def sample_chat_query():
    """Chat-style list[dict] query."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]


@pytest.fixture
def sample_generation_result():
    """Typical LLM generation result."""
    return {
        "response": "The capital of France is Paris.",
        "model": "gpt-4",
        "tokens_used": 15,
        "finish_reason": "stop",
    }


@pytest.fixture
def complex_generation_result():
    """Complex nested generation result."""
    return {
        "response": "Paris is the capital.",
        "metadata": {
            "model": "gpt-4",
            "temperature": 0.7,
            "tokens": {
                "prompt": 10,
                "completion": 5,
                "total": 15,
            },
        },
        "choices": [
            {"text": "Paris", "score": 0.95},
            {"text": "Lyon", "score": 0.03},
        ],
        "timestamp": "2024-01-01T00:00:00Z",
    }


# ============================================================================
# Helper Functions
# ============================================================================


def create_generation_result(response: str, **kwargs) -> dict[str, Any]:
    """Helper to create generation result with custom fields."""
    result = {"response": response}
    result.update(kwargs)
    return result


# ============================================================================
# Category 1: Initialization & Configuration Tests
# ============================================================================


class TestInitialization:
    """Tests for cache initialization and configuration."""

    def test_initialization_creates_directory(self, temp_cache_dir):
        """Test that initialization creates the cache directory."""
        cache = GenerationCache(
            cache_dir=temp_cache_dir,
            config_dict={},
        )

        assert cache.cache_path.exists()
        assert cache.cache_path.is_dir()
        assert "generation" in str(cache.cache_path)

    def test_initialization_with_config_creates_subdirectory(self, temp_cache_dir):
        """Test that config dict creates a hashed subdirectory."""
        config = {"model": "gpt-4", "temperature": 0.7}
        cache = GenerationCache(
            cache_dir=temp_cache_dir,
            config_dict=config,
        )

        # Should create generation/<hash> directory
        assert cache.cache_path.parent.name == "generation"
        assert len(cache.cache_path.name) == 32  # MD5 hash length

    def test_initialization_creates_config_yaml(self, temp_cache_dir):
        """Test that config dict is saved as YAML file."""
        config = {"model": "gpt-4", "temperature": 0.7}
        cache = GenerationCache(
            cache_dir=temp_cache_dir,
            config_dict=config,
        )

        yaml_file = cache.cache_path / "generation_cache.yaml"
        assert yaml_file.exists()

        content = yaml_file.read_text(encoding="utf-8")
        assert "model" in content
        assert "gpt-4" in content
        assert "temperature" in content

    def test_different_configs_create_different_paths(self, temp_cache_dir):
        """Test that different configs create different cache paths."""
        cache1 = GenerationCache(temp_cache_dir, config_dict={"model": "gpt-4"})
        cache2 = GenerationCache(temp_cache_dir, config_dict={"model": "gpt-3.5"})

        assert cache1.cache_path != cache2.cache_path

    def test_same_config_reuses_same_path(self, temp_cache_dir):
        """Test that same config uses same cache path."""
        config = {"model": "gpt-4", "temperature": 0.7}
        cache1 = GenerationCache(temp_cache_dir, config_dict=config)
        cache2 = GenerationCache(temp_cache_dir, config_dict=config)

        assert cache1.cache_path == cache2.cache_path

    def test_initialization_with_path_object(self, temp_cache_dir):
        """Test initialization with Path object."""
        cache = GenerationCache(
            cache_dir=temp_cache_dir,  # Already a Path
            config_dict={},
        )

        assert cache.cache_path.exists()

    def test_initialization_with_string_path(self, temp_cache_dir):
        """Test initialization with string path."""
        cache = GenerationCache(
            cache_dir=str(temp_cache_dir),  # Convert to string
            config_dict={},
        )

        assert cache.cache_path.exists()

    def test_empty_cache_initialization(self, temp_cache_dir):
        """Test initializing cache in empty directory."""
        cache = GenerationCache(
            cache_dir=temp_cache_dir,
            config_dict={},
        )

        assert cache.read_files == 0
        assert len(cache.cache_dict) == 0
        assert cache.cache_hit == 0
        assert cache.cache_miss == 0


# ============================================================================
# Category 2: Query Hash Generation Tests
# ============================================================================


class TestQueryHashGeneration:
    """Tests for query hash generation."""

    def test_hash_generation_with_string_query(self, cache_instance):
        """Test hash generation with string query."""
        query = "What is the capital of France?"
        hash1 = cache_instance._get_parameters_hash(query)

        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length

    def test_hash_generation_with_chat_query(self, cache_instance):
        """Test hash generation with list[dict] query."""
        query = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        hash1 = cache_instance._get_parameters_hash(query)

        assert isinstance(hash1, str)
        assert len(hash1) == 32

    def test_hash_consistency_string_query(self, cache_instance):
        """Test that same string query produces same hash."""
        query = "What is the capital of France?"
        hash1 = cache_instance._get_parameters_hash(query)
        hash2 = cache_instance._get_parameters_hash(query)

        assert hash1 == hash2

    def test_hash_consistency_chat_query(self, cache_instance):
        """Test that same chat query produces same hash."""
        query = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        hash1 = cache_instance._get_parameters_hash(query)
        hash2 = cache_instance._get_parameters_hash(query)

        assert hash1 == hash2

    def test_hash_uniqueness_different_strings(self, cache_instance):
        """Test that different string queries produce different hashes."""
        query1 = "What is the capital of France?"
        query2 = "What is the capital of Germany?"
        hash1 = cache_instance._get_parameters_hash(query1)
        hash2 = cache_instance._get_parameters_hash(query2)

        assert hash1 != hash2

    def test_hash_uniqueness_different_chat_queries(self, cache_instance):
        """Test that different chat queries produce different hashes."""
        query1 = [{"role": "user", "content": "Hello"}]
        query2 = [{"role": "user", "content": "Hi"}]
        hash1 = cache_instance._get_parameters_hash(query1)
        hash2 = cache_instance._get_parameters_hash(query2)

        assert hash1 != hash2

    def test_hash_with_complex_nested_dict(self, cache_instance):
        """Test hash generation with complex nested dictionaries."""
        query = [
            {
                "role": "user",
                "content": "Test",
                "metadata": {
                    "nested": {"level1": {"level2": "value"}},
                    "list": [1, 2, 3],
                },
            }
        ]
        hash1 = cache_instance._get_parameters_hash(query)

        assert isinstance(hash1, str)
        assert len(hash1) == 32

    def test_hash_with_special_characters(self, cache_instance):
        """Test hash generation with special characters."""
        query = "What is 2+2? Answer: 4! 🎉"
        hash1 = cache_instance._get_parameters_hash(query)

        assert isinstance(hash1, str)
        assert len(hash1) == 32

    def test_hash_with_unicode_characters(self, cache_instance):
        """Test hash generation with Unicode characters."""
        query = "你好世界 🌍 Привет мир"
        hash1 = cache_instance._get_parameters_hash(query)

        assert isinstance(hash1, str)
        assert len(hash1) == 32

    def test_hash_order_matters_in_chat(self, cache_instance):
        """Test that message order affects hash in chat queries."""
        query1 = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ]
        query2 = [
            {"role": "assistant", "content": "B"},
            {"role": "user", "content": "A"},
        ]
        hash1 = cache_instance._get_parameters_hash(query1)
        hash2 = cache_instance._get_parameters_hash(query2)

        # Different order should produce different hash
        assert hash1 != hash2


# ============================================================================
# Category 3: Serialization Tests
# ============================================================================


class TestSerialization:
    """Tests for generation result serialization to JSON."""

    def test_content_to_json_simple_dict(self, cache_instance):
        """Test serialization of simple dictionary."""
        result = {"response": "Paris", "tokens": 5}
        json_str = cache_instance._content_to_json(result)

        # Parse JSON
        data = json.loads(json_str)
        assert data == result

    def test_content_to_json_nested_dict(
        self, cache_instance, complex_generation_result
    ):
        """Test serialization of nested dictionary."""
        json_str = cache_instance._content_to_json(complex_generation_result)
        data = json.loads(json_str)

        assert data == complex_generation_result
        assert data["metadata"]["tokens"]["total"] == 15

    def test_content_to_json_with_lists(self, cache_instance):
        """Test serialization with lists in result."""
        result = {
            "responses": ["Paris", "Lyon", "Marseille"],
            "scores": [0.9, 0.05, 0.05],
        }
        json_str = cache_instance._content_to_json(result)
        data = json.loads(json_str)

        assert data == result
        assert len(data["responses"]) == 3

    def test_content_to_json_with_various_types(self, cache_instance):
        """Test serialization with various data types."""
        result = {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }
        json_str = cache_instance._content_to_json(result)
        data = json.loads(json_str)

        assert data == result

    def test_content_to_json_indentation(self, cache_instance):
        """Test that JSON is properly indented (indent=4)."""
        result = {"key1": "value1", "key2": {"nested": "value2"}}
        json_str = cache_instance._content_to_json(result)

        # Check for indentation
        assert "\n" in json_str
        assert "    " in json_str  # 4 spaces

    def test_content_to_json_empty_dict(self, cache_instance):
        """Test serialization of empty dictionary."""
        result = {}
        json_str = cache_instance._content_to_json(result)
        data = json.loads(json_str)

        assert data == {}

    def test_content_to_json_preserves_unicode(self, cache_instance):
        """Test that Unicode characters are preserved."""
        result = {"response": "你好世界 🌍", "language": "Chinese"}
        json_str = cache_instance._content_to_json(result)
        data = json.loads(json_str)

        assert data["response"] == "你好世界 🌍"


# ============================================================================
# Category 4: Deserialization Tests
# ============================================================================


class TestDeserialization:
    """Tests for loading generation results from JSON."""

    def test_read_content_valid_json(self, temp_cache_dir, cache_instance):
        """Test loading valid JSON file."""
        result = {"response": "Paris", "tokens": 5}
        json_file = temp_cache_dir / "test.json"
        json_file.write_text(json.dumps(result, indent=4), encoding="utf-8")

        loaded = cache_instance._read_content(json_file)

        assert loaded == result

    def test_read_content_complex_json(
        self, temp_cache_dir, cache_instance, complex_generation_result
    ):
        """Test loading complex nested JSON."""
        json_file = temp_cache_dir / "test.json"
        json_file.write_text(
            json.dumps(complex_generation_result, indent=4), encoding="utf-8"
        )

        loaded = cache_instance._read_content(json_file)

        assert loaded == complex_generation_result
        assert loaded["metadata"]["tokens"]["total"] == 15

    def test_read_content_preserves_types(self, temp_cache_dir, cache_instance):
        """Test that data types are preserved during deserialization."""
        result = {
            "string": "text",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "none": None,
        }
        json_file = temp_cache_dir / "test.json"
        json_file.write_text(json.dumps(result), encoding="utf-8")

        loaded = cache_instance._read_content(json_file)

        assert isinstance(loaded["string"], str)
        assert isinstance(loaded["integer"], int)
        assert isinstance(loaded["float"], float)
        assert isinstance(loaded["boolean"], bool)
        assert loaded["none"] is None

    def test_read_content_malformed_json_raises_error(
        self, temp_cache_dir, cache_instance
    ):
        """Test that malformed JSON raises JSONDecodeError."""
        json_file = temp_cache_dir / "invalid.json"
        json_file.write_text("{ invalid json }", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            cache_instance._read_content(json_file)

    def test_read_content_empty_file_raises_error(self, temp_cache_dir, cache_instance):
        """Test that empty file raises JSONDecodeError."""
        json_file = temp_cache_dir / "empty.json"
        json_file.write_text("", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            cache_instance._read_content(json_file)


# ============================================================================
# Category 5: Cache Operations Tests
# ============================================================================


class TestCacheOperations:
    """Tests for cache add/get operations."""

    def test_add_and_get_string_query(
        self, cache_instance, sample_string_query, sample_generation_result
    ):
        """Test adding and retrieving with string query."""
        cache_instance.add(sample_string_query, sample_generation_result)

        retrieved = cache_instance.get(sample_string_query)

        assert retrieved == sample_generation_result
        assert cache_instance.cache_hit == 1
        assert cache_instance.cache_miss == 0

    def test_add_and_get_chat_query(
        self, cache_instance, sample_chat_query, sample_generation_result
    ):
        """Test adding and retrieving with chat query."""
        cache_instance.add(sample_chat_query, sample_generation_result)

        retrieved = cache_instance.get(sample_chat_query)

        assert retrieved == sample_generation_result

    def test_get_nonexistent_query(self, cache_instance):
        """Test getting a query that doesn't exist."""
        result = cache_instance.get("nonexistent query")

        assert result is None
        assert cache_instance.cache_hit == 0
        assert cache_instance.cache_miss == 1

    def test_cache_hit_tracking(
        self, cache_instance, sample_string_query, sample_generation_result
    ):
        """Test that cache hits are tracked correctly."""
        cache_instance.add(sample_string_query, sample_generation_result)

        # Multiple gets
        cache_instance.get(sample_string_query)
        cache_instance.get(sample_string_query)
        cache_instance.get(sample_string_query)

        assert cache_instance.cache_hit == 3
        assert cache_instance.cache_miss == 0

    def test_cache_miss_tracking(self, cache_instance):
        """Test that cache misses are tracked correctly."""
        cache_instance.get("query1")
        cache_instance.get("query2")
        cache_instance.get("query3")

        assert cache_instance.cache_hit == 0
        assert cache_instance.cache_miss == 3

    def test_multiple_add_and_get(self, cache_instance):
        """Test adding and retrieving multiple queries."""
        queries_and_results = {
            "query1": {"response": "answer1"},
            "query2": {"response": "answer2"},
            "query3": {"response": "answer3"},
        }

        # Add all
        for query, result in queries_and_results.items():
            cache_instance.add(query, result)

        # Retrieve all
        for query, expected_result in queries_and_results.items():
            retrieved = cache_instance.get(query)
            assert retrieved == expected_result

    def test_overwriting_existing_entry(self, cache_instance, sample_string_query):
        """Test that adding again overwrites previous entry."""
        result1 = {"response": "first answer"}
        result2 = {"response": "second answer"}

        cache_instance.add(sample_string_query, result1)
        cache_instance.add(sample_string_query, result2)

        retrieved = cache_instance.get(sample_string_query)
        assert retrieved == result2

    def test_cache_file_created_on_disk(
        self, cache_instance, sample_string_query, sample_generation_result
    ):
        """Test that cache files are created on disk."""
        cache_instance.add(sample_string_query, sample_generation_result)

        json_files = list(cache_instance.cache_path.glob("*.json"))
        assert len(json_files) == 1

        # Verify content
        content = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert content == sample_generation_result

    def test_get_cache_stats(self, cache_instance, sample_string_query):
        """Test cache statistics tracking."""
        # Initial stats
        stats = cache_instance.get_cache_stats()
        assert stats["cache_hit"] == 0
        assert stats["cache_miss"] == 0
        assert stats["total_entries"] == 0

        # Add items
        cache_instance.add(sample_string_query, {"response": "answer"})
        cache_instance.add("query2", {"response": "answer2"})

        # Get existing and non-existing
        cache_instance.get(sample_string_query)
        cache_instance.get(sample_string_query)
        cache_instance.get("nonexistent")

        stats = cache_instance.get_cache_stats()
        assert stats["cache_hit"] == 2
        assert stats["cache_miss"] == 1
        assert stats["total_entries"] == 2


# ============================================================================
# Category 6: Data Integrity Tests
# ============================================================================


class TestDataIntegrity:
    """Tests for data integrity and deep copying."""

    def test_deepcopy_on_get(
        self, cache_instance, sample_string_query, sample_generation_result
    ):
        """Test that get returns a deep copy, not the original."""
        cache_instance.add(sample_string_query, sample_generation_result)

        # Get and modify
        retrieved = cache_instance.get(sample_string_query)
        retrieved["modified"] = True

        # Get again and verify original is unchanged
        retrieved_again = cache_instance.get(sample_string_query)
        assert "modified" not in retrieved_again

    def test_deepcopy_on_add(self, cache_instance, sample_string_query):
        """Test that add stores a deep copy, not the original."""
        result = {"response": "answer", "metadata": {"key": "value"}}

        cache_instance.add(sample_string_query, result)

        # Modify original
        result["modified"] = True
        result["metadata"]["new_key"] = "new_value"

        # Verify cached value is unchanged
        retrieved = cache_instance.get(sample_string_query)
        assert "modified" not in retrieved
        assert "new_key" not in retrieved["metadata"]

    def test_nested_dict_deepcopy(self, cache_instance, sample_string_query):
        """Test deep copy with nested dictionaries."""
        result = {"level1": {"level2": {"level3": {"value": "original"}}}}

        cache_instance.add(sample_string_query, result)

        # Get and modify nested value
        retrieved = cache_instance.get(sample_string_query)
        retrieved["level1"]["level2"]["level3"]["value"] = "modified"

        # Get again and verify original is unchanged
        retrieved_again = cache_instance.get(sample_string_query)
        assert retrieved_again["level1"]["level2"]["level3"]["value"] == "original"

    def test_list_deepcopy(self, cache_instance, sample_string_query):
        """Test deep copy with lists."""
        result = {"responses": ["a", "b", "c"]}

        cache_instance.add(sample_string_query, result)

        # Get and modify list
        retrieved = cache_instance.get(sample_string_query)
        retrieved["responses"].append("d")

        # Get again and verify original is unchanged
        retrieved_again = cache_instance.get(sample_string_query)
        assert retrieved_again["responses"] == ["a", "b", "c"]


# ============================================================================
# Category 7: Persistence Tests
# ============================================================================


class TestPersistence:
    """Tests for cache persistence across instances."""

    def test_cache_persistence_across_instances(
        self, temp_cache_dir, sample_string_query, sample_generation_result
    ):
        """Test that cache persists across different instances."""
        config = {"model": "gpt-4"}

        # Create first instance and add item
        cache1 = GenerationCache(
            cache_dir=temp_cache_dir,
            config_dict=config,
        )
        cache1.add(sample_string_query, sample_generation_result)

        # Clear class-level cache to force reload from disk
        AbstractFileSystemCache.cache_path_to_contents.clear()

        # Create second instance
        cache2 = GenerationCache(
            cache_dir=temp_cache_dir,
            config_dict=config,
        )

        # Should load from disk
        assert cache2.read_files == 1
        retrieved = cache2.get(sample_string_query)
        assert retrieved == sample_generation_result

    def test_class_level_cache_sharing(self, temp_cache_dir):
        """Test that multiple instances share class-level cache."""
        config = {"model": "gpt-4"}

        # Create first instance
        cache1 = GenerationCache(temp_cache_dir, config_dict=config)
        cache1.add("query1", {"response": "answer1"})

        # Create second instance (should use class-level cache)
        cache2 = GenerationCache(temp_cache_dir, config_dict=config)

        # Second instance should not read from disk
        assert cache2.read_files == 0

        # But should have access to the data
        assert cache2.get("query1") == {"response": "answer1"}

    def test_cache_with_existing_files(self, temp_cache_dir):
        """Test loading cache with existing files."""
        config = {"model": "gpt-4"}
        cache_path = temp_cache_dir / "generation"

        # Create cache directory and hash subdirectory
        hash_dir = cache_path / AbstractFileSystemCache.get_hash_dict(config)
        hash_dir.mkdir(parents=True)

        # Create some cache files manually
        hash1 = AbstractFileSystemCache.get_hash_string("query1")
        hash2 = AbstractFileSystemCache.get_hash_string("query2")
        (hash_dir / f"{hash1}.json").write_text(
            '{"response": "answer1"}', encoding="utf-8"
        )
        (hash_dir / f"{hash2}.json").write_text(
            '{"response": "answer2"}', encoding="utf-8"
        )

        # Initialize cache
        cache = GenerationCache(
            cache_dir=temp_cache_dir,
            config_dict=config,
        )

        assert cache.read_files == 2
        assert len(cache.cache_dict) == 2


# ============================================================================
# Category 8: Edge Cases & Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_string_query(self, cache_instance):
        """Test with empty string query."""
        result = {"response": "answer"}
        cache_instance.add("", result)

        retrieved = cache_instance.get("")
        assert retrieved == result

    def test_empty_list_query(self, cache_instance):
        """Test with empty list query."""
        result = {"response": "answer"}
        cache_instance.add([], result)

        retrieved = cache_instance.get([])
        assert retrieved == result

    def test_very_long_query(self, cache_instance):
        """Test with very long query string."""
        long_query = "A" * 10000  # 10k characters
        result = {"response": "answer"}

        cache_instance.add(long_query, result)
        retrieved = cache_instance.get(long_query)

        assert retrieved == result

    def test_query_with_newlines(self, cache_instance):
        """Test query with newline characters."""
        query = "Line 1\nLine 2\nLine 3"
        result = {"response": "answer"}

        cache_instance.add(query, result)
        retrieved = cache_instance.get(query)

        assert retrieved == result

    def test_query_with_special_json_characters(self, cache_instance):
        """Test query with characters that need JSON escaping."""
        query = 'Query with "quotes" and \\backslashes\\ and \ttabs'
        result = {"response": "answer"}

        cache_instance.add(query, result)
        retrieved = cache_instance.get(query)

        assert retrieved == result

    def test_result_with_special_characters(self, cache_instance):
        """Test generation result with special characters."""
        query = "test query"
        result = {
            "response": 'Answer with "quotes" and \\backslashes\\',
            "special": "\n\t\r",
        }

        cache_instance.add(query, result)
        retrieved = cache_instance.get(query)

        assert retrieved == result

    def test_result_with_emoji(self, cache_instance):
        """Test generation result with emoji."""
        query = "test query"
        result = {"response": "Great answer! 🎉 👍 ✨"}

        cache_instance.add(query, result)
        retrieved = cache_instance.get(query)

        assert retrieved == result

    def test_chat_query_with_empty_messages(self, cache_instance):
        """Test chat query with empty message content."""
        query = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
        ]
        result = {"response": "answer"}

        cache_instance.add(query, result)
        retrieved = cache_instance.get(query)

        assert retrieved == result

    def test_large_generation_result(self, cache_instance):
        """Test with large generation result."""
        query = "test query"
        result = {
            "response": "A" * 100000,  # 100k characters
            "metadata": {"tokens": 50000},
        }

        cache_instance.add(query, result)
        retrieved = cache_instance.get(query)

        assert retrieved == result
        assert len(retrieved["response"]) == 100000


# ============================================================================
# Category 9: Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for round-trip operations."""

    def test_round_trip_string_query(
        self, cache_instance, sample_string_query, sample_generation_result
    ):
        """Test complete round-trip with string query."""
        # Add to cache
        cache_instance.add(sample_string_query, sample_generation_result)

        # Retrieve from cache
        retrieved = cache_instance.get(sample_string_query)

        # Verify
        assert retrieved == sample_generation_result
        assert retrieved["response"] == sample_generation_result["response"]
        assert retrieved["tokens_used"] == sample_generation_result["tokens_used"]

    def test_round_trip_chat_query(
        self, cache_instance, sample_chat_query, complex_generation_result
    ):
        """Test complete round-trip with chat query."""
        cache_instance.add(sample_chat_query, complex_generation_result)
        retrieved = cache_instance.get(sample_chat_query)

        assert retrieved == complex_generation_result
        assert retrieved["metadata"]["tokens"]["total"] == 15

    def test_round_trip_preserves_all_fields(self, cache_instance):
        """Test that all fields are preserved in round-trip."""
        query = "test query"
        result = {
            "response": "Paris",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "tokens_used": 15,
            "finish_reason": "stop",
            "metadata": {
                "timestamp": "2024-01-01T00:00:00Z",
                "user_id": "user123",
            },
        }

        cache_instance.add(query, result)
        retrieved = cache_instance.get(query)

        # Verify all fields
        for key, value in result.items():
            assert retrieved[key] == value

    def test_multiple_queries_in_same_cache(self, cache_instance):
        """Test multiple different queries in same cache."""
        queries_and_results = [
            ("What is 2+2?", {"response": "4", "tokens": 3}),
            ("What is the capital of France?", {"response": "Paris", "tokens": 5}),
            (
                [{"role": "user", "content": "Hello"}],
                {"response": "Hi there!", "tokens": 4},
            ),
        ]

        # Add all
        for query, result in queries_and_results:
            cache_instance.add(query, result)

        # Verify all
        for query, expected_result in queries_and_results:
            retrieved = cache_instance.get(query)
            assert retrieved == expected_result

    def test_cache_miss_then_hit(self, cache_instance, sample_string_query):
        """Test cache miss followed by cache hit."""
        # First get - should be a miss
        result1 = cache_instance.get(sample_string_query)
        assert result1 is None
        assert cache_instance.cache_miss == 1
        assert cache_instance.cache_hit == 0

        # Add the item
        result = {"response": "Paris"}
        cache_instance.add(sample_string_query, result)

        # Second get - should be a hit
        result2 = cache_instance.get(sample_string_query)
        assert result2 == result
        assert cache_instance.cache_miss == 1
        assert cache_instance.cache_hit == 1

    def test_realistic_llm_scenario(self, cache_instance):
        """Test with realistic LLM generation scenario."""
        # Simulate a chat conversation
        query = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        result = {
            "response": "The capital of France is Paris. Paris is not only the capital but also the largest city in France, known for its art, fashion, gastronomy, and culture.",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000,
            "tokens": {
                "prompt": 25,
                "completion": 35,
                "total": 60,
            },
            "finish_reason": "stop",
            "created": 1704067200,
        }

        # Cache the result
        cache_instance.add(query, result)

        # Retrieve it
        retrieved = cache_instance.get(query)

        assert retrieved == result
        assert "Paris" in retrieved["response"]

    def test_config_affects_cache_isolation(self, temp_cache_dir):
        """Test that different configs create isolated caches."""
        query = "What is 2+2?"
        result1 = {"response": "4", "model": "gpt-4"}
        result2 = {"response": "Four", "model": "gpt-3.5"}

        # Create two caches with different configs
        cache1 = GenerationCache(temp_cache_dir, config_dict={"model": "gpt-4"})
        cache2 = GenerationCache(temp_cache_dir, config_dict={"model": "gpt-3.5"})

        # Add different results to each
        cache1.add(query, result1)
        cache2.add(query, result2)

        # Verify isolation
        assert cache1.get(query) == result1
        assert cache2.get(query) == result2

    def test_clear_cache(self, cache_instance, sample_string_query):
        """Test clearing the cache."""
        # Add items
        cache_instance.add(sample_string_query, {"response": "answer1"})
        cache_instance.add("query2", {"response": "answer2"})

        assert len(cache_instance.cache_dict) == 2

        # Clear cache
        cache_instance.clear_cache()

        assert len(cache_instance.cache_dict) == 0
        assert cache_instance.get(sample_string_query) is None
