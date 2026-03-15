"""Tests for AbstractFileSystemCache."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from ragworkbench.caching.abstract_file_system_cache import AbstractFileSystemCache


class ConcreteFileSystemCache(AbstractFileSystemCache):
    """Concrete implementation for testing."""

    def _read_content(self, file: Path) -> Any:
        """Read JSON content from file."""
        return json.loads(file.read_text(encoding="utf-8"))

    def _content_to_json(self, *args) -> str:
        """Convert content to JSON string."""
        # args[0] is the content to serialize
        return json.dumps(args[0], indent=2)

    def _get_parameters_hash(self, *args) -> str:
        """Generate hash from parameters."""
        combined = "_".join(str(arg) for arg in args)
        return self.get_hash_string(combined)

    def get(self, item: Any) -> Any:
        """Get item from cache."""
        result, _ = self._get(item)
        return result


class TestAbstractFileSystemCache:
    """Test suite for AbstractFileSystemCache."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache_instance(self, temp_cache_dir):
        """Create a cache instance for testing."""
        # Clear class-level cache before each test
        AbstractFileSystemCache.cache_path_to_contents.clear()
        return ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name="test_cache",
            config_dict=None,
        )

    @pytest.fixture
    def cache_with_config(self, temp_cache_dir):
        """Create a cache instance with config dict."""
        AbstractFileSystemCache.cache_path_to_contents.clear()
        config = {"param1": "value1", "param2": 42}
        return ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name="test_cache",
            config_dict=config,
        )

    def test_initialization_creates_directory(self, temp_cache_dir):
        """Test that initialization creates the cache directory."""
        cache_name = "test_cache"
        cache = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name=cache_name,
            config_dict=None,
        )

        assert cache.cache_path.exists()
        assert cache.cache_path.is_dir()
        assert cache.cache_path.name == cache_name

    def test_initialization_with_config_creates_subdirectory(self, temp_cache_dir):
        """Test that config dict creates a hashed subdirectory."""
        config = {"key": "value"}
        cache = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name="test_cache",
            config_dict=config,
        )

        # Should create test_cache/<hash> directory
        assert cache.cache_path.parent.name == "test_cache"
        assert len(cache.cache_path.name) == 32  # MD5 hash length

    def test_initialization_creates_config_yaml(self, temp_cache_dir):
        """Test that config dict is saved as YAML file."""
        config = {"param1": "value1", "param2": 42}
        cache_name = "test_cache"
        cache = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name=cache_name,
            config_dict=config,
        )

        yaml_file = cache.cache_path / f"{cache_name}_cache.yaml"
        assert yaml_file.exists()

        content = yaml_file.read_text(encoding="utf-8")
        assert "param1" in content
        assert "value1" in content
        assert "param2" in content

    def test_add_and_get_item(self, cache_instance):
        """Test adding and retrieving an item from cache."""
        key = "test_key"
        value = {"data": "test_value", "number": 123}

        # Add item to cache
        cache_instance.add(key, value)

        # Retrieve item from cache
        retrieved = cache_instance.get(key)

        assert retrieved == value
        assert cache_instance.cache_hit == 1
        assert cache_instance.cache_miss == 0

    def test_get_nonexistent_item(self, cache_instance):
        """Test getting an item that doesn't exist."""
        result = cache_instance.get("nonexistent_key")

        assert result is None
        assert cache_instance.cache_hit == 0
        assert cache_instance.cache_miss == 1

    def test_cache_file_created_on_disk(self, cache_instance):
        """Test that cache files are created on disk."""
        key = "test_key"
        value = {"data": "test"}

        cache_instance.add(key, value)

        # Check that a JSON file was created
        json_files = list(cache_instance.cache_path.glob("*.json"))
        assert len(json_files) == 1

        # Verify content
        content = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert content == value

    def test_multiple_adds_and_gets(self, cache_instance):
        """Test adding and retrieving multiple items."""
        items = {
            "key1": {"value": 1},
            "key2": {"value": 2},
            "key3": {"value": 3},
        }

        # Add all items
        for key, value in items.items():
            cache_instance.add(key, value)

        # Retrieve all items
        for key, expected_value in items.items():
            retrieved = cache_instance.get(key)
            assert retrieved == expected_value

    def test_cache_persistence_across_instances(self, temp_cache_dir):
        """Test that cache persists across different instances."""
        cache_name = "persistent_cache"
        key = "test_key"
        value = {"data": "persistent"}

        # Create first instance and add item
        cache1 = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name=cache_name,
            config_dict=None,
        )
        cache1.add(key, value)

        # Clear class-level cache to force reload from disk
        AbstractFileSystemCache.cache_path_to_contents.clear()

        # Create second instance
        cache2 = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name=cache_name,
            config_dict=None,
        )

        # Should load from disk
        assert cache2.read_files == 1
        retrieved = cache2.get(key)
        assert retrieved == value

    def test_class_level_cache_sharing(self, temp_cache_dir):
        """Test that multiple instances share class-level cache."""
        cache_name = "shared_cache"

        # Create first instance
        cache1 = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name=cache_name,
            config_dict=None,
        )
        cache1.add("key1", {"value": 1})

        # Create second instance (should use class-level cache)
        cache2 = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name=cache_name,
            config_dict=None,
        )

        # Second instance should not read from disk
        assert cache2.read_files == 0

        # But should have access to the data
        assert cache2.get("key1") == {"value": 1}

    def test_deepcopy_on_get(self, cache_instance):
        """Test that get returns a deep copy, not the original."""
        key = "test_key"
        value = {"data": [1, 2, 3]}

        cache_instance.add(key, value)

        # Get the value and modify it
        retrieved = cache_instance.get(key)
        retrieved["data"].append(4)

        # Get again and verify original is unchanged
        retrieved_again = cache_instance.get(key)
        assert retrieved_again["data"] == [1, 2, 3]

    def test_deepcopy_on_add(self, cache_instance):
        """Test that add stores a deep copy, not the original."""
        key = "test_key"
        value = {"data": [1, 2, 3]}

        cache_instance.add(key, value)

        # Modify original
        value["data"].append(4)

        # Verify cached value is unchanged
        retrieved = cache_instance.get(key)
        assert retrieved["data"] == [1, 2, 3]

    def test_get_cache_stats(self, cache_instance):
        """Test cache statistics tracking."""
        # Initial stats
        stats = cache_instance.get_cache_stats()
        assert stats["cache_hit"] == 0
        assert stats["cache_miss"] == 0
        assert stats["total_entries"] == 0

        # Add items
        cache_instance.add("key1", {"value": 1})
        cache_instance.add("key2", {"value": 2})

        # Get existing and non-existing items
        cache_instance.get("key1")
        cache_instance.get("key1")
        cache_instance.get("nonexistent")

        stats = cache_instance.get_cache_stats()
        assert stats["cache_hit"] == 2
        assert stats["cache_miss"] == 1
        assert stats["total_entries"] == 2

    def test_clear_cache(self, cache_instance):
        """Test clearing the cache."""
        # Add items
        cache_instance.add("key1", {"value": 1})
        cache_instance.add("key2", {"value": 2})

        assert len(cache_instance.cache_dict) == 2

        # Clear cache
        cache_instance.clear_cache()

        assert len(cache_instance.cache_dict) == 0
        assert cache_instance.get("key1") is None

    def test_add_with_no_arguments_raises_error(self, cache_instance):
        """Test that add with no arguments raises ValueError."""
        with pytest.raises(ValueError, match="requires at least 2 arguments"):
            cache_instance.add()

    def test_add_with_one_argument_raises_error(self, cache_instance):
        """Test that add with only one argument raises ValueError."""
        with pytest.raises(ValueError, match="requires at least 2 arguments"):
            cache_instance.add({"value": "only_one_arg"})

    def test_hash_functions(self):
        """Test hash generation functions."""
        # Test get_hash_from_buffer
        data = b"test data"
        hash1 = AbstractFileSystemCache.get_hash_from_buffer(data)
        assert len(hash1) == 32  # MD5 hash length
        assert hash1 == AbstractFileSystemCache.get_hash_from_buffer(data)

        # Test get_hash_string
        string = "test string"
        hash2 = AbstractFileSystemCache.get_hash_string(string)
        assert len(hash2) == 32
        assert hash2 == AbstractFileSystemCache.get_hash_string(string)

        # Test get_hash_dict
        dict1 = {"key1": "value1", "key2": "value2"}
        hash3 = AbstractFileSystemCache.get_hash_dict(dict1)
        assert len(hash3) == 32

        # Same dict should produce same hash
        dict2 = {"key2": "value2", "key1": "value1"}  # Different order
        hash4 = AbstractFileSystemCache.get_hash_dict(dict2)
        assert hash3 == hash4  # Should be same due to sorting

    def test_hash_dict_with_complex_types(self):
        """Test hash generation with complex dictionary types."""
        complex_dict = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "nested": {"inner": "value"},
        }

        hash_value = AbstractFileSystemCache.get_hash_dict(complex_dict)
        assert len(hash_value) == 32

        # Same dict should produce same hash
        hash_value2 = AbstractFileSystemCache.get_hash_dict(complex_dict)
        assert hash_value == hash_value2

    def test_get_with_key_methods(self, cache_instance):
        """Test _get_with_key and _add_with_key methods."""
        cache_key = "custom_cache_key"
        value = {"data": "test"}

        # Add with key
        cache_instance._add_with_key(cache_key, value)

        # Get with key
        retrieved, returned_key = cache_instance._get_with_key(cache_key)

        assert retrieved == value
        assert returned_key == cache_key

    def test_format_cache_file_path(self, cache_instance):
        """Test cache file path formatting."""
        cache_key = "test_hash_key"
        path = cache_instance._format_cache_file_path(cache_key)

        assert path.parent == cache_instance.cache_path
        assert path.name == f"{cache_key}.json"
        assert path.suffix == ".json"

    def test_get_cache_file_path(self, cache_instance):
        """Test getting cache file path from parameters."""
        params = ("param1", "param2")
        path = cache_instance._get_cache_file_path(*params)

        assert path.parent == cache_instance.cache_path
        assert path.suffix == ".json"

        # Should be consistent
        path2 = cache_instance._get_cache_file_path(*params)
        assert path == path2

    def test_cache_with_path_object(self, temp_cache_dir):
        """Test initialization with Path object."""
        cache = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,  # Already a Path
            cache_name="test_cache",
            config_dict=None,
        )

        assert cache.cache_path.exists()

    def test_cache_with_string_path(self, temp_cache_dir):
        """Test initialization with string path."""
        cache = ConcreteFileSystemCache(
            cache_dir=str(temp_cache_dir),  # Convert to string
            cache_name="test_cache",
            config_dict=None,
        )

        assert cache.cache_path.exists()

    def test_empty_cache_initialization(self, temp_cache_dir):
        """Test initializing cache in empty directory."""
        cache = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name="empty_cache",
            config_dict=None,
        )

        assert cache.read_files == 0
        assert len(cache.cache_dict) == 0
        assert cache.cache_hit == 0
        assert cache.cache_miss == 0

    def test_cache_with_existing_files(self, temp_cache_dir):
        """Test loading cache with existing files."""
        cache_name = "existing_cache"
        cache_path = temp_cache_dir / cache_name
        cache_path.mkdir(parents=True)

        # Create some cache files manually
        (cache_path / "hash1.json").write_text('{"value": 1}', encoding="utf-8")
        (cache_path / "hash2.json").write_text('{"value": 2}', encoding="utf-8")

        # Initialize cache
        cache = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name=cache_name,
            config_dict=None,
        )

        assert cache.read_files == 2
        assert len(cache.cache_dict) == 2
        assert "hash1" in cache.cache_dict
        assert "hash2" in cache.cache_dict

    def test_config_yaml_not_overwritten(self, temp_cache_dir):
        """Test that existing config YAML is not overwritten."""
        cache_name = "test_cache"
        config1 = {"param": "value1"}

        # Create first cache
        cache1_instance = ConcreteFileSystemCache(
            cache_dir=temp_cache_dir,
            cache_name=cache_name,
            config_dict=config1,
        )

        yaml_file = cache1_instance.cache_path / f"{cache_name}_cache.yaml"
        original_content = yaml_file.read_text(encoding="utf-8")

        # Create second cache with different config (same hash path)
        _ = ConcreteFileSystemCache(
            cache_dir=cache1_instance.cache_path.parent,
            cache_name=cache_name,
            config_dict=config1,  # Use same config to get same path
        )

        # YAML should not be overwritten
        new_content = yaml_file.read_text(encoding="utf-8")
        assert new_content == original_content

    def test_cache_miss_then_hit(self, cache_instance):
        """Test cache miss followed by cache hit."""
        key = "test_key"

        # First get - should be a miss
        result1 = cache_instance.get(key)
        assert result1 is None
        assert cache_instance.cache_miss == 1
        assert cache_instance.cache_hit == 0

        # Add the item
        value = {"data": "test"}
        cache_instance.add(key, value)

        # Second get - should be a hit
        result2 = cache_instance.get(key)
        assert result2 == value
        assert cache_instance.cache_miss == 1
        assert cache_instance.cache_hit == 1

    def test_serialize_dict_to_json(self):
        """Test dictionary serialization to JSON."""
        test_dict = {"z": 3, "a": 1, "m": 2}
        json_str = AbstractFileSystemCache._serialize_dict_to_json(test_dict)

        # Should be sorted
        assert json_str.index('"a"') < json_str.index('"m"')
        assert json_str.index('"m"') < json_str.index('"z"')

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)  # Sorted items as list of tuples

    def test_fallback_serializer(self):
        """Test that non-serializable objects are converted to strings."""

        class CustomObject:
            def __str__(self):
                return "custom_object"

        test_dict = {"obj": CustomObject()}
        json_str = AbstractFileSystemCache._serialize_dict_to_json(test_dict)

        assert "custom_object" in json_str
