import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar

import yaml

logger = logging.getLogger(__name__)


class AbstractFileSystemCache(ABC):
    """
    Abstract base class for file system-based caching.

    This class provides a framework for caching data to disk using JSON files,
    with support for configuration-based cache directories and hash-based file naming.

    The class maintains a class-level cache to avoid reloading the same cache
    directory multiple times across different instances.
    """

    # A class level cache of cache contents intended to make sure each cache is loaded from
    # the disk just once.
    cache_path_to_contents: ClassVar[dict[Path, dict[str, Any]]] = {}

    def __init__(
        self,
        cache_dir: Path | str,
        cache_name: str,
        config_dict: dict[str, Any] | None = None,
    ):
        """
        Initialize the file system cache.

        Args:
            cache_dir: Base directory for the cache
            cache_name: Name of the cache (used as subdirectory)
            config_dict: Optional configuration dictionary that will be hashed
                        to create a unique subdirectory
        """
        cache_dir = Path(cache_dir)
        self.cache_path = cache_dir / cache_name
        if config_dict is not None:
            dir_name = self.get_hash_dict(config_dict)
            self.cache_path = self.cache_path / dir_name

        self.cache_path.mkdir(exist_ok=True, parents=True)
        cache_params_file = self.cache_path / f"{cache_name}_cache.yaml"
        if not cache_params_file.exists() and config_dict is not None:
            cache_params_file.write_text(
                yaml.dump(
                    config_dict, default_flow_style=False, sort_keys=False, indent=2
                ),
                encoding="utf-8",
            )
        cached_dict = self.cache_path_to_contents.get(self.cache_path)
        self.read_files = 0
        if cached_dict is None:
            # The cache_dict maps from the file stem (a hash of the parameter without *json suffix) to the cache object
            cache_files = list(self.cache_path.glob("*.json"))
            start_time = time.time()
            self.cache_dict: dict[str, Any] = {
                f.stem: self._read_content(f) for f in cache_files
            }
            elapsed_time = time.time() - start_time
            logger.info(
                f"Loaded in {elapsed_time:.2f}s: {len(cache_files)} cache files from '{self.cache_path}'"
            )
            self.cache_path_to_contents[self.cache_path] = self.cache_dict
            self.read_files = len(self.cache_dict)
        else:
            self.cache_dict = cached_dict
        self.cache_hit = 0
        self.cache_miss = 0

    @abstractmethod
    def _read_content(self, file: Path) -> Any:
        """
        Read and deserialize content from a cache file.

        Args:
            file: Path to the cache file

        Returns:
            Deserialized content
        """
        pass

    def _get(self, *args) -> tuple[Any | None, str]:
        """
        Get cached value using parameters to generate cache key.

        Args:
            *args: Parameters used to generate the cache key

        Returns:
            Tuple of (cached value or None, cache key)
        """
        cache_key = self._get_parameters_hash(*args)
        return self._get_with_key(cache_key)

    def _get_with_key(self, cache_key: str) -> tuple[Any | None, str]:
        """
        Get cached value using a pre-computed cache key.

        Args:
            cache_key: The cache key to lookup

        Returns:
            Tuple of (cached value or None, cache key)
        """
        cached_value = self.cache_dict.get(cache_key)
        result = None
        if cached_value is not None:
            self.cache_hit += 1
            result = deepcopy(cached_value)
        else:
            self.cache_miss += 1
        return result, cache_key

    @abstractmethod
    def _content_to_json(self, *args) -> str:
        """
        Serialize content to JSON string.

        Args:
            *args: Content to serialize

        Returns:
            JSON string representation
        """
        pass

    def _get_cache_file_path(self, *args) -> Path:
        """
        Get the cache file path for given parameters.

        Args:
            *args: Parameters to hash for the filename

        Returns:
            Path to the cache file
        """
        cache_key = self._get_parameters_hash(*args)
        return self._format_cache_file_path(cache_key)

    def _format_cache_file_path(self, cache_key: str) -> Path:
        """
        Format a cache file path from a cache key.

        Args:
            cache_key: The cache key (hash)

        Returns:
            Path to the cache file
        """
        return self.cache_path / f"{cache_key}.json"

    @abstractmethod
    def _get_parameters_hash(self, *args) -> str:
        """
        Generate a hash from parameters.

        Args:
            *args: Parameters to hash

        Returns:
            Hash string
        """
        pass

    @abstractmethod
    def get(self, item: Any) -> Any:
        """
        Get an item from the cache (public interface).

        Args:
            item: Item to retrieve

        Returns:
            Cached value
        """
        pass

    def add(self, *args):
        """
        Add an item to the cache.

        The last argument is treated as the value to cache,
        all preceding arguments are used to generate the cache key.

        Args:
            *args: Parameters followed by the value to cache (minimum 2 arguments)

        Raises:
            ValueError: If fewer than 2 arguments are provided
        """
        if len(args) < 2:
            raise ValueError(
                "add() requires at least 2 arguments: "
                "one or more key parameters and the cached item"
            )

        # Get all parameters except the last one
        key_args = args[:-1]
        # Create the file name from these "key" parameters
        cache_file_path = self._get_cache_file_path(*key_args)
        self._add(cache_file_path, cached_item=args[-1])

    def _add_with_key(self, cache_key: str, cached_item: Any):
        """
        Add an item to the cache using a pre-computed cache key.

        Args:
            cache_key: The cache key
            cached_item: The item to cache
        """
        cache_file_path = self._format_cache_file_path(cache_key)
        self._add(cache_file_path, cached_item=cached_item)

    def _add(self, cache_file_path: Path, cached_item: Any):
        """
        Internal method to add an item to the cache.

        Args:
            cache_file_path: Path where the cache file should be written
            cached_item: The item to cache
        """
        copied_cached_item = deepcopy(cached_item)
        # Write a json representation of the object
        cache_file_path.write_text(
            self._content_to_json(copied_cached_item), encoding="utf-8"
        )
        # Update the dictionary with the filename stem (equivalent to hash_params)
        self.cache_dict[cache_file_path.stem] = copied_cached_item

    def get_cache_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hits, misses, and total entries
        """
        return {
            "cache_hit": self.cache_hit,
            "cache_miss": self.cache_miss,
            "total_entries": len(self.cache_dict),
            "read_files": self.read_files,
        }

    def clear_cache(self):
        """
        Clear the in-memory cache dictionary.

        Note: This does not delete files from disk.
        """
        self.cache_dict.clear()
        if self.cache_path in self.cache_path_to_contents:
            del self.cache_path_to_contents[self.cache_path]

    @staticmethod
    def get_hash_from_buffer(data: bytes) -> str:
        """
        Generate MD5 hash from bytes.

        Args:
            data: Bytes to hash

        Returns:
            Hexadecimal hash string
        """
        hash_object = hashlib.md5()
        hash_object.update(data)
        return hash_object.hexdigest()

    @staticmethod
    def get_hash_string(s: str) -> str:
        """
        Generate MD5 hash from string.

        Args:
            s: String to hash

        Returns:
            Hexadecimal hash string
        """
        return AbstractFileSystemCache.get_hash_from_buffer(s.encode("utf-8"))

    @staticmethod
    def get_hash_dict(d: dict[str, Any]) -> str:
        """
        Generate MD5 hash from dictionary.

        The dictionary is serialized to JSON with sorted keys for consistency.

        Args:
            d: Dictionary to hash

        Returns:
            Hexadecimal hash string
        """
        s = AbstractFileSystemCache._serialize_dict_to_json(d)
        return AbstractFileSystemCache.get_hash_string(s)

    @staticmethod
    def _serialize_dict_to_json(d: dict[str, Any]) -> str:
        """
        Serialize dictionary to JSON string with sorted keys.

        Args:
            d: Dictionary to serialize

        Returns:
            JSON string
        """

        def fallback_serializer(obj):
            return str(obj)

        sorted_dict_items = sorted(d.items())
        s = json.dumps(sorted_dict_items, default=fallback_serializer)
        return s

    @staticmethod
    def get_hash_list(lst: list[Any]) -> str:
        serialized = []
        for item in lst:
            if isinstance(item, dict):
                s = AbstractFileSystemCache._serialize_dict_to_json(item)
                serialized.append(s)
            else:
                serialized.append(str(item))
        return AbstractFileSystemCache.get_hash_string(" ".join(serialized))
