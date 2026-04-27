"""Cache for Docling document objects."""

import json
import logging
from pathlib import Path
from typing import Any

from docling_core.types.doc.document import DoclingDocument

from ragworkbench.boards.board_model import CacheMode
from ragworkbench.caching.abstract_file_system_cache import AbstractFileSystemCache

logger = logging.getLogger(__name__)


class DoclingCache(AbstractFileSystemCache):
    """
    File system cache for Docling document objects.

    Caches the converted Docling documents to avoid re-processing the same
    documents through the DocumentConverter, which can be expensive.

    The cache key is based on the document name, and the cached value is
    the JSON-serialized Docling document.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        config_dict: dict[str, Any] | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ):
        """
        Initialize the Docling document cache.

        Args:
            cache_dir: Base directory for cache storage.
            config_dict: Optional configuration dictionary for cache subdirectory hashing.
            cache_mode: Cache operation mode (on/off/refresh).
        """
        super().__init__(
            cache_dir,
            "docling",
            config_dict=config_dict,
            cache_mode=cache_mode,
        )

    def _read_content(self, file: Path) -> DoclingDocument:
        """
        Read and deserialize a Docling document from a cache file.

        Args:
            file: Path to the cache file.

        Returns:
            Deserialized DoclingDocument instance.
        """
        cached_doc_dict: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
        return DoclingDocument(**cached_doc_dict)

    def _content_to_json(self, docling_doc: DoclingDocument) -> str:
        """
        Serialize a Docling document to JSON string.

        Args:
            docling_doc: The DoclingDocument to serialize.

        Returns:
            JSON string representation.
        """
        return json.dumps(docling_doc.model_dump(), indent=4)

    def _get_parameters_hash(self, document_name: str) -> str:
        """
        Generate hash from document name.

        Args:
            document_name: Name of the document.

        Returns:
            Hash string based on document name.
        """
        return AbstractFileSystemCache.get_hash_string(document_name)

    # We force signature
    def get(self, document_name: str) -> DoclingDocument | None:
        """
        Get a cached Docling document by name.

        Args:
            document_name: Name of the document to retrieve.

        Returns:
            Cached DoclingDocument if found, None otherwise.
        """
        cached_value, _ = super()._get(document_name)
        return cached_value

    # We force signature
    def add(self, document_name: str, docling_doc: DoclingDocument) -> None:
        """
        Add a Docling document to the cache.

        Args:
            document_name: Name of the document (used as cache key).
            docling_doc: The DoclingDocument to cache.
        """
        super().add(document_name, docling_doc)
