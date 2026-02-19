"""
Data loader caching implementation for RAG benchmarks and corpora.

This module provides specialized caching for RAG (Retrieval-Augmented Generation)
datasets, handling serialization of document streams and benchmark data to disk.
It extends AbstractFileSystemCache to provide type-safe caching operations for
RagCorpus and RagBenchmark objects.

Key Features:
    - Base64 encoding of document streams for JSON serialization
    - Preservation of stream positions during serialization
    - Type-safe cache operations with proper error handling
    - Automatic cache key generation based on dataset configuration

Example:
    >>> cache = DataLoaderCache(
    ...     cache_dir="./cache",
    ...     dataset_config_dict={"name": "my_dataset", "version": "1.0"}
    ... )
    >>> cache.add(benchmark, corpus)
    >>> loaded_corpus, loaded_benchmark = cache.get()
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from ragbench.caching.abstract_file_system_cache import AbstractFileSystemCache
from ragbench.datasets_loader.data_models import DocumentObject, RagBenchmark, RagCorpus


class DataLoaderCacheError(Exception):
    """Base exception for DataLoaderCache errors."""

    pass


class InvalidCacheFileError(DataLoaderCacheError):
    """Raised when cache file format is invalid or unrecognized."""

    pass


class StreamSerializationError(DataLoaderCacheError):
    """Raised when stream serialization or deserialization fails."""

    pass


class _CacheKeys:
    """Internal constants for cache file naming and JSON keys."""

    RAG_CORPUS = "rag_corpus"
    RAG_BENCHMARK = "rag_benchmark"
    DOCUMENTS = "documents"


class DataLoaderCache(AbstractFileSystemCache):
    """
    File system cache for RAG data loaders.

    Provides caching functionality for RagCorpus and RagBenchmark objects,
    with specialized handling for document streams and metadata.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        dataset_config_dict: dict,
    ):
        """
        Initialize the data loader cache.

        Args:
            cache_dir: Base directory for cache storage.
            dataset_config_dict: Configuration dictionary for the dataset,
                used to generate a unique cache subdirectory.
        """
        super().__init__(
            cache_dir,
            "data_loader",
            config_dict={"dataset": dataset_config_dict},
        )

    def _get_parameters_hash(self, *args) -> str:
        """
        Generate hash from parameters.

        For DataLoaderCache, we use constant keys (rag_corpus, rag_benchmark)
        rather than hashing, as we always store exactly two files per dataset.

        Args:
            *args: Variable arguments, expected to be a single constant key string.

        Returns:
            The constant key unchanged.
        """
        # We expect a single constant key argument
        return str(args[0]) if args else ""

    @staticmethod
    def _save_rag_corpus_to_json(rag_corpus: RagCorpus) -> str:
        """
        Serialize a RagCorpus to JSON format.

        Converts document streams to base64-encoded strings for JSON serialization.
        Stream positions are preserved during serialization.

        Args:
            rag_corpus: The RagCorpus instance to serialize.

        Returns:
            JSON string representation of the corpus.

        Raises:
            StreamSerializationError: If stream reading or encoding fails.
            ValueError: If corpus contains no documents.
        """
        if not rag_corpus.documents:
            raise ValueError("Cannot serialize empty RagCorpus")

        serialized_docs = []
        for doc in rag_corpus.documents:
            try:
                # Save current position
                current_position = doc.stream.tell()

                # Validate stream is seekable
                if not doc.stream.seekable():
                    raise StreamSerializationError(
                        f"Document '{doc.name}' has non-seekable stream"
                    )

                # Read stream content
                doc.stream.seek(0)
                stream_bytes = doc.stream.read()

                # Encode to base64
                stream_base64 = base64.b64encode(stream_bytes).decode("utf-8")

                serialized_docs.append(
                    {
                        "name": doc.name,
                        "mime_type": doc.mime_type,
                        "metadata": doc.metadata,
                        "stream": stream_base64,
                    }
                )

                # Restore original position
                doc.stream.seek(current_position)

            except OSError as e:
                raise StreamSerializationError(
                    f"Failed to read stream for document '{doc.name}': {e}"
                ) from e
            except StreamSerializationError:
                # Re-raise our own exceptions
                raise
            except Exception as e:
                raise StreamSerializationError(
                    f"Unexpected error serializing document '{doc.name}': {e}"
                ) from e

        output: dict[str, Any] = {
            _CacheKeys.DOCUMENTS: serialized_docs,
        }
        return json.dumps(output, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_rag_corpus_from_json(file_path: Path) -> RagCorpus:
        """
        Load a RagCorpus from a JSON cache file.

        Deserializes document streams from base64-encoded strings and reconstructs
        DocumentObject instances with their metadata.

        Args:
            file_path: Path to the JSON cache file.

        Returns:
            Reconstructed RagCorpus instance.

        Raises:
            InvalidCacheFileError: If file format is invalid or required fields are missing.
            StreamSerializationError: If base64 decoding fails.
        """
        try:
            json_data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise InvalidCacheFileError(
                f"Invalid JSON in cache file '{file_path}': {e}"
            ) from e
        except Exception as e:
            raise InvalidCacheFileError(
                f"Failed to read cache file '{file_path}': {e}"
            ) from e

        # Validate required fields
        if _CacheKeys.DOCUMENTS not in json_data:
            raise InvalidCacheFileError(
                f"Cache file missing required '{_CacheKeys.DOCUMENTS}' field"
            )

        documents_data = json_data[_CacheKeys.DOCUMENTS]
        if not isinstance(documents_data, list):
            raise InvalidCacheFileError(f"'{_CacheKeys.DOCUMENTS}' must be a list")

        if not documents_data:
            raise InvalidCacheFileError("Cache file contains no documents")

        # Deserialize documents
        documents: list[DocumentObject] = []
        for idx, doc_data in enumerate(documents_data):
            try:
                # Validate required fields
                required_fields = ["name", "mime_type", "metadata", "stream"]
                for field in required_fields:
                    if field not in doc_data:
                        raise InvalidCacheFileError(
                            f"Document {idx} missing required field '{field}'"
                        )

                # Decode base64 stream
                stream_bytes = base64.b64decode(doc_data["stream"])
                stream = BytesIO(stream_bytes)

                document = DocumentObject(
                    name=doc_data["name"],
                    mime_type=doc_data["mime_type"],
                    metadata=doc_data["metadata"],
                    stream=stream,
                )
                documents.append(document)

            except (ValueError, TypeError) as e:
                raise StreamSerializationError(
                    f"Failed to decode stream for document {idx}: {e}"
                ) from e
            except InvalidCacheFileError:
                # Re-raise our own exceptions
                raise
            except Exception as e:
                raise InvalidCacheFileError(
                    f"Failed to deserialize document {idx}: {e}"
                ) from e

        return RagCorpus(documents=documents)

    def _read_content(self, file: Path) -> RagCorpus | RagBenchmark:
        """
        Read and deserialize content from a cache file.

        Determines the content type based on filename and delegates to
        appropriate deserialization method.

        Args:
            file: Path to the cache file.

        Returns:
            Either a RagCorpus or RagBenchmark instance.

        Raises:
            InvalidCacheFileError: If file type cannot be determined or is invalid.
        """
        filename = file.name.lower()

        # Check for corpus file
        if _CacheKeys.RAG_CORPUS in filename:
            return DataLoaderCache._load_rag_corpus_from_json(file)

        # Check for benchmark file
        if _CacheKeys.RAG_BENCHMARK in filename:
            try:
                return RagBenchmark.model_validate_json(
                    file.read_text(encoding="utf-8")
                )
            except Exception as e:
                raise InvalidCacheFileError(
                    f"Failed to load RagBenchmark from '{file}': {e}"
                ) from e

        # Unknown file type
        raise InvalidCacheFileError(
            f"Cannot determine cache file type from filename: '{file.name}'. "
            f"Expected filename to contain '{_CacheKeys.RAG_CORPUS}' or "
            f"'{_CacheKeys.RAG_BENCHMARK}'"
        )

    def _content_to_json(self, *args) -> str:
        """
        Serialize cache content to JSON string.

        Args:
            *args: Variable arguments, expected to be a single object
                (RagCorpus or RagBenchmark) to serialize.

        Returns:
            JSON string representation.

        Raises:
            TypeError: If obj is not a supported type.
        """
        obj = args[0] if args else None
        if isinstance(obj, RagCorpus):
            return self._save_rag_corpus_to_json(rag_corpus=obj)
        elif isinstance(obj, RagBenchmark):
            return obj.model_dump_json(indent=2)
        else:
            raise TypeError(
                f"Unsupported type for serialization: {type(obj).__name__}. "
                f"Expected RagCorpus or RagBenchmark"
            )

    def add(self, *args) -> None:
        """
        Add a benchmark and corpus pair to the cache.

        Both the RagBenchmark and RagCorpus are serialized and stored as
        separate cache files with standardized naming.

        Args:
            *args: Expected to be (rag_benchmark, rag_corpus) tuple.
                rag_benchmark: The benchmark dataset to cache.
                rag_corpus: The document corpus to cache.

        Raises:
            StreamSerializationError: If corpus serialization fails.
            ValueError: If corpus is empty or wrong number of arguments.
        """
        if len(args) != 2:
            raise ValueError(
                f"add() requires exactly 2 arguments (rag_benchmark, rag_corpus), "
                f"got {len(args)}"
            )
        rag_benchmark, rag_corpus = args
        super().add(_CacheKeys.RAG_CORPUS, rag_corpus)
        super().add(_CacheKeys.RAG_BENCHMARK, rag_benchmark)

    def get(self, item: Any | None = None) -> tuple[RagCorpus, RagBenchmark]:
        """
        Retrieve cached corpus and benchmark.

        Args:
            item: Unused parameter (maintained for interface compatibility).

        Returns:
            Tuple of (RagCorpus, RagBenchmark) retrieved from cache.
            Returns None for each component if not found in cache.

        Raises:
            InvalidCacheFileError: If cached files are corrupted or invalid.
        """
        cached_corpus, _ = super()._get(_CacheKeys.RAG_CORPUS)
        cached_benchmark, _ = super()._get(_CacheKeys.RAG_BENCHMARK)
        return cached_corpus, cached_benchmark  # type: ignore[return-value]
