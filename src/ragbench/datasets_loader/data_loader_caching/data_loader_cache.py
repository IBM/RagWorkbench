from __future__ import annotations

import base64
import hashlib
import json
import logging
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml

from ragbench.datasets_loader.data_models import DocumentObject, RagBenchmark, RagCorpus

logger = logging.getLogger(__name__)


class DataLoaderCache:

    cache_path_to_contents: dict[Path, dict[str, Any]] = {}

    def __init__(
        self,
        cache_dir: Path | str,
        dataset_config_dict: dict,
    ):
        cache_name = "data_loader"
        config_dict = {"dataset": dataset_config_dict}
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
        cache_dict = self.cache_path_to_contents.get(self.cache_path)
        self.read_files = 0
        if cache_dict is None:
            # The cache_dict maps from the file stem (a hash of the parameter without *json suffix) to the cache object
            cache_files = []
            for f in self.cache_path.glob("*.json"):
                cache_files.append(f)
            logger.info(
                f"Loading {len(cache_files)} cache files from '{self.cache_path}'.."
            )
            self.cache_dict: dict[str, Any] = {
                f.stem: self._read_content(f) for f in cache_files
            }
            logger.info(f"Loading of {len(cache_files)} cache files is done.")
            self.cache_path_to_contents[self.cache_path] = self.cache_dict
        elif cache_dict is not None:
            self.cache_dict: dict[Path, dict[str, Any]] = cache_dict  # type: ignore[no-redef]
        self.read_files = len(self.cache_dict)

        self.cache_hit = 0
        self.cache_miss = 0

    @staticmethod
    def _get_parameters_hash(constant_key: str) -> str:
        return constant_key

    def _get(self, key) -> tuple[Any | None, Any | None]:
        cache_key = self._get_parameters_hash(key)
        return self._get_with_key(cache_key)

    @staticmethod
    def _save_rag_corpus_to_json(rag_corpus: RagCorpus) -> str:
        serialized_docs = []
        for doc in rag_corpus.documents:
            # Reads the full contents of the stream
            current_position = doc.stream.tell()
            doc.stream.seek(0)  # Ensure we're at the beginning
            stream_bytes = doc.stream.read()
            stream_base64 = base64.b64encode(stream_bytes).decode("utf-8")

            serialized_docs.append(
                {
                    "name": doc.name,
                    "mime_type": doc.mime_type,
                    "metadata": doc.metadata,
                    "stream": stream_base64,
                }
            )
            # restore current position to before the write,
            # so when the stream is retrieved from the cache, its in the same position.
            doc.stream.seek(current_position)

        output: dict[str, Any] = {
            "documents": serialized_docs,
        }
        return json.dumps(output, indent=2, ensure_ascii=False)

    @staticmethod
    def _load_rag_corpus_from_json(file_path: Path) -> RagCorpus:
        """
        Loads a RagCorpus from a JSON file.

        Expects the JSON to contain:
        - "corpus_metadata": dict
        - "documents": list of dicts with:
            - "name": str
            - "mime_type": str
            - "metadata": dict
            - "stream": base64-encoded string
        """
        json_data = json.loads(file_path.read_text(encoding="utf-8"))

        # Load documents
        documents: list[DocumentObject] = []
        for doc in json_data["documents"]:
            stream_bytes = base64.b64decode(doc["stream"])
            stream = BytesIO(stream_bytes)

            document = DocumentObject(
                name=doc["name"],
                mime_type=doc["mime_type"],
                metadata=doc["metadata"],
                stream=stream,
            )
            documents.append(document)

        return RagCorpus(documents=documents)

    @staticmethod
    def _read_content(file: Path) -> RagCorpus | RagBenchmark:
        if "rag_corpus" in file.name:
            return DataLoaderCache._load_rag_corpus_from_json(file)
        elif "rag_benchmark" in file.name:
            return RagBenchmark.model_validate_json(file.read_text(encoding="utf-8"))
        raise Exception(f"Got an unexpected file {file}")

    def _content_to_json(self, obj: RagCorpus | RagBenchmark) -> str:
        if isinstance(obj, RagCorpus):
            return self._save_rag_corpus_to_json(rag_corpus=obj)
        return obj.model_dump_json(indent=4)

    def _get_with_key(self, cache_key: str) -> tuple[Any | None, str]:
        cached_value = self.cache_dict.get(cache_key, None)
        result = None
        if cached_value:
            self.cache_hit += 1
            result = deepcopy(cached_value)
        else:
            self.cache_miss += 1
        return result, cache_key

    def _format_cache_file_path(self, cache_key: str) -> Path:
        return self.cache_path / f"{cache_key}.json"

    # We force signature
    def add(self, rag_benchmark: RagBenchmark, rag_corpus: RagCorpus):

        for key, value in [
            ("rag_corpus", rag_corpus),
            ("rag_benchmark", rag_benchmark),
        ]:
            cache_key = self._get_parameters_hash(key)
            cache_file_path = self._format_cache_file_path(cache_key)
            self._add(cache_file_path, cached_item=value)

    def get(self):
        cached_corpus, corpus_key = self._get("rag_corpus")
        cached_benchmark, benchmark_key = self._get("rag_benchmark")
        return cached_corpus, cached_benchmark

    def _add_with_key(self, cache_key, cached_item):
        cache_file_path = self._format_cache_file_path(cache_key)
        self._add(cache_file_path, cached_item=cached_item)

    def _add(self, cache_file_path: Path, cached_item):
        copied_cached_item = deepcopy(cached_item)
        # Write a json representation of the object
        cache_file_path.write_text(
            self._content_to_json(copied_cached_item), encoding="utf-8"
        )
        # Update the dictionary with the filename stem (equivalent to hash_params)
        self.cache_dict[cache_file_path.stem] = copied_cached_item

    @staticmethod
    def get_hash_from_buffer(data: bytes) -> str:
        hash_object = hashlib.md5()
        hash_object.update(data)
        return hash_object.hexdigest()

    @staticmethod
    def get_hash_string(s: str) -> str:
        return DataLoaderCache.get_hash_from_buffer(s.encode("utf-8"))

    @staticmethod
    def get_hash_dict(d: dict) -> str:
        s = DataLoaderCache._serialize_dict_to_json(d)
        return DataLoaderCache.get_hash_string(s)

    @staticmethod
    def _serialize_dict_to_json(d: dict):

        def fallback_serializer(obj):
            return str(obj)

        sorted_dict_items = sorted(d.items())
        s = json.dumps(sorted_dict_items, default=fallback_serializer)
        return s
