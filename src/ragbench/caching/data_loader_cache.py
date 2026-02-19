from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from ragbench.caching.abstract_file_system_cache import AbstractFileSystemCache
from ragbench.datasets_loader.data_models import DocumentObject, RagBenchmark, RagCorpus


class DataLoaderCache(AbstractFileSystemCache):
    def __init__(
        self,
        cache_dir: Path | str,
        dataset_config_dict: dict,
    ):
        super().__init__(
            cache_dir,
            "data_loader",
            config_dict={"dataset": dataset_config_dict},
        )

    def _get_parameters_hash(self, constant_key: str) -> str:
        return constant_key

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

    def _read_content(self, file: Path) -> RagCorpus | RagBenchmark:
        if "rag_corpus" in file.name:
            return DataLoaderCache._load_rag_corpus_from_json(file)
        elif "rag_benchmark" in file.name:
            return RagBenchmark.model_validate_json(file.read_text(encoding="utf-8"))
        raise Exception(f"Got an unexpected file {file}")

    def _content_to_json(self, obj: RagCorpus | RagBenchmark) -> str:
        if isinstance(obj, RagCorpus):
            return self._save_rag_corpus_to_json(rag_corpus=obj)
        return obj.model_dump_json(indent=4)

    # We force signature
    def add(self, rag_benchmark: RagBenchmark, rag_corpus: RagCorpus):
        super().add("rag_corpus", rag_corpus)
        super().add("rag_benchmark", rag_benchmark)

    # We force signature
    def get(self, item: Any | None) -> tuple[RagCorpus, RagBenchmark]:  # type: ignore[unused-ignore]
        cached_corpus, corpus_key = super()._get("rag_corpus")
        cached_benchmark, benchmark_key = super()._get("rag_benchmark")
        return cached_corpus, cached_benchmark  # type: ignore[return-value]
