import json
from pathlib import Path
from typing import Any

from ragbench.caching.abstract_file_system_cache import AbstractFileSystemCache


class GenerationCache(AbstractFileSystemCache):
    def __init__(
        self,
        cache_dir: Path | str,
        config_dict: dict[str, Any],
    ):
        super().__init__(
            cache_dir,
            "generation",
            config_dict=config_dict,
        )

    def _read_content(self, file: Path) -> dict[str, Any]:
        cached_generation: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
        return cached_generation

    def _content_to_json(self, generation_results: dict[str, Any]) -> str:
        return json.dumps(generation_results, indent=4)

    def _get_parameters_hash(self, query: str | list[dict[str, Any]]) -> str:
        return AbstractFileSystemCache.get_hash_list([query])

    # We force signature
    def get(self, query: str | list[dict[str, Any]]):
        cached_value, cache_key = super()._get(query)
        return cached_value

    # We force signature
    def add(
        self,
        query: str | list[dict[str, Any]],
        generation_results: dict[str, Any],
    ):
        super().add(query, generation_results)
