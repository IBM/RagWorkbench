import json
from pathlib import Path
from typing import Any

from ragworkbench.api.inference import InferenceParams
from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.boards.board_model import CacheMode
from ragworkbench.caching.abstract_file_system_cache import AbstractFileSystemCache
from ragworkbench.datasets_loader.data_models import RagBenchmarkEntry


class GenerationCache(AbstractFileSystemCache):
    def __init__(
        self,
        cache_dir: Path | str,
        inference_params: InferenceParams,
        additional_cache_params: dict[str, Any] | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ):
        """
        Initialize generation cache.

        Args:
            cache_dir: Base directory for the cache
            inference_params: Inference parameters to include in cache key
            additional_cache_params: Optional additional parameters (e.g., index_name)
                                    to include in the cache directory hash
            cache_mode: Cache operation mode (on/off/refresh)
        """
        # Exclude tracking_api_key from cache key as it's only for cost tracking
        config_dict = inference_params.model_dump(exclude={"tracking_api_key"})

        # Merge additional parameters into config_dict if provided
        if additional_cache_params:
            config_dict.update(additional_cache_params)

        super().__init__(
            cache_dir,
            "generation",
            config_dict=config_dict,
            cache_mode=cache_mode,
        )

    def _read_content(self, file: Path) -> InferenceResult:
        cached_generation: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
        return InferenceResult(**cached_generation)

    def _content_to_json(self, inference_result: InferenceResult) -> str:
        return json.dumps(inference_result.model_dump(), indent=4)

    def _get_parameters_hash(self, benchmark_entry: RagBenchmarkEntry) -> str:
        return AbstractFileSystemCache.get_hash_dict(benchmark_entry.model_dump())

    # We force signature
    def get(self, benchmark_entry: RagBenchmarkEntry) -> InferenceResult | None:
        cached_value, cache_key = super()._get(benchmark_entry)
        return cached_value

    # We force signature
    def add(
        self,
        benchmark_entry: RagBenchmarkEntry,
        inference_results: InferenceResult,
    ):
        super().add(benchmark_entry, inference_results)
