from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.api.ingest_artifact import IngestArtifact
from ragworkbench.boards.board_model import CacheMode
from ragworkbench.datasets_loader.data_models import RagBenchmarkEntry

if TYPE_CHECKING:
    from ragworkbench.caching.generation_cache import GenerationCache


class InferenceParams(BaseModel):
    tracking_api_key: str | None = None


class InferenceRuntimeParams(BaseModel):
    pass


class InferencePipeline(ABC):

    def __init__(
        self,
        _params: InferenceParams,
        cache_dir: Path | str | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ) -> None:
        """
        Initialize the inference pipeline.

        Args:
            _params: Inference parameters for the pipeline.
            cache_dir: Optional directory for caching generation results.
                      If provided, a GenerationCache will be created.
            cache_mode: Cache operation mode (on/off/refresh).
        """
        self._params = _params
        self.generation_cache: GenerationCache | None = None
        self._cache_dir = cache_dir
        self._cache_mode = cache_mode

    @abstractmethod
    def set_ingest_artifacts(self, ingest_artifacts: list[IngestArtifact]) -> None:
        pass

    def _get_additional_cache_params(self) -> dict[str, Any] | None:
        """
        Get additional parameters to include in the cache key.

        Subclasses can override this method to add extra parameters
        (like index_name) to the cache directory hash.

        Returns:
            Dictionary of additional parameters or None
        """
        return None

    def _lazy_cache_initialization(self) -> None:
        """
        Lazily initialize the generation cache.
        """
        if (
            self._cache_dir is not None
            and self.generation_cache is None
            and self._cache_mode != CacheMode.OFF
        ):
            # Import here to avoid circular import
            from ragworkbench.caching.generation_cache import GenerationCache

            additional_params = self._get_additional_cache_params()
            self.generation_cache = GenerationCache(
                cache_dir=self._cache_dir,
                inference_params=self._params,
                additional_cache_params=additional_params,
                cache_mode=self._cache_mode,
            )

    def process(self, benchmark_entry: RagBenchmarkEntry) -> InferenceResult:
        """
        Process a benchmark entry and return the inference result.

        If a cache is configured, this method will:
        1. Initialize cache with additional parameters if needed
        2. Check if the result exists in cache and return it if found
        3. Otherwise, call process_no_cache to generate the result
        4. Save the result to cache before returning

        If no cache is configured, it directly calls process_no_cache.

        Args:
            benchmark_entry: The benchmark entry to process.

        Returns:
            The inference result.
        """
        # Lazily initialize cache with all parameters
        self._lazy_cache_initialization()

        # If cache exists, try to get cached result
        if self.generation_cache is not None:
            cached_result = self.generation_cache.get(benchmark_entry)
            if cached_result is not None:
                return cached_result

        # Generate result using the abstract method
        result = self.process_no_cache(benchmark_entry)

        # Save to cache if cache exists
        if self.generation_cache is not None:
            self.generation_cache.add(benchmark_entry, result)

        return result

    @abstractmethod
    def process_no_cache(self, benchmark_entry: RagBenchmarkEntry) -> InferenceResult:
        """
        Process a benchmark entry without using cache.

        This method must be implemented by subclasses to define the actual
        inference logic.

        Args:
            benchmark_entry: The benchmark entry to process.

        Returns:
            The inference result.
        """
        pass
