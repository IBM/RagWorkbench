# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.api.ingest_artifact import IngestArtifact
from ragworkbench.datasets_loader.data_models import RagBenchmarkEntry

if TYPE_CHECKING:
    from ragworkbench.caching.generation_cache import GenerationCache


class InferenceParams(BaseModel):
    pass


class InferenceRuntimeParams(BaseModel):
    pass


class InferencePipeline(ABC):

    def __init__(
        self,
        _params: InferenceParams,
        cache_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize the inference pipeline.

        Args:
            _params: Inference parameters for the pipeline.
            cache_dir: Optional directory for caching generation results.
                      If provided, a GenerationCache will be created.
        """
        self._params = _params
        self.generation_cache: GenerationCache | None = None

        if cache_dir is not None:
            # Import here to avoid circular import
            from ragworkbench.caching.generation_cache import GenerationCache

            self.generation_cache = GenerationCache(
                cache_dir=cache_dir,
                inference_params=_params,
            )

    @abstractmethod
    def set_ingest_artifacts(self, ingest_artifacts: list[IngestArtifact]) -> None:
        pass

    def process(self, benchmark_entry: RagBenchmarkEntry) -> InferenceResult:
        """
        Process a benchmark entry and return the inference result.

        If a cache is configured, this method will:
        1. Check if the result exists in cache and return it if found
        2. Otherwise, call process_no_cache to generate the result
        3. Save the result to cache before returning

        If no cache is configured, it directly calls process_no_cache.

        Args:
            benchmark_entry: The benchmark entry to process.

        Returns:
            The inference result.
        """
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
