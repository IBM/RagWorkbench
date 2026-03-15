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

import json
from pathlib import Path
from typing import Any

from ragworkbench.api.inference import InferenceParams
from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.caching.abstract_file_system_cache import AbstractFileSystemCache
from ragworkbench.datasets_loader.data_models import RagBenchmarkEntry


class GenerationCache(AbstractFileSystemCache):
    def __init__(
        self,
        cache_dir: Path | str,
        inference_params: InferenceParams,
    ):
        super().__init__(
            cache_dir,
            "generation",
            config_dict=inference_params.model_dump(),
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
