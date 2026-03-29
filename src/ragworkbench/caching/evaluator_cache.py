import json
from pathlib import Path
from typing import Any

from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.boards.board_model import CacheMode
from ragworkbench.caching.abstract_file_system_cache import AbstractFileSystemCache


class EvaluatorCache(AbstractFileSystemCache):
    def __init__(
        self,
        cache_dir: Path | str,
        config_params: dict,
        cache_key_fields: set | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ):
        super().__init__(
            cache_dir,
            cache_name="evaluator",
            config_dict=config_params,
            cache_mode=cache_mode,
        )
        self.cache_key_fields = cache_key_fields
        if self.cache_key_fields is None or len(self.cache_key_fields) == 0:
            # By default, we look at all the fields...
            self.cache_key_fields = {
                "question",
                "answer",
                "ground_truths",
                "ground_truths_context_ids",
                "context_ids",
                "contexts",
            }

    def _get_parameters_hash(self, reduced_dict: dict[str, Any]) -> str:
        return AbstractFileSystemCache.get_hash_dict(reduced_dict)

    def _read_content(self, file: Path) -> Any:
        return json.loads(file.read_text(encoding="utf-8"))

    def _content_to_json(self, scores_dict: dict[str, float]) -> str:
        return json.dumps(scores_dict, indent=4)

    def _create_key_dict(self, inference_result: InferenceResult) -> dict[str, Any]:
        assert self.cache_key_fields is not None, "cache_key_fields is None"
        evaluation_dict = inference_result.model_dump()
        missing_keys = [
            key for key in self.cache_key_fields if key not in evaluation_dict.keys()
        ]
        if missing_keys:
            raise RuntimeError(
                f"Missing fields in evaluation dict: missing '{missing_keys}', expected '{self.cache_key_fields}', existing: {evaluation_dict.keys()}."
            )
        reduced_dict = {
            k: v for k, v in evaluation_dict.items() if k in self.cache_key_fields
        }

        return reduced_dict

    # We force signature
    def add(self, inference_result: InferenceResult, score_dict: dict[str, float]):
        key_dict = self._create_key_dict(inference_result)
        # For debugging, we keep also the parameters that generated the entry in the cache:
        cache_value = key_dict | score_dict
        super().add(key_dict, cache_value)

    # We force signature
    def get(self, inference_result: InferenceResult):
        key_dict = self._create_key_dict(inference_result)
        cached_value, _ = super()._get(key_dict)
        # We remove the key_dict for the cached returned values:
        if cached_value:
            cached_value = {
                k: v for k, v in cached_value.items() if k not in key_dict.keys()
            }
        return cached_value
