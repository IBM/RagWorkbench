import json
from pathlib import Path
from typing import Any

from ragworkbench.api.experiment_result import ExperimentResult
from ragworkbench.boards.board_model import CacheMode
from ragworkbench.caching.abstract_file_system_cache import AbstractFileSystemCache


class ExperimentCache(AbstractFileSystemCache):
    """
    Cache for full experiment results.

    Cache key: experiment_id
    Cache value: ExperimentResult object
    """

    def __init__(
        self,
        cache_dir: Path | str,
        cache_mode: CacheMode = CacheMode.ON,
    ):
        """
        Initialize experiment cache.

        Args:
            cache_dir: Base directory for the cache
            cache_mode: Cache operation mode (on/off/refresh)
        """
        super().__init__(
            cache_dir,
            cache_name="experiment",
            config_dict=None,
            cache_mode=cache_mode,
        )

    def _read_content(self, file: Path) -> ExperimentResult:
        """
        Read and deserialize ExperimentResult from a cache file.

        Args:
            file: Path to the cache file

        Returns:
            Deserialized ExperimentResult object
        """
        cached_experiment: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
        return ExperimentResult(**cached_experiment)

    def _content_to_json(self, *args) -> str:
        """
        Serialize ExperimentResult to JSON string.

        Args:
            *args: ExperimentResult object to serialize

        Returns:
            JSON string representation
        """
        experiment_result: ExperimentResult = args[0]
        return json.dumps(experiment_result.model_dump(), indent=4)

    def _get_parameters_hash(self, *args) -> str:
        """
        Generate hash from experiment_id.

        Args:
            *args: The experiment identifier

        Returns:
            Hash string
        """
        experiment_id: str = args[0]
        return AbstractFileSystemCache.get_hash_string(experiment_id)

    # We force signature
    def get(self, experiment_id: str) -> ExperimentResult | None:
        """
        Get an ExperimentResult from the cache by experiment_id.

        Args:
            experiment_id: The experiment identifier

        Returns:
            Cached ExperimentResult or None if not found
        """
        cached_value, _ = super()._get(experiment_id)
        return cached_value

    # We force signature
    def add(self, experiment_id: str, experiment_result: ExperimentResult):
        """
        Add an ExperimentResult to the cache.

        Args:
            experiment_id: The experiment identifier (cache key)
            experiment_result: The ExperimentResult object to cache
        """
        super().add(experiment_id, experiment_result)
