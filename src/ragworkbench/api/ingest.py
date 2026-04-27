from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel

from ragworkbench import RagDataLoader
from ragworkbench.api.ingest_artifact import IngestArtifact
from ragworkbench.boards.board_model import CacheMode


class IngestParams(BaseModel):
    tracking_api_key: str | None = None


class IngestRuntimeParams(BaseModel):
    pass


class IngestPipeline(ABC):

    def __init__(
        self,
        _params: IngestParams,
        cache_dir: Path | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ) -> None:
        """
        Initialize the ingest pipeline.

        Args:
            _params: Ingest parameters for the pipeline.
            cache_dir: Optional cache directory for pipeline-specific caching.
            cache_mode: Cache operation mode (ON/OFF/REFRESH).
        """
        self._params = _params
        self._cache_dir = cache_dir
        self._cache_mode = cache_mode

    @abstractmethod
    def process(
        self,
        data_loader: RagDataLoader,
    ) -> list[IngestArtifact]:
        pass
