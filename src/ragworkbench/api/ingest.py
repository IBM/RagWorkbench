from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragworkbench import RagDataLoader
from ragworkbench.api.ingest_artifact import IngestArtifact


class IngestParams(BaseModel):
    tracking_api_key: str | None = None


class IngestRuntimeParams(BaseModel):
    pass


class IngestPipeline(ABC):

    def __init__(self, _params: IngestParams) -> None:
        """
        Initialize the ingest pipeline.

        Args:
            _params: Ingest parameters for the pipeline.
        """
        self._params = _params

    @abstractmethod
    def process(
        self,
        data_loader: RagDataLoader,
    ) -> list[IngestArtifact]:
        pass
