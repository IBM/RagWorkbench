from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragworkbench import RagDataLoader
from ragworkbench.api.ingest_artifact import IngestArtifact


class IngestParams(BaseModel):
    pass


class IngestRuntimeParams(BaseModel):
    pass


class IngestPipeline(ABC):

    @abstractmethod
    def __init__(self, _params: IngestParams) -> None:
        pass

    @abstractmethod
    def process(
        self,
        data_loader: RagDataLoader,
    ) -> list[IngestArtifact]:
        pass
