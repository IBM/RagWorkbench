from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from ragbench import RagDataLoader
from ragbench.api.ingest_artifact import IngestArtifact


class IngestParams(BaseModel):
    pass


class IngestRuntimeParams(BaseModel):
    pass


T = TypeVar("T", bound=IngestParams)


class IngestPipeline(ABC, Generic[T]):

    def __init__(self, params: T):
        self.params = params

    @abstractmethod
    def process(
        self,
        data_loader: RagDataLoader,
    ) -> list[IngestArtifact]:
        pass
