from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from ragbench.api.inference_result import InferenceResult
from ragbench.api.ingest_artifact import IngestArtifact
from ragbench.datasets_loader.data_models import RagBenchmarkEntry


class InferenceParams(BaseModel):
    pass


class InferenceRuntimeParams(BaseModel):
    pass


T = TypeVar("T", bound=InferenceParams)


class InferencePipeline(ABC, Generic[T]):

    def __init__(self, params: T):
        self.params = params

    @abstractmethod
    def set_ingest_artifacts(self, ingest_artifacts: list[IngestArtifact]):
        pass

    @abstractmethod
    def process(self, benchmark_entry: RagBenchmarkEntry) -> InferenceResult:
        pass
