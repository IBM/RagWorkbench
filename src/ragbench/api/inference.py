from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragbench.api.inference_result import InferenceResult
from ragbench.api.ingest_artifact import IngestArtifact
from ragbench.datasets_loader.data_models import RagBenchmarkEntry


class InferenceParams(BaseModel):
    pass


class InferenceRuntimeParams(BaseModel):
    pass


class InferencePipeline(ABC):

    @abstractmethod
    def __init__(self, _params: InferenceParams) -> None:
        pass

    @abstractmethod
    def set_ingest_artifacts(self, ingest_artifacts: list[IngestArtifact]) -> None:
        pass

    @abstractmethod
    def process(self, benchmark_entry: RagBenchmarkEntry) -> InferenceResult:
        pass
