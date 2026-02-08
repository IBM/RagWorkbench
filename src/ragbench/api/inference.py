from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragbench.api.ingest_artifact import IngestArtifact
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.dataset_names import DatasetName


class InferenceRuntimeParams(BaseModel):
    pass


class InferencePipeline(ABC):
    @abstractmethod
    def process(
        self,
        dataset_name: DatasetName,
        rag_benchmark: RagBenchmark,
        ingest_artifacts: list[IngestArtifact],
        runtime_params: InferenceRuntimeParams,
    ):
        pass
