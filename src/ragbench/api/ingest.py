from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragbench.api.ingest_artifact import IngestArtifact
from ragbench.datasets_loader.data_models.dataset_names import DatasetName
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus


class IngestRuntimeParams(BaseModel):
    pass


class IngestPipeline(ABC):
    @abstractmethod
    def process(
        self,
        dataset_name: DatasetName,
        rag_corpus: RagCorpus,
        runtime_params: IngestRuntimeParams,
    ) -> list[IngestArtifact]:
        pass
