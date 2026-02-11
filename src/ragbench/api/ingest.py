from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragbench.api.ingest_artifact import IngestArtifact
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus
from ragbench.datasets_loader.dataset_names import DatasetName


class IngestRuntimeParams(BaseModel):
    pass


class IngestPipeline(ABC):
    @abstractmethod
    def process(
        self,
        dataset_name: DatasetName | str,
        rag_corpus: RagCorpus,
        runtime_params: IngestRuntimeParams,
    ) -> list[IngestArtifact]:
        pass
