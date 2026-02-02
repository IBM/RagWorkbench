from typing import List

from pydantic import BaseModel

from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus

class DatasetId:
    pass

class DataLoader:

    def load_benchmark(self) -> RagBenchmark:
        return RagBenchmark()

    def load_corpus(self) -> RagCorpus:
        return RagCorpus()

class DataStoreDescription:
    pass

class IngestParams(BaseModel):
    pass

class IngestRuntimeParams(BaseModel):
    pass

class IngestPipeline:

    def __init__(self, ingest_params: IngestParams):
        self.ingest_params = ingest_params

    def process(self, dataset_id: DatasetId, rag_corpus: RagCorpus, runtime_params: IngestRuntimeParams) -> List[DataStoreDescription]:
        pass


class InferenceParams(BaseModel):
    pass

class InferenceRuntimeParams(BaseModel):
    pass

class InferencePipeline:

    def __init__(self, inference_params: InferenceParams):
        self.inference_params = inference_params

    def process(self, dataset_id: DatasetId,
                rag_benchmark: RagBenchmark,
                data_store_descriptions: List[DataStoreDescription],
                runtime_params: InferenceRuntimeParams) -> List[DataStoreDescription]:
        pass


class Experiment:

    def __init__(self,
                 dataset_id: DatasetId,
                 data_loader: DataLoader,
                 ingest_pipeline: IngestPipeline,
                 inference_pipeline: InferencePipeline):

        self.dataset_id = dataset_id
        self.data_loader = data_loader
        self.ingest_pipeline = ingest_pipeline
        self.inference_pipeline = inference_pipeline


    def run(self, ingest_runtime_params: IngestRuntimeParams, inference_runtime_params: InferenceRuntimeParams):
        rag_benchmark = self.data_loader.load_benchmark()
        rag_corpus = self.data_loader.load_corpus()

        data_tore_descriptions = self.ingest_pipeline.process(
            dataset_id=self.dataset_id,
            rag_corpus=rag_corpus,
            runtime_params=ingest_runtime_params)

        results = self.inference_pipeline.process(
            dataset_id=self.dataset_id,
            rag_benchmark=rag_benchmark,
            data_store_descriptions=data_tore_descriptions,
            runtime_params=inference_runtime_params)


