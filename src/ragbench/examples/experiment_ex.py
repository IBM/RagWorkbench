from typing import List

from ragbench.api.experiment import Experiment, IngestPipeline, IngestRuntimeParams, IngestArtifact, \
    InferencePipeline, InferenceRuntimeParams, InferenceParams, IngestParams
from ragbench.datasets_loader import HotpotQaDataLoader
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus


class EmptyIngestPipeline(IngestPipeline):

    def process(self, rag_corpus: RagCorpus, runtime_params: IngestRuntimeParams) -> List[IngestArtifact]:
        pass

class EmptyInferencePipeline(InferencePipeline):

    def process(self, rag_benchmark: RagBenchmark, ingest_artifacts: List[IngestArtifact],
                runtime_params: InferenceRuntimeParams) -> List[IngestArtifact]:
        pass

def test_empty_flow():
    ingest_params = IngestParams()
    ingest_pipeline = EmptyIngestPipeline(
        ingest_params=ingest_params)

    inference_params = InferenceParams()
    inference_pipeline = EmptyInferencePipeline(
        inference_params=inference_params
    )

    print("Loading dataset")
    data_loader = HotpotQaDataLoader()
    print("Done loading dataset")

    experiment = Experiment(
        data_loader=data_loader,
        ingest_pipeline=ingest_pipeline,
        inference_pipeline=inference_pipeline)

    ingest_runtime_params = IngestRuntimeParams()
    inference_runtime_params = InferenceRuntimeParams()

    experiment.run(
        ingest_runtime_params=ingest_runtime_params,
        inference_runtime_params=inference_runtime_params
    )

if __name__ == '__main__':
    test_empty_flow()
