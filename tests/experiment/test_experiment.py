import pytest

from ragworkbench.api.inference import InferencePipeline, InferenceRuntimeParams
from ragworkbench.api.ingest import IngestPipeline, IngestRuntimeParams
from ragworkbench.api.ingest_artifact import IngestArtifact
from ragworkbench.datasets_loader import DataLoaderFactory
from ragworkbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragworkbench.datasets_loader.data_models.rag_corpus import RagCorpus
from ragworkbench.datasets_loader.dataset_names import DatasetName
from ragworkbench.experiment import Experiment


class EmptyIngestPipeline(IngestPipeline):
    def process(
        self,
        dataset_name: DatasetName | str,
        rag_corpus: RagCorpus,
        runtime_params: IngestRuntimeParams,
    ) -> list[IngestArtifact]:
        pass


class EmptyInferencePipeline(InferencePipeline):
    def process(
        self,
        dataset_name: DatasetName | str,
        rag_benchmark: RagBenchmark,
        ingest_artifacts: list[IngestArtifact],
        runtime_params: InferenceRuntimeParams,
    ):
        pass


@pytest.mark.skip(reason="Test takes >1 second (31.25s) and fails")
def test_empty_flow():
    data_loader = DataLoaderFactory.create_loader(dataset_name=DatasetName.HOTPOT_QA)
    ingest_pipeline = EmptyIngestPipeline()
    inference_pipeline = EmptyInferencePipeline()
    experiment = Experiment(
        data_loader=data_loader,
        ingest_pipeline=ingest_pipeline,
        inference_pipeline=inference_pipeline,
    )

    ingest_runtime_params = IngestRuntimeParams()
    inference_runtime_params = InferenceRuntimeParams()
    experiment.run(
        ingest_runtime_params=ingest_runtime_params,
        inference_runtime_params=inference_runtime_params,
    )
