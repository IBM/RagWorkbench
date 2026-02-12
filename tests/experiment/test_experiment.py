from ragbench.api.inference import InferencePipeline, InferenceRuntimeParams
from ragbench.api.ingest import IngestPipeline, IngestRuntimeParams
from ragbench.api.ingest_artifact import IngestArtifact
from ragbench.datasets_loader import DataLoaderFactory
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus
from ragbench.datasets_loader.dataset_names import DatasetName
from ragbench.experiment import Experiment


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
