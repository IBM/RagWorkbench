import uuid
from abc import ABC, abstractmethod

import mlflow
import pandas as pd
from pydantic import BaseModel

from ragbench.datasets_loader import RagDataLoader
from ragbench.datasets_loader.data_models.dataset_names import DatasetName
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus


class IngestArtifact(BaseModel):
    pass


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


class InferenceRuntimeParams(BaseModel):
    pass


class InferencePipeline(ABC):
    def __init__(self):
        self.ingest_artifacts = None

    def set_ingest_artifact(self, ingest_artifacts: list[IngestArtifact]):
        self.ingest_artifacts = ingest_artifacts

    @abstractmethod
    def process(
        self,
        dataset_name: DatasetName,
        rag_benchmark: RagBenchmark,
        runtime_params: InferenceRuntimeParams,
    ):
        pass


class Experiment:
    def __init__(
        self,
        data_loader: RagDataLoader,
        ingest_pipeline: IngestPipeline,
        inference_pipeline: InferencePipeline,
    ):
        self.data_loader = data_loader
        self.ingest_pipeline = ingest_pipeline
        self.inference_pipeline = inference_pipeline

    def run(
        self,
        ingest_runtime_params: IngestRuntimeParams,
        inference_runtime_params: InferenceRuntimeParams,
    ):
        experiment_id = str(uuid.uuid4())
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_id)

        rag_benchmark = self.data_loader.get_benchmark()

        rag_corpus = self.data_loader.get_corpus()

        rag_benchmark_df = pd.DataFrame(
            {
                "question_id": entry.question_id,
                "question": entry.question,
                "ground-truths": entry.ground_truth_answers,
            }
            for entry in rag_benchmark.get_benchmark_entries()
        )

        dataset = mlflow.data.from_pandas(
            rag_benchmark_df,
            source="my source",
            name=self.data_loader.dataset_name,
            targets=None,
        )

        with mlflow.start_run():
            mlflow.log_param("dataset.name", self.data_loader.dataset_name)
            mlflow.log_param("dataset.split", self.data_loader.split)
            mlflow.log_param(
                "dataset.question_limit",
                self.data_loader.sampling_params.question_limit,
            )
            mlflow.log_param(
                "dataset.document_factor",
                self.data_loader.sampling_params.document_factor,
            )

            ingest_artifacts = self.ingest_pipeline.process(
                dataset_name=self.data_loader.dataset_name,
                rag_corpus=rag_corpus,
                runtime_params=ingest_runtime_params,
            )

            mlflow.log_input(dataset, context="inference")

            self.inference_pipeline.set_ingest_artifact(ingest_artifacts)

            self.inference_pipeline.process(
                dataset_name=self.data_loader.dataset_name,
                rag_benchmark=rag_benchmark,
                runtime_params=inference_runtime_params,
            )
