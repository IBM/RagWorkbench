import uuid

import mlflow
import pandas as pd

from ragbench.api.inference import InferencePipeline, InferenceRuntimeParams
from ragbench.api.ingest import IngestPipeline, IngestRuntimeParams
from ragbench.datasets_loader import RagDataLoader


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

            self.inference_pipeline.process(
                dataset_name=self.data_loader.dataset_name,
                rag_benchmark=rag_benchmark,
                ingest_artifacts=ingest_artifacts,
                runtime_params=inference_runtime_params,
            )
