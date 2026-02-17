import uuid

import mlflow
import pandas as pd

from ragbench.api.inference import InferencePipeline
from ragbench.api.ingest import IngestPipeline
from ragbench.datasets_loader import RagDataLoader


class Experiment:
    def __init__(
        self,
        name: str,
        data_loader: RagDataLoader,
        ingest_pipeline: IngestPipeline,
        inference_pipeline: InferencePipeline,
    ):
        self.name = name
        self.data_loader = data_loader
        self.ingest_pipeline = ingest_pipeline
        self.inference_pipeline = inference_pipeline

    def run(self):
        experiment_id = f"{self.name}/{str(uuid.uuid4())}"
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(experiment_id)

        # prepare the data
        rag_benchmark = self.data_loader.get_benchmark()
        # rag_benchmark_df = pd.DataFrame(
        #     entry.model_dump() for entry in rag_benchmark.get_benchmark_entries()
        # )
        # dataset = mlflow.data.from_pandas(
        #     rag_benchmark_df,
        #     source="my source",
        #     name=self.data_loader.dataset_name,
        #     targets=None,
        # )

        # with mlflow.start_run():
        #     mlflow.log_param("experiment.name", self.name)
        #     mlflow.log_param("experiment.id", experiment_id)
        #
        #     # log the dataset
        #     mlflow.log_input(dataset, context="inference")
        #
        #     # log the dataset properties
        #     mlflow.log_param("dataset.name", self.data_loader.dataset_name)
        #     mlflow.log_param("dataset.split", self.data_loader.split)
        #     mlflow.log_param(
        #         "dataset.question_limit",
        #         self.data_loader.sampling_params.question_limit,
        #     )
        #     mlflow.log_param(
        #         "dataset.document_factor",
        #         self.data_loader.sampling_params.document_factor,
        #     )

        # run the ingest
        ingest_artifacts = self.ingest_pipeline.process(data_loader=self.data_loader)

        # set the ingest artifacts for the inference pipeline
        self.inference_pipeline.set_ingest_artifacts(ingest_artifacts=ingest_artifacts)

        # ml-flow wrapper for the inference part
        # def model_fn(benchmark_df: pd.DataFrame) -> pd.DataFrame:
        results = []

        # iterate the benchmark dataframe
        for entry in rag_benchmark.get_benchmark_entries():
            # back to python classes
            # entry = RagBenchmarkEntry.model_validate(row)

            # run the inference
            result = self.inference_pipeline.process(benchmark_entry=entry)

            # collect the result
            results.append(result.model_dump())
        # Results contain a list of inference_results

        # TODO : We must apply the metrics
        # Look at evaluator.py from RagWorkbench - we want to run unitxt
        # Look at board_model.py from RagWorkbench -

        # We can return a kind of PatternResult class

        # return a new dataframe
        return pd.DataFrame(results)

        # # ml-flow wrapper for evaluation
        # def multi_task_evaluator(eval_df, predictions, targets):
        #     return {
        #         "mae_word_count": 0,
        #         "sentiment_accuracy": 0,
        #         "sentence_split_accuracy": 0,
        #     }
        #
        # mlflow.evaluate(
        #     model=model_fn,
        #     data=rag_benchmark_df,
        #     targets=None,  # we have multiple targets → so we handle them in evaluator
        #     model_type="text",
        #     evaluators=[multi_task_evaluator],
        # )

        # print("Run:", mlflow.active_run().info.run_id)
