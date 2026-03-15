from pathlib import Path
from typing import Any

from ragbench.api.inference import InferencePipeline
from ragbench.api.inference_result import InferenceResult
from ragbench.api.ingest import IngestPipeline
from ragbench.datasets_loader import RagDataLoader
from ragbench.datasets_loader.data_models import RagBenchmark
from ragbench.eval import MetricDefinition
from ragbench.eval.evaluator import Evaluator


class Experiment:
    def __init__(
        self,
        name: str,
        data_loader: RagDataLoader,
        ingest_pipeline: IngestPipeline,
        inference_pipeline: InferencePipeline,
        eval_metrics: list[MetricDefinition],
        cache_dir: Path | None = None,
    ):
        self.name = name
        self.data_loader = data_loader
        self.ingest_pipeline = ingest_pipeline
        self.inference_pipeline = inference_pipeline
        self.cache_dir = cache_dir

        self.metric_definitions: list[MetricDefinition] = eval_metrics

    def run(self) -> tuple[list[InferenceResult], dict[str, dict[str, Any]]]:
        """
        Run the complete experiment: ingest, inference, and evaluation.

        Returns:
            A tuple containing:
            - List of inference results
            - Dictionary mapping metric IDs to their evaluation results, where each
              evaluation result contains 'per_question' scores and aggregate 'statistics'
        """
        # prepare the data
        rag_benchmark: RagBenchmark = self.data_loader.get_benchmark()

        # run ingest
        ingest_artifacts = self.ingest_pipeline.process(data_loader=self.data_loader)

        # set the ingest artifacts for the inference pipeline
        self.inference_pipeline.set_ingest_artifacts(ingest_artifacts=ingest_artifacts)

        results: list[InferenceResult] = []
        for benchmark_entry in rag_benchmark.get_benchmark_entries():
            # run the inference
            result: InferenceResult = self.inference_pipeline.process(
                benchmark_entry=benchmark_entry,
            )
            # collect the result
            results.append(result)

        # Now run the evaluation via the evaluator code!
        # Run evaluation for each metric
        evaluation_results: dict[str, dict[str, Any]] = {}
        for metric_def in self.metric_definitions:
            # Create evaluator for this metric
            evaluator = Evaluator(
                metric_definition=metric_def,
                rag_benchmark=rag_benchmark,
                rag_corpus=self.data_loader.get_corpus(),
                cache_dir=self.cache_dir,
            )

            # Run metrics and get per-question scores
            question_scores: dict[str, dict[str, float]] = evaluator.run_metrics(
                results
            )

            # Compute aggregate statistics
            metric_stats = evaluator.compute_stats_from_per_question_results(
                question_scores
            )

            # Store results
            evaluation_results[metric_def.metric_id] = {
                "per_question": question_scores,
                "statistics": metric_stats,
            }

        return results, evaluation_results
