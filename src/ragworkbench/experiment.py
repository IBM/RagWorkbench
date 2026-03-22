import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ragworkbench.api.inference import InferencePipeline
from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.api.ingest import IngestPipeline
from ragworkbench.caching.generation_cache import GenerationCache
from ragworkbench.datasets_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models import RagBenchmark
from ragworkbench.eval import MetricDefinition
from ragworkbench.eval.evaluator import Evaluator

logger = logging.getLogger(__name__)


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
            - Dictionary mapping metric names to their evaluation results, where each
              evaluation result contains 'per_question' scores and aggregate 'statistics'
        """
        # prepare the data
        rag_benchmark: RagBenchmark = self.data_loader.get_benchmark()

        # run ingest
        ingest_artifacts = self.ingest_pipeline.process(data_loader=self.data_loader)

        # set the ingest artifacts for the inference pipeline
        self.inference_pipeline.set_ingest_artifacts(ingest_artifacts=ingest_artifacts)

        results: list[InferenceResult] = []
        benchmark_entries = rag_benchmark.get_benchmark_entries()
        total_entries = len(benchmark_entries)
        
        logger.info(f"Starting inference on {total_entries} benchmark entries")
        
        for idx, benchmark_entry in enumerate(benchmark_entries, start=1):
            # run the inference
            result: InferenceResult = self.inference_pipeline.process(
                benchmark_entry=benchmark_entry,
            )
            # collect the result
            results.append(result)
            
            # Log progress every 10 entries or at the end
            if idx % 10 == 0 or idx == total_entries:
                logger.info(f"Inference progress: {idx}/{total_entries} entries processed ({idx/total_entries*100:.1f}%)")

        # Log cache statistics before evaluation
        self._log_cache_statistics(self.inference_pipeline.generation_cache)

        # Now run the evaluation via the evaluator code!
        # Run evaluation in a separate thread to avoid event loop conflicts
        # Unitxt's inference engine uses asyncio internally with run_until_complete,
        # which cannot be nested. Running in a thread allows it to create its own event loop.
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._run_evaluation,
                results=results,
                rag_benchmark=rag_benchmark,
            )
            evaluation_results = future.result()

        return results, evaluation_results

    def _run_evaluation(
        self,
        results: list[InferenceResult],
        rag_benchmark: RagBenchmark,
    ) -> dict[str, dict[str, Any]]:
        """
        Run evaluation for all metrics.
        
        This method is executed in a separate thread to allow unitxt's inference engine
        to create and manage its own event loop without conflicts. We create a new event
        loop for this thread since threads don't have event loops by default.

        Args:
            results: List of inference results to evaluate
            rag_benchmark: The RAG benchmark containing ground truth data

        Returns:
            Dictionary mapping metric names to their evaluation results, where each
            evaluation result contains 'per_question' scores and aggregate 'statistics'
        """
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
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

                # Store results using metric_name as key
                evaluation_results[metric_def.metric_name] = {
                    "per_question": question_scores,
                    "statistics": metric_stats,
                }

            return evaluation_results
        finally:
            # Clean up the event loop
            loop.close()

    @staticmethod
    def _log_cache_statistics(generation_cache: GenerationCache | None) -> None:
        """
        Log cache hit statistics from the generation cache.
        
        Args:
            generation_cache: The GenerationCache instance to get statistics from, or None if caching is disabled.
        """
        if generation_cache is not None:
            cache_stats = generation_cache.get_cache_stats()
            cache_hits = cache_stats["cache_hit"]
            cache_misses = cache_stats["cache_miss"]
            total_queries = cache_hits + cache_misses
            cache_path = generation_cache.cache_path
            if total_queries > 0:
                hit_rate = (cache_hits / total_queries) * 100
                logger.info(
                    f"Inference complete: {cache_hits}/{total_queries} queries served from cache "
                    f"({hit_rate:.1f}% cache hit rate) - Cache path: {cache_path}"
                )
            else:
                logger.info(f"Inference complete: No cache queries recorded - Cache path: {cache_path}")
        else:
            logger.info("Inference complete: Caching disabled")
