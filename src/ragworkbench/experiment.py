import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ragworkbench.api.experiment_result import ExperimentResult
from ragworkbench.api.inference import InferencePipeline
from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.api.ingest import IngestPipeline
from ragworkbench.boards.board_model import ExperimentConfig
from ragworkbench.caching.generation_cache import GenerationCache
from ragworkbench.datasets_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models import RagBenchmark
from ragworkbench.eval import MetricDefinition
from ragworkbench.eval.cost_tracking import CostTracker
from ragworkbench.eval.evaluator import Evaluator

logger = logging.getLogger(__name__)


class Experiment:
    def __init__(
        self,
        experiment_id: str,
        data_loader: RagDataLoader,
        ingest_pipeline: IngestPipeline,
        inference_pipeline: InferencePipeline,
        eval_metrics: list[MetricDefinition],
        experiment_config: ExperimentConfig,
        cache_dir: Path | None = None,
    ):
        self.experiment_id = experiment_id
        self.data_loader = data_loader
        self.ingest_pipeline = ingest_pipeline
        self.inference_pipeline = inference_pipeline
        self.cache_dir = cache_dir
        self.cache_mode = experiment_config.cache

        self.metric_definitions: list[MetricDefinition] = eval_metrics

        # Initialize cost tracker from experiment config
        self.cost_tracker = CostTracker(
            enabled=experiment_config.usage_tracking,
            litellm_proxy_url=experiment_config.litellm_proxy_url,
        )

    def run(self) -> ExperimentResult:
        """
        Run the complete experiment: ingest, inference, and evaluation.

        Returns:
            ExperimentResult object containing:
            - inference_results: List of inference results
            - evaluation_results: Dictionary mapping metric names to their evaluation results
            - cost_data: Dictionary containing cost tracking data
        """
        # Generate tracking API key if cost tracking is enabled
        tracking_api_key = self.cost_tracker.generate_tracking_key()
        if tracking_api_key:
            logger.info(
                f"Cost tracking enabled with API key: {tracking_api_key[:20]}..."
            )
            # Set the tracking API key in both ingest and inference pipeline params
            self.ingest_pipeline._params.tracking_api_key = tracking_api_key
            self.inference_pipeline._params.tracking_api_key = tracking_api_key
            logger.info(
                "Tracking API key set in ingest and inference pipeline parameters"
            )

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
                logger.info(
                    f"Inference progress: {idx}/{total_entries} entries processed ({idx/total_entries*100:.1f}%)"
                )

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

        # Retrieve cost data if cost tracking is enabled
        from ragworkbench.eval.cost_tracking import UsageData

        cost_data: UsageData = UsageData()
        if self.cost_tracker.enabled:
            logger.info("Retrieving cost tracking data from LiteLLM proxy...")
            try:
                cost_data = asyncio.run(self.cost_tracker.get_usage_data())
            except RuntimeError as e:
                logger.warning(f"Failed to retrieve cost data: {e}")

        return ExperimentResult(
            experiment_id=self.experiment_id,
            inference_results=results,
            evaluation_results=evaluation_results,
            cost_data=cost_data,
        )

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
                    cache_mode=self.cache_mode,
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
            # Clean up pending tasks before closing the event loop
            # This prevents "Task was destroyed but it is pending" warnings from litellm
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    logger.info(
                        f"Cleaning up {len(pending)} pending async tasks before closing event loop"
                    )
                    # Cancel all pending tasks
                    for task in pending:
                        task.cancel()
                    # Wait for all tasks to complete cancellation with a timeout
                    try:
                        logger.info(f"Waiting for {len(pending)} pending tasks..")
                        loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=5.0,
                            )
                        )
                        logger.info("Successfully cleaned up all pending tasks")
                    except TimeoutError:
                        logger.warning(
                            f"Timeout while waiting for {len(pending)} tasks to cancel - forcing cleanup"
                        )
            except Exception as e:
                # Log any errors during cleanup but don't fail the evaluation
                logger.warning(
                    f"Error during event loop cleanup: {type(e).__name__}: {e}"
                )
            finally:
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
                logger.info(
                    f"Inference complete: No cache queries recorded - Cache path: {cache_path}"
                )
        else:
            logger.info("Inference complete: Caching disabled")
