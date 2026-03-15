# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any

from ragworkbench.api.inference import InferencePipeline
from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.api.ingest import IngestPipeline
from ragworkbench.datasets_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models import RagBenchmark
from ragworkbench.eval import MetricDefinition
from ragworkbench.eval.evaluator import Evaluator


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

        # ml-flow wrapper for the inference part
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
