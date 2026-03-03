import logging
from collections import defaultdict
from io import UnsupportedOperation
from pathlib import Path

from ragbench.api.inference_result import InferenceResult
from ragbench.caching.evaluator_cache import EvaluatorCache
from ragbench.datasets_loader.data_models import (
    RagBenchmark,
    RagCorpus,
)
from ragbench.eval import MetricDefinition
from ragbench.eval.evaluation import BaseEvaluationMetric
from ragbench.eval.evaluation_level import EvaluationLevel
from ragbench.eval.evaluation_result import (
    EvaluationResults,
    MetricScore,
    QuestionEvaluationResult,
)
from ragbench.eval.unitxt_evaluation import UnitxtEvaluationMetric

logger = logging.getLogger(__name__)


class Evaluator:

    evaluation_metric_factory = {
        "unitxt": UnitxtEvaluationMetric,
        # "ragas": RagasEvaluationMetric,
        # "workbench": RagWorkbenchEvaluationMetric,
    }

    def __init__(
        self,
        metric_definition: MetricDefinition,
        rag_benchmark: RagBenchmark | None = None,
        rag_corpus: RagCorpus | None = None,
        cache_dir: Path | None = None,
    ):

        self.metric: BaseEvaluationMetric = self.evaluation_metric_factory[
            metric_definition.vendor
        ](
            name=metric_definition.metric_id,
            metric_params=metric_definition.metric_params,
        )

        fields = metric_definition.metric_fields

        self.full_score_names: list[str] = [
            self.metric.full_score_name(metric_score)
            for metric_score in self.metric.get_score_names()
        ]

        self.evaluation_cache = None
        if cache_dir:
            self.evaluation_cache = EvaluatorCache(
                cache_dir=cache_dir,
                config_params={
                    "name": metric_definition.metric_id,
                    "metric_params": metric_definition.metric_params,
                },
                cache_key_fields=set(fields),
            )

    def run_metrics(
        self,
        list_of_inference_results: list[InferenceResult],
        evaluation_level: EvaluationLevel = EvaluationLevel.DOC_ID,
    ) -> EvaluationResults:

        if evaluation_level != EvaluationLevel.DOC_ID:
            raise UnsupportedOperation(
                f"Currently, we support evaluation at level of doc_id not `{evaluation_level}`"
            )
        question_id_to_inference_results_dict: dict[str, InferenceResult] = {
            entry.question_id: entry for entry in list_of_inference_results
        }
        # dataset = self._gt_context_id_to_str(dataset, evaluation_level)
        not_in_cache_qids: set[str] = set(question_id_to_inference_results_dict.keys())

        # Result data structure - track cache hits and scores
        question_id_to_metric_scores: dict[str, dict[str, float]] = defaultdict(dict)
        question_id_from_cache: dict[str, bool] = defaultdict(lambda: False)
        cache_hits = 0

        # We first look for the cache content
        if self.evaluation_cache:
            for q_id, inference_res in question_id_to_inference_results_dict.items():
                scores_dict = self.evaluation_cache.get(inference_res)
                if scores_dict is not None:
                    # We check that we have all the score_names
                    if not all(
                        score_name in scores_dict
                        for score_name in self.full_score_names
                    ):
                        logger.error(
                            f"We do not have all the full_scores_names in the cache : {scores_dict.keys()} vs. {self.full_score_names}"
                        )
                    else:
                        not_in_cache_qids.remove(q_id)
                        cache_hits += 1
                        for full_score_name, score in scores_dict.items():
                            question_id_to_metric_scores[q_id][full_score_name] = score
                            question_id_from_cache[q_id] = True

        not_in_cache_dataset: list[InferenceResult] = [
            d for d in list_of_inference_results if d.question_id in not_in_cache_qids
        ]

        if len(not_in_cache_dataset) == 0:
            logger.info(f"Metric {self.metric} is skipped (loaded entirely from cache)")
        else:
            metric_instance_scores = self.metric.compute(not_in_cache_dataset)

            for score_name, scores in metric_instance_scores.items():
                full_score_name = self.metric.full_score_name(score_name)
                for q_id, score in scores.items():
                    question_id_to_metric_scores[q_id][full_score_name] = score

            if self.evaluation_cache:
                for q_id in not_in_cache_qids:
                    metric_scores = question_id_to_metric_scores[q_id]
                    inference_res = question_id_to_inference_results_dict[q_id]
                    self.evaluation_cache.add(
                        inference_result=inference_res, score_dict=metric_scores
                    )

        # Build structured results
        results = []
        for q_id, scores_dict in question_id_to_metric_scores.items():
            metric_score_list = [
                MetricScore(
                    metric_name=metric_name,
                    score=score,
                    from_cache=question_id_from_cache[q_id],
                )
                for metric_name, score in scores_dict.items()
            ]
            results.append(
                QuestionEvaluationResult(question_id=q_id, scores=metric_score_list)
            )

        # Calculate cache hit rate
        total_questions = len(list_of_inference_results)
        cache_hit_rate = cache_hits / total_questions if total_questions > 0 else 0.0

        return EvaluationResults(
            results=results,
            evaluation_level=evaluation_level,
            metric_names=self.full_score_names,
            cache_hit_rate=cache_hit_rate,
        )

    def compute_stats_from_per_question_results(
        self, evaluation_results: EvaluationResults
    ) -> dict[str, dict[str, float]]:
        """
        Compute statistics from per-question evaluation results.

        Parameters
        ----------
        evaluation_results : EvaluationResults
            The evaluation results containing per-question scores

        Returns
        -------
        dict[str, dict[str, float]]
            Statistics for each metric (mean, ci_low, ci_high, coverage)
        """
        metric_to_scores = defaultdict(list)
        for full_metric_score in self.full_score_names:
            for question_result in evaluation_results.results:
                score = question_result.get_score(full_metric_score)
                if score is not None:
                    metric_to_scores[full_metric_score].append(score)
        metric_stats = self._compute_stats(metric_to_scores)
        return metric_stats

    def _compute_stats(
        self, metric_to_scores: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        # compute the mean, ci_low, ci_high and coverage
        eval_stats = {}
        for full_metric_score in self.full_score_names:
            eval_stats[full_metric_score] = BaseEvaluationMetric.compute_stats(
                metric_to_scores[full_metric_score]
            )
        return eval_stats
