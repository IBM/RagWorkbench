import logging
from collections import defaultdict
from enum import StrEnum, auto
from io import UnsupportedOperation
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ragbench.api.inference_result import InferenceResult
from ragbench.datasets_loader.data_models import GroundTruthContextId, RagBenchmark, RagCorpus
from ragbench.evaluation.base_evaluation_metric import BaseEvaluationMetric
from ragbench.evaluation.evaluator_cache import EvaluatorCache
from ragbench.evaluation.unitxt_evaluator import UnitxtEvaluationMetric

logger = logging.getLogger(__name__)


class MetricDefinition(BaseModel):
    """Pydantic model for metric definition configuration."""
    
    vendor: str = Field(..., description="The vendor/provider of the metric (e.g., 'unitxt')")
    metric_id: str = Field(..., description="Unique identifier for the metric")
    metric_params: dict[str, Any] = Field(..., description="Parameters for the metric")
    metric_fields: dict[str, Any] | None = Field(None, description="Optional fields for cache key")


class EvaluationLevel(StrEnum):
    DOC_ID = auto()
    PAGE_ID = auto()
    TABLE_ID = auto()

    def gt_context_id_to_str(self, gt_context_id: GroundTruthContextId):
        match self:
            case EvaluationLevel.DOC_ID:
                return gt_context_id.document_id
            case EvaluationLevel.PAGE_ID:
                if gt_context_id.page is None:
                    raise Exception(
                        f"gt_context_id does not contain page info `{gt_context_id}`"
                    )
                return f"{gt_context_id.document_id}_page-{gt_context_id.page}"
            case EvaluationLevel.TABLE_ID:
                if gt_context_id.page is None:
                    raise Exception(
                        f"gt_context_id does not contain page info `{gt_context_id}`"
                    )
                if gt_context_id.table_id is None:
                    raise Exception(
                        f"gt_context_id does not contain table info `{gt_context_id}`"
                    )
                return f"{gt_context_id.document_id}_page-{gt_context_id.page}_table-{gt_context_id.table_id}"
            case _:
                raise Exception(f"Unknown stage {self.name}")


class Evaluator:

    evaluation_metric_factory = {
        "unitxt": UnitxtEvaluationMetric,
        # "workbench": RagWorkbenchEvaluationMetric, # We must keep it! It is a custom evaluator (nice to have)
    }

    def __init__(
        self,
        metric_definition: MetricDefinition | dict[str, Any],
        rag_benchmark: RagBenchmark | None = None,
        rag_corpus: RagCorpus | None = None,
        cache_dir: Path | None = None,
    ):
        # Convert dict to MetricDefinition if needed
        if isinstance(metric_definition, dict):
            metric_definition = MetricDefinition(**metric_definition)

        self.metric: BaseEvaluationMetric = self.evaluation_metric_factory[
            metric_definition.vendor
        ](
            name=metric_definition.metric_id,
            metric_params=metric_definition.metric_params,
            rag_benchmark=rag_benchmark,
            rag_corpus=rag_corpus,
        )

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
                cache_key_fields=metric_definition.metric_fields,
            )

    @staticmethod
    def _gt_context_id_to_str(
        dataset: list[dict[str, Any]],
        evaluation_level: EvaluationLevel = EvaluationLevel.DOC_ID,
    ) -> list[dict[str, Any]]:
        result = list()

        for d in dataset:
            if "ground_truths_context_ids" in d:
                d["ground_truths_context_ids"] = [
                    (
                        evaluation_level.gt_context_id_to_str(
                            gt_context_id=ground_truth_context_id
                        )
                        if isinstance(ground_truth_context_id, GroundTruthContextId)
                        else ground_truth_context_id
                    )
                    for ground_truth_context_id in d["ground_truths_context_ids"]
                ]
            result.append(d)
        return result

    def run_metrics(
        self,
        inference_result_list: list[InferenceResult],
        evaluation_level: EvaluationLevel = EvaluationLevel.DOC_ID,
    ) -> dict[str, dict[str, float]]:
        """
        TODO
        Example input:
            dataset = [
                {"q_id": "1", "ground_truths": ["Joe Biden"], "answer": "Wrong Answer", },
                { "q_id": "2", "ground_truths": ["Joe Biden"], "answer": "Joe Biden", # Correct answer }
            ]
        Example output:
            {
                '1': {'metrics.rag.answer_correctness': 0.0},
                '2': {'metrics.rag.answer_correctness': 1.0}
            }
        """

        # Auxiliary data structure
        if evaluation_level != EvaluationLevel.DOC_ID:
            raise UnsupportedOperation(
                f"Currently, we support evaluation at level of doc_id not `{evaluation_level}`"
            )
        q_id_to_data : dict[str, InferenceResult] = {entry.question_id: entry for entry in inference_result_list}
        #dataset = self._gt_context_id_to_str(dataset, evaluation_level)
        not_in_cache_qids = set(q_id_to_data.keys())

        # Result data structure
        # TODO : MetricScore
        question_id_to_metric_scores: dict[str, dict[str, float]] = defaultdict(dict)

        # TODO
        # # We first look for the cache content
        # if self.evaluation_cache:
        #     for q_id, d in q_id_to_data.items():
        #         scores_dict: dict[str, float] = self.evaluation_cache.get(d)
        #         if scores_dict is not None:
        #             # We check that we have all the score_names
        #             if not (
        #                 score_name in scores_dict
        #                 for score_name in self.full_score_names
        #             ):
        #                 logger.error(
        #                     "We do not have all the full_scores_names in the cache : {scores_dict.keys()} vs. {self.full_score_names}]"
        #                 )
        #             else:
        #                 not_in_cache_qids.remove(q_id)
        #                 for full_score_name, score in scores_dict.items():
        #                     question_id_to_metric_scores[q_id][full_score_name] = score

        not_in_cache_inference_result_list : list[InferenceResult] = [d for d in inference_result_list if d.question_id in not_in_cache_qids]

        if len(not_in_cache_dataset) == 0:
            logger.info(f"Metric {self.metric} is skipped (loaded entirely from cache)")
        else:
            metric_instance_scores : dict[str, Any] = self.metric.compute(not_in_cache_inference_result_list)

            for score_name, scores in metric_instance_scores.items():
                full_score_name = self.metric.full_score_name(score_name)
                for q_id, score in scores.items():
                    question_id_to_metric_scores[q_id][full_score_name] = score

            if self.evaluation_cache:
                for q_id in not_in_cache_qids:
                    metric_scores = question_id_to_metric_scores[q_id]
                    evaluation_dict = q_id_to_data[q_id]
                    self.evaluation_cache.add(
                        evaluation_dict=evaluation_dict, score_dict=metric_scores
                    )
        return question_id_to_metric_scores

    def compute_stats_from_per_question_results(
        self, question_id_to_metric_scores: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float]]:
        metric_to_scores = defaultdict(list)
        for full_metric_score in self.full_score_names:
            for evaluation_result in question_id_to_metric_scores.values():
                metric_to_scores[full_metric_score].append(
                    evaluation_result[full_metric_score]
                )
        metric_stats = self.compute_stats(metric_to_scores)
        return metric_stats

    def compute_stats(
        self, metric_to_scores: dict[str, list[float]]
    ) -> dict[str, dict[str, float]]:
        # compute the mean, ci_low, ci_high and coverage
        eval_stats = dict()
        for full_metric_score in self.full_score_names:
            eval_stats[full_metric_score] = BaseEvaluationMetric.compute_stats(
                metric_to_scores[full_metric_score]
            )
        return eval_stats
