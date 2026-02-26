from collections import defaultdict
from typing import Any

from unitxt.operator import (  # type: ignore[import-not-found]
    MultiStream,
    SequentialOperator,
)

from ragbench.api.inference_result import InferenceResult
from ragbench.eval.evaluation import BaseEvaluationMetric
from ragbench.eval.evaluation_level import EvaluationLevel


class UnitxtEvaluationMetric(BaseEvaluationMetric):
    """"""

    def __init__(
        self,
        name: str,
        metric_params: dict[str, Any],
        evaluation_level: EvaluationLevel = EvaluationLevel.DOC_ID,
        **kwargs,
    ):
        super().__init__(name, metric_params, evaluation_level)
        self.sub_scores = (
            metric_params["sub_scores"] if "sub_scores" in metric_params.keys() else []
        )

    def get_name(self) -> str:
        return self.name

    def get_score_names(self) -> list[str]:
        return [self.name] if len(self.sub_scores) == 0 else self.sub_scores

    def compute(self, inference_result_list: list[InferenceResult]) -> dict:
        metric_id = self.get_name()
        # load the metric and prepare the dataset
        metrics_operator = SequentialOperator(steps=[metric_id])
        dataset = [
            {
                "q_id": r.question_id,
                "question": r.question,
                "answer": r.answer,
                "ground_truths": r.ground_truth_answers,
                "ground_truths_context_ids": [
                    self.evaluation_level.gt_context_id_to_str(gt_context_id)
                    for gt_context_id in r.ground_truths_context_ids
                ],
                "contexts": r.contexts,
                "context_ids": r.context_ids,
            }
            for r in inference_result_list
        ]
        multi_stream = MultiStream.from_iterables({"test": dataset}, copying=True)

        # compute the metric
        instances = list(metrics_operator(multi_stream)["test"])

        # collect instance scores
        metric_scores: dict = defaultdict(dict)
        for entry, instance in zip(dataset, instances, strict=True):
            q_id = entry["q_id"]
            instance_scores = instance["score"]["instance"]
            for score_name, score_value in instance_scores.items():
                if (
                    score_name == "score" and len(self.sub_scores) == 0
                ) or score_name in self.sub_scores:
                    name = self.name if score_name == "score" else score_name
                    metric_scores[name][q_id] = score_value
        return metric_scores

    def cleanup(self) -> None:
        pass
