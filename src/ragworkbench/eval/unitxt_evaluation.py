from collections import defaultdict

from unitxt.operator import (  # type: ignore[import-not-found]
    MultiStream,
    SequentialOperator,
)

from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.eval.evaluation import BaseEvaluationMetric


class UnitxtEvaluationMetric(BaseEvaluationMetric):
    """
    Evaluation metric implementation using IBM's Unitxt framework.
    """

    def compute(
        self, inference_result_list: list[InferenceResult]
    ) -> dict[str, dict[str, float]]:
        # Use metric_id for Unitxt (the actual metric identifier)
        # but use name (metric_name) for score naming
        metrics_operator = SequentialOperator(steps=[self.metric_id])
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
