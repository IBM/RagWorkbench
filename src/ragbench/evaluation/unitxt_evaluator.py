from collections import defaultdict
from typing import Any

from ragbench.api.inference_result import InferenceResult
from ragbench.evaluation.base_evaluation_metric import BaseEvaluationMetric


class UnitxtEvaluationMetric(BaseEvaluationMetric):
    """"""

    def __init__(
        self,
        name: str,
        metric_params: dict[str, Any],
        **kwargs,
    ):
        super().__init__(name, metric_params)
        self.sub_scores = (
            metric_params["sub_scores"] if "sub_scores" in metric_params.keys() else []
        )

    def get_name(self) -> str:
        return self.name

    def get_score_names(self) -> list[str]:
        return [self.name] if len(self.sub_scores) == 0 else self.sub_scores

    def compute(self, inference_result_list:list[InferenceResult]):
        metric_id = self.get_name()
        # load the metric and prepare the dataset
        metrics_operator = SequentialOperator(steps=[metric_id])
        # We must create strings from GTContextID!
        multi_stream = MultiStream.from_iterables({"test": dataset}, copying=True)

        # compute the metric
        instances = list(metrics_operator(multi_stream)["test"])

        # collect instance scores
        metric_scores = defaultdict(dict)
        models = []
        for entry, instance in zip(dataset, instances):
            q_id = entry["q_id"]
            instance_scores = instance["score"]["instance"]
            for score_name, score_value in instance_scores.items():
                if (
                    score_name == "score" and len(self.sub_scores) == 0
                ) or score_name in self.sub_scores:
                    name = self.name if score_name == "score" else score_name
                    metric_scores[name][q_id] = score_value
            if "model_id" in instance_scores:
                models.append(
                    Model(
                        model_id=instance_scores["model_id"],
                        host=instance_scores["host"],
                        input_tokens=instance_scores.get("input_tokens", 0),
                        output_tokens=instance_scores.get("output_tokens", 0),
                    )
                )
        return metric_scores

    def cleanup(self) -> None:
        pass
