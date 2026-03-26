from collections import defaultdict

from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.eval.evaluation import BaseEvaluationMetric


class WorkbenchEvaluationMetric(BaseEvaluationMetric):
    """
    Custom evaluation metrics implemented directly in RagWorkbench.

    These metrics don't rely on external evaluation frameworks like Unitxt,
    but instead compute scores based on the InferenceResult data directly.
    """

    def compute(
        self, inference_result_list: list[InferenceResult]
    ) -> dict[str, dict[str, float]]:
        """
        Compute workbench-specific metrics based on the metric_id.

        Currently supported metrics:
        - tool_use_count: Counts the number of tool uses in the trajectory
        """
        metric_scores: dict = defaultdict(dict)

        if self.metric_id == "workbench.tool_use_count":
            for result in inference_result_list:
                q_id = result.question_id
                # Count the number of items in the trajectory
                tool_count = len(result.trajectory) if result.trajectory else 0
                metric_scores[self.name][q_id] = float(tool_count)
        else:
            raise ValueError(f"Unknown workbench metric: {self.metric_id}")

        return metric_scores
