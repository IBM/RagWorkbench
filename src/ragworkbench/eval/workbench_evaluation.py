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
        - usage: Aggregates token usage from all model calls
          (outputs: total_tokens, prompt_tokens, completion_tokens)
        """
        metric_scores: dict = defaultdict(dict)

        if self.metric_id == "workbench.tool_use_count":
            for result in inference_result_list:
                q_id = result.question_id
                # Count the number of items in the trajectory
                tool_count = len(result.trajectory) if result.trajectory else 0
                metric_scores[self.name][q_id] = float(tool_count)
        elif self.metric_id == "workbench.usage":
            # Compute all token counts first
            all_scores: dict[str, dict[str, float]] = {
                "total_tokens": {},
                "prompt_tokens": {},
                "completion_tokens": {},
            }

            for result in inference_result_list:
                q_id = result.question_id
                # Aggregate token counts from all model calls
                total_tokens = 0
                prompt_tokens = 0
                completion_tokens = 0

                if result.model_calls:
                    for model_call in result.model_calls:
                        total_tokens += model_call.usage.total_tokens
                        prompt_tokens += model_call.usage.prompt_tokens
                        completion_tokens += model_call.usage.completion_tokens

                # Store each token type
                all_scores["total_tokens"][q_id] = float(total_tokens)
                all_scores["prompt_tokens"][q_id] = float(prompt_tokens)
                all_scores["completion_tokens"][q_id] = float(completion_tokens)

            # Only include scores that are in sub_scores
            metric_scores.update(
                {
                    score_name: all_scores[score_name]
                    for score_name in self.sub_scores
                    if score_name in all_scores
                }
            )
        else:
            raise ValueError(f"Unknown workbench metric: '{self.metric_id}'")

        return metric_scores
