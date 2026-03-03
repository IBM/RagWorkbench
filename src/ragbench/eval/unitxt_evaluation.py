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
    """
    Unitxt-based evaluation metric for RAG systems.

    This class integrates with the Unitxt library to compute evaluation metrics
    on RAG inference results. It supports both single metrics and metrics with
    multiple sub-scores.
    """

    def __init__(
        self,
        name: str,
        metric_params: dict[str, Any],
        evaluation_level: EvaluationLevel = EvaluationLevel.DOC_ID,
    ) -> None:
        """
        Initialize the Unitxt evaluation metric.

        Parameters
        ----------
        name : str
            Unique identifier for the metric (must match a Unitxt metric ID)
        metric_params : dict[str, Any]
            Configuration parameters for the metric. May include:
            - 'sub_scores': list of sub-score names to extract (optional)
        evaluation_level : EvaluationLevel
            Level at which evaluation is performed (default: DOC_ID)
        """
        super().__init__(name, metric_params, evaluation_level)
        self.sub_scores: list[str] = metric_params.get("sub_scores", [])

    def get_name(self) -> str:
        """
        Get the metric name.

        Returns
        -------
        str
            The metric identifier.
        """
        return self.name

    def get_score_names(self) -> list[str]:
        """
        Get all score names produced by this metric.

        Returns
        -------
        list[str]
            List of score names. Returns the metric name if no sub-scores
            are defined, otherwise returns the list of sub-score names.
        """
        return [self.name] if not self.sub_scores else self.sub_scores

    def compute(
        self, inference_result_list: list[InferenceResult]
    ) -> dict[str, dict[str, float]]:
        """
        Compute the Unitxt metric on inference results.

        This method:
        1. Prepares the dataset in Unitxt format
        2. Applies the metric operator
        3. Extracts and organizes scores by question ID

        Parameters
        ----------
        inference_result_list : list[InferenceResult]
            List of inference results to evaluate.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dictionary where:
            - Outer keys are score names (metric name or sub-score names)
            - Inner keys are question IDs
            - Values are the computed scores

        Raises
        ------
        KeyError
            If expected score fields are missing from Unitxt output.
        """
        metric_id = self.get_name()

        # Initialize the Unitxt metric operator
        metrics_operator = SequentialOperator(steps=[metric_id])

        # Prepare dataset in Unitxt format
        dataset = self._prepare_dataset(inference_result_list)

        # Create Unitxt MultiStream and compute metrics
        multi_stream = MultiStream.from_iterables({"test": dataset}, copying=True)
        instances = list(metrics_operator(multi_stream)["test"])

        # Extract and organize scores
        return self._extract_scores(dataset, instances)

    def _prepare_dataset(
        self, inference_result_list: list[InferenceResult]
    ) -> list[dict[str, Any]]:
        """
        Convert inference results to Unitxt dataset format.

        Parameters
        ----------
        inference_result_list : list[InferenceResult]
            Raw inference results.

        Returns
        -------
        list[dict[str, Any]]
            Dataset formatted for Unitxt processing.
        """
        return [
            {
                "q_id": result.question_id,
                "question": result.question,
                "answer": result.answer,
                "ground_truths": result.ground_truth_answers,
                "ground_truths_context_ids": [
                    self.evaluation_level.gt_context_id_to_str(gt_context_id)
                    for gt_context_id in result.ground_truths_context_ids
                ],
                "contexts": result.contexts,
                "context_ids": result.context_ids,
            }
            for result in inference_result_list
        ]

    def _extract_scores(
        self, dataset: list[dict[str, Any]], instances: list[dict[str, Any]]
    ) -> dict[str, dict[str, float]]:
        """
        Extract scores from Unitxt instances.

        Parameters
        ----------
        dataset : list[dict[str, Any]]
            Original dataset entries with question IDs.
        instances : list[dict[str, Any]]
            Unitxt instances with computed scores.

        Returns
        -------
        dict[str, dict[str, float]]
            Organized scores by metric name and question ID.
        """
        metric_scores: dict[str, dict[str, float]] = defaultdict(dict)

        for entry, instance in zip(dataset, instances, strict=True):
            q_id = entry["q_id"]
            instance_scores = instance["score"]["instance"]

            for score_name, score_value in instance_scores.items():
                # Include score if it's the main score (when no sub-scores defined)
                # or if it's in the list of requested sub-scores
                if self._should_include_score(score_name):
                    # Use metric name for main score, otherwise use sub-score name
                    final_name = self.name if score_name == "score" else score_name
                    metric_scores[final_name][q_id] = score_value

        return metric_scores

    def _should_include_score(self, score_name: str) -> bool:
        """
        Determine if a score should be included in the results.

        Parameters
        ----------
        score_name : str
            Name of the score from Unitxt output.

        Returns
        -------
        bool
            True if the score should be included, False otherwise.
        """
        # Include main score if no sub-scores are specified
        if score_name == "score" and not self.sub_scores:
            return True
        # Include if it's in the list of requested sub-scores
        return score_name in self.sub_scores

    def cleanup(self) -> None:
        """
        Perform cleanup operations after metric computation.

        Currently no cleanup is required for Unitxt metrics.
        """
        pass
