from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.stats import bootstrap  # type: ignore[import-not-found]

from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.eval.evaluation_level import EvaluationLevel


class EvaluationException(Exception):
    """
    Raised for eval errors of mis-configurations.
    """


class BaseEvaluationMetric(ABC):
    """
    This class defines the functionality to evaluate a RAG application
    and compare different RAG applications.
    """

    def __init__(
        self,
        name: str,
        metric_params: dict[str, Any],
        evaluation_level: EvaluationLevel = EvaluationLevel.DOC_ID,
        metric_id: str | None = None,
        **kwargs,
    ):
        self.name = name
        self.metric_id = metric_id if metric_id else name
        self.metric_params = metric_params
        self.evaluation_level = evaluation_level
        self.sub_scores = (
            metric_params["sub_scores"] if "sub_scores" in metric_params.keys() else []
        )

    @abstractmethod
    def compute(
        self, inference_result_list: list[InferenceResult]
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate the RAGPattern.query() response against list of different metrics.

        Parameters
        ----------
        inference_result_list : List[InferenceResult]
            Dataset to preform the metric eval on.

        Returns
        -------
        dict
            Evaluation result data
        """

    def get_name(self) -> str:
        """
        Returns the metric full name

        Returns
        -------
        str
            the full name of the metric
        """
        return self.name

    def get_score_names(self) -> list[str]:
        """
        Returns the list of score names for this metric.

        If sub_scores are defined in metric_params, returns those.
        Otherwise, returns a list containing just the metric name.

        Returns
        -------
        list[str]
            List of score names
        """
        return [self.name] if len(self.sub_scores) == 0 else self.sub_scores

    def full_score_name(self, score_name):
        return self.name if score_name == self.name else self.name + "." + score_name

    @staticmethod
    def compute_stats(scores: list[float]) -> dict[str, float]:
        full_size = len(scores)
        scores = np.array(scores)  # type: ignore[assignment]
        scores = scores[np.isfinite(scores)]  # type: ignore[assignment]
        coverage = len(scores) / full_size
        mean = np.mean(scores)
        if np.allclose(scores, mean):
            ci = mean, mean
        else:
            ci = bootstrap((scores,), np.mean, random_state=1024).confidence_interval
        return {
            "mean": float(mean),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "coverage": coverage,
        }
