from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.stats import bootstrap  # type: ignore[import-not-found]

from ragbench.api.inference_result import InferenceResult
from ragbench.eval.evaluation_level import EvaluationLevel

# Bootstrap random state for reproducibility
BOOTSTRAP_RANDOM_STATE = 1024


class EvaluationException(Exception):
    """
    Raised when evaluation errors occur due to invalid configurations,
    missing data, or incompatible metric parameters.
    """


class BaseEvaluationMetric(ABC):
    """
    Abstract base class for RAG evaluation metrics.

    Defines the interface for evaluating RAG applications and computing
    statistical measures over inference results.
    """

    def __init__(
        self,
        name: str,
        metric_params: dict[str, Any],
        evaluation_level: EvaluationLevel = EvaluationLevel.DOC_ID,
    ) -> None:
        """
        Initialize the evaluation metric.

        Parameters
        ----------
        name : str
            Unique identifier for the metric
        metric_params : dict[str, Any]
            Configuration parameters for the metric
        evaluation_level : EvaluationLevel
            Level at which evaluation is performed (default: DOC_ID)
        """
        self.name = name
        self.metric_params = metric_params
        self.evaluation_level = evaluation_level

    @abstractmethod
    def compute(
        self, inference_result_list: list[InferenceResult]
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate inference results against the metric.

        Parameters
        ----------
        inference_result_list : list[InferenceResult]
            Dataset to perform the metric evaluation on.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dictionary where outer keys are metric names and inner
            dicts contain statistical measures (mean, ci_low, ci_high, coverage).
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the full metric name.

        Returns
        -------
        str
            The complete name identifier of the metric.
        """

    @abstractmethod
    def get_score_names(self) -> list[str]:
        """
        Get all score names produced by this metric.

        Returns
        -------
        list[str]
            List of score identifiers this metric computes.
        """

    def full_score_name(self, score_name: str) -> str:
        """
        Construct the fully qualified score name.

        Parameters
        ----------
        score_name : str
            The score identifier

        Returns
        -------
        str
            Fully qualified score name with metric namespace.
        """
        if score_name == self.name:
            return self.name

        last_dot_index = self.name.rfind(".")
        if last_dot_index == -1:
            return f"{self.name}.{score_name}"

        return f"{self.name[:last_dot_index]}.{score_name}"

    @staticmethod
    def compute_stats(scores: list[float]) -> dict[str, float]:
        """
        Compute statistical measures for a list of scores.

        Parameters
        ----------
        scores : list[float]
            Raw scores to analyze

        Returns
        -------
        dict[str, float]
            Dictionary containing mean, confidence intervals, and coverage.

        Raises
        ------
        ValueError
            If scores list is empty.
        """
        if not scores:
            raise ValueError("Cannot compute statistics on empty score list")

        full_size = len(scores)
        scores_array = np.array(scores)
        finite_scores = scores_array[np.isfinite(scores_array)]

        if len(finite_scores) == 0:
            raise ValueError("All scores are non-finite (NaN or Inf)")

        coverage = len(finite_scores) / full_size
        mean = np.mean(finite_scores)

        # If all scores are identical, confidence interval equals the mean
        if np.allclose(finite_scores, mean):
            ci = (mean, mean)
        else:
            ci = bootstrap(
                (finite_scores,), np.mean, random_state=BOOTSTRAP_RANDOM_STATE
            ).confidence_interval

        return {
            "mean": float(mean),
            "ci_low": float(ci[0]),
            "ci_high": float(ci[1]),
            "coverage": float(coverage),
        }

    @abstractmethod
    def cleanup(self) -> None:
        """
        Perform cleanup operations after metric computation.

        This method should release any resources, close connections,
        or perform necessary teardown operations.
        """
