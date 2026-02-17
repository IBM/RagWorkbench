from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from scipy.stats import bootstrap


class EvaluationException(Exception):
    """
    Raised for evaluation errors of mis-configurations.
    """


class BaseEvaluationMetric(ABC):
    """
    This class defines the functionality to evaluate a RAG application
    and compare different RAG applications.
    """

    def __init__(
        self,
        name: str,
        metric_params: Dict[str, Any],
        **kwargs,
    ):
        self.name = name
        self.metric_params = metric_params

    @abstractmethod
    def compute(self, dataset: List[Dict[str, Any]]):
        """
        Evaluate the RAGPattern.query() response against list of different metrics.

        Parameters
        ----------
        dataset : List[Dict[str, Any]]
            Dataset to preform the metric evaluation on.

        Returns
        -------
        dict
            Evaluation result data
        """

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the metric full name

        Returns
        -------
        str
            the full name of the metric
        """

    @abstractmethod
    def get_score_names(self) -> List[str]:
        pass

    def full_score_name(self, score_name):
        return (
            self.name
            if score_name == self.name
            else self.name[: self.name.rindex(".")] + "." + score_name
        )

    @staticmethod
    def compute_stats(scores: List[float]) -> dict[str, float]:
        full_size = len(scores)
        scores = np.array(scores)
        scores = scores[np.isfinite(scores)]
        coverage = len(scores) / full_size
        mean = np.mean(scores)
        if np.allclose(scores, mean):
            ci = mean, mean
        else:
            ci = bootstrap((scores,), np.mean, random_state=1024).confidence_interval
        return {
            "mean": mean,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "coverage": coverage,
        }

    @abstractmethod
    def cleanup(self) -> None:
        pass
