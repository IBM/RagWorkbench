"""
Data models for evaluation results.

This module defines structured data models for representing evaluation results
using Pydantic for type safety, validation, and better API clarity.
"""

from pydantic import BaseModel, Field

from ragbench.eval.evaluation_level import EvaluationLevel


class MetricScore(BaseModel):
    """
    Individual metric score for a question.

    Attributes
    ----------
    metric_name : str
        Full metric name (e.g., 'unitxt.context_correctness.map')
    score : float
        Computed score value
    from_cache : bool
        Whether score was loaded from cache (default: False)
    """

    metric_name: str = Field(
        description="Full metric name (e.g., 'unitxt.context_correctness.map')"
    )
    score: float = Field(description="Computed score value")
    from_cache: bool = Field(
        default=False, description="Whether score was loaded from cache"
    )


class QuestionEvaluationResult(BaseModel):
    """
    Evaluation results for a single question.

    Attributes
    ----------
    question_id : str
        Unique question identifier
    scores : list[MetricScore]
        List of metric scores for this question
    """

    question_id: str = Field(description="Unique question identifier")
    scores: list[MetricScore] = Field(description="List of metric scores")

    def get_score(self, metric_name: str) -> float | None:
        """
        Get score by metric name.

        Parameters
        ----------
        metric_name : str
            The metric name to look up

        Returns
        -------
        float | None
            The score value if found, None otherwise
        """
        for score in self.scores:
            if score.metric_name == metric_name:
                return score.score
        return None

    def to_dict(self) -> dict[str, float]:
        """
        Convert to simple dict for backward compatibility.

        Returns
        -------
        dict[str, float]
            Dictionary mapping metric names to scores
        """
        return {score.metric_name: score.score for score in self.scores}


class EvaluationResults(BaseModel):
    """
    Complete evaluation results for all questions.

    Attributes
    ----------
    results : list[QuestionEvaluationResult]
        Per-question evaluation results
    evaluation_level : EvaluationLevel
        Level at which evaluation was performed
    metric_names : list[str]
        All metric names that were evaluated
    cache_hit_rate : float
        Proportion of results loaded from cache (0.0 to 1.0)
    """

    results: list[QuestionEvaluationResult] = Field(description="Per-question results")
    evaluation_level: EvaluationLevel = Field(
        description="Level at which evaluation was performed"
    )
    metric_names: list[str] = Field(description="All metric names evaluated")
    cache_hit_rate: float = Field(
        ge=0.0, le=1.0, description="Proportion of results from cache"
    )

    def get_question_result(self, question_id: str) -> QuestionEvaluationResult | None:
        """
        Get results for a specific question.

        Parameters
        ----------
        question_id : str
            The question ID to look up

        Returns
        -------
        QuestionEvaluationResult | None
            The question result if found, None otherwise
        """
        for result in self.results:
            if result.question_id == question_id:
                return result
        return None

    def to_dict(self) -> dict[str, dict[str, float]]:
        """
        Convert to simple nested dict for backward compatibility.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dictionary: question_id -> {metric_name: score}
        """
        return {result.question_id: result.to_dict() for result in self.results}


# Made with Bob
