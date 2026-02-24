from typing import Any

from pydantic import BaseModel, Field, field_validator


class MetricDefinition(BaseModel):
    """
    Represents a single metric definition from metric_defs.yaml.

    Each metric definition specifies how to evaluate a RAG system using
    a particular metric from a vendor's evaluation framework. The definition
    includes the metric identifier, configuration parameters, required fields,
    and the vendor framework to use.

    Attributes:
        metric_id: Unique identifier for the metric in the vendor's evaluation framework.
                   This is the actual metric name used by the framework (e.g.,
                   "metrics.rag.context_correctness.retrieval_at_k").
        metric_params: Configuration parameters for the metric. Can include settings
                       like "sub_scores" for metrics that compute multiple scores.
                       Empty dict if no parameters are needed.
        metric_fields: List of field names required from the dataset for evaluation.
                       These fields must be present in the evaluation dataset
                       (e.g., ["context_ids", "ground_truths_context_ids"]).
        vendor: The evaluation framework vendor providing this metric (e.g., "unitxt").
                Must be one of the known/supported vendors.

    Example:
        >>> metric = MetricDefinition(
        ...     metric_id="metrics.rag.context_correctness.retrieval_at_k",
        ...     metric_params={"sub_scores": ["match_at_1", "match_at_3"]},
        ...     metric_fields=["context_ids", "ground_truths_context_ids"],
        ...     vendor="unitxt"
        ... )
    """

    metric_id: str = Field(
        min_length=1,
        description="Unique identifier for the metric in the vendor's evaluation framework.",
    )
    metric_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters for the metric (e.g., sub_scores).",
    )
    metric_fields: list[str] = Field(
        min_length=1,
        description="List of field names required from the dataset for evaluation.",
    )
    vendor: str = Field(
        min_length=1,
        description="The evaluation framework vendor (e.g., 'unitxt').",
    )

    @field_validator("vendor")
    @classmethod
    def validate_vendor(cls, v: str) -> str:
        """
        Validate that the vendor is a known evaluation framework.

        Currently supported vendors:
        - unitxt: IBM's Unitxt evaluation framework

        Args:
            v: The vendor string to validate.

        Returns:
            The validated vendor string.

        Raises:
            ValueError: If the vendor is not in the list of known vendors.
        """
        known_vendors = {"unitxt"}  # Extensible for future vendors like "ragas"
        if v not in known_vendors:
            raise ValueError(f"Unknown vendor: '{v}'. Known vendors: {known_vendors}")
        return v

    @field_validator("metric_fields")
    @classmethod
    def validate_metric_fields(cls, v: list[str]) -> list[str]:
        """
        Validate that metric_fields contains valid field names.

        Args:
            v: The list of metric field names to validate.

        Returns:
            The validated list of metric field names.

        Raises:
            ValueError: If any field name is empty or contains only whitespace.
        """
        for field in v:
            if not field or not field.strip():
                raise ValueError("Metric field names cannot be empty or whitespace")
        return v


class MetricDefinitionsConfig(BaseModel):
    """
    Represents the complete metric definitions configuration from metric_defs.yaml.

    This model provides a type-safe way to work with the metric configuration file,
    mapping metric names (keys in the YAML) to their corresponding definitions.
    It enables validation and structured access to all available metrics.

    Attributes:
        definitions: Dictionary mapping metric names to their MetricDefinition objects.
                     Keys are the metric names as they appear in the YAML file
                     (e.g., "unitxt.context_correctness.retrieval_at_k").

    Example:
        >>> import yaml
        >>> with open("metric_defs.yaml") as f:
        ...     data = yaml.safe_load(f)
        >>> config = MetricDefinitionsConfig(definitions={
        ...     name: MetricDefinition(**definition)
        ...     for name, definition in data.items()
        ... })
        >>> # Access a specific metric
        >>> retrieval_metric = config.definitions["unitxt.context_correctness.retrieval_at_k"]
        >>> print(retrieval_metric.metric_id)
        metrics.rag.context_correctness.retrieval_at_k
    """

    definitions: dict[str, MetricDefinition] = Field(
        description="Dictionary mapping metric names to their definitions.",
    )

    @field_validator("definitions")
    @classmethod
    def validate_definitions(
        cls, v: dict[str, MetricDefinition]
    ) -> dict[str, MetricDefinition]:
        """
        Validate that the definitions dictionary is not empty.

        Args:
            v: The definitions dictionary to validate.

        Returns:
            The validated definitions dictionary.

        Raises:
            ValueError: If the definitions dictionary is empty.
        """
        if not v:
            raise ValueError(
                "MetricDefinitionsConfig must contain at least one metric definition"
            )
        return v
