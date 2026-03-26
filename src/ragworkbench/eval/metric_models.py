from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class MetricDefinition(BaseModel):
    """
    Represents a single metric definition from metric_defs.yaml.

    Each metric definition specifies how to evaluate a RAG system using
    a particular metric from a vendor's evaluation framework. The definition
    includes the metric identifier, configuration parameters, required fields,
    and the vendor framework to use.

    Attributes:
        metric_name: The name/key of the metric from metric_defs.yaml (e.g.,
                     "unitxt.answer_correctness.llmaaj_llama"). This is used
                     for referencing the metric in configurations and reports.
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
        ...     metric_name="unitxt.context_correctness.retrieval_at_k",
        ...     metric_id="metrics.rag.context_correctness.retrieval_at_k",
        ...     metric_params={"sub_scores": ["match_at_1", "match_at_3"]},
        ...     metric_fields=["context_ids", "ground_truths_context_ids"],
        ...     vendor="unitxt"
        ... )
    """

    metric_name: str = Field(
        min_length=1,
        description="The name/key of the metric from metric_defs.yaml for referencing in configs.",
    )
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
        - workbench: RagWorkbench's built-in metrics

        Args:
            v: The vendor string to validate.

        Returns:
            The validated vendor string.

        Raises:
            ValueError: If the vendor is not in the list of known vendors.
        """
        known_vendors = {"unitxt", "workbench"}  # Extensible for future vendors
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

    def get_metric_names(self) -> list[str]:
        """
        Get all available metric entry names from the definitions.

        Returns a list of all metric names (keys) defined in the configuration.
        These names correspond to the top-level keys in the metric_defs.yaml file.

        Returns:
            List of metric entry names (e.g., ["unitxt.context_correctness.retrieval_at_k",
                                               "unitxt.context_correctness.map", ...])

        Example:
            >>> config = load_metric_definitions()
            >>> names = config.get_metric_names()
            >>> print(names[0])
            unitxt.context_correctness.retrieval_at_k
        """
        return list(self.definitions.keys())

    def get_metric_definition(self, metric_name: str) -> MetricDefinition:
        """
        Get a specific metric definition by its name.

        Args:
            metric_name: The name of the metric to retrieve (e.g.,
                        "unitxt.context_correctness.retrieval_at_k").

        Returns:
            The MetricDefinition object for the specified metric.

        Raises:
            KeyError: If the metric name is not found in the definitions.

        Example:
            >>> config = load_metric_definitions()
            >>> metric_def = config.get_metric_definition("unitxt.context_correctness.map")
            >>> print(metric_def.metric_id)
            metrics.rag.context_correctness.map
            >>> print(metric_def.vendor)
            unitxt
        """
        if metric_name not in self.definitions:
            available_metrics = "\n".join(self.get_metric_names())
            raise KeyError(
                f"Metric '{metric_name}' not found. "
                f"Available metrics:\n{available_metrics}"
            )
        return self.definitions[metric_name]


def load_metric_definitions(
    yaml_path: Path | str | None = None,
) -> MetricDefinitionsConfig:
    """
    Load metric definitions from a YAML file and return a MetricDefinitionsConfig instance.

    This function reads the metric definitions YAML file, parses it, and constructs
    a validated MetricDefinitionsConfig object. If no path is provided, it loads
    the default metric_defs.yaml file bundled with the package.

    Args:
        yaml_path: Optional path to the YAML file containing metric definitions.
                   If None, uses the default metric_defs.yaml file in the same
                   directory as this module.

    Returns:
        MetricDefinitionsConfig instance containing all parsed and validated
        metric definitions.

    Raises:
        FileNotFoundError: If the specified YAML file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
        ValueError: If the YAML content doesn't match the expected schema.

    Example:
        >>> # Load default metric definitions
        >>> config = load_metric_definitions()
        >>> metric_names = config.get_metric_names()
        >>> print(len(metric_names))
        10

        >>> # Load from custom path
        >>> config = load_metric_definitions("path/to/custom_metrics.yaml")
        >>> retrieval_metric = config.definitions["unitxt.context_correctness.retrieval_at_k"]
        >>> print(retrieval_metric.vendor)
        unitxt
    """
    # Determine the YAML file path
    if yaml_path is None:
        # Use the default metric_defs.yaml in the same directory as this module
        yaml_path = Path(__file__).parent / "metric_defs.yaml"
    else:
        yaml_path = Path(yaml_path)

    # Check if file exists
    if not yaml_path.exists():
        raise FileNotFoundError(f"Metric definitions file not found: {yaml_path}")

    # Load and parse the YAML file
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Validate that we got a dictionary
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML file to contain a dictionary, got {type(data).__name__}"
        )

    # Convert the raw dictionary to MetricDefinition objects
    # Add the metric_name field to each definition
    definitions = {
        name: MetricDefinition(metric_name=name, **definition)
        for name, definition in data.items()
    }

    # Create and return the MetricDefinitionsConfig instance
    return MetricDefinitionsConfig(definitions=definitions)
