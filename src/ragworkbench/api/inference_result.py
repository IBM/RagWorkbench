from typing import Any

from pydantic import BaseModel, Field

from ragworkbench.datasets_loader.data_models import RagBenchmarkEntry

# Type alias for trajectory data structure
# Each trajectory entry represents a search query and its results
Trajectory = list[dict[str, Any]]


class ModelCallUsage(BaseModel):
    """
    Usage statistics for a single model call.

    Attributes:
        total_tokens: Total number of tokens used (prompt + completion + reasoning)
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        reasoning_tokens: Number of tokens used for reasoning (if applicable)
        input_cost: Cost in USD for input tokens
        output_cost: Cost in USD for output tokens
        total_cost: Total cost in USD for this call
    """

    total_tokens: int = Field(default=0, description="Total tokens used", ge=0)
    prompt_tokens: int = Field(default=0, description="Prompt tokens used", ge=0)
    completion_tokens: int = Field(
        default=0, description="Completion tokens used", ge=0
    )
    reasoning_tokens: int = Field(default=0, description="Reasoning tokens used", ge=0)
    input_cost: float = Field(default=0.0, description="Input cost in USD", ge=0.0)
    output_cost: float = Field(default=0.0, description="Output cost in USD", ge=0.0)
    total_cost: float = Field(default=0.0, description="Total cost in USD", ge=0.0)


class ModelCall(BaseModel):
    """
    Represents a single call to a language model.

    Attributes:
        request_id: Unique identifier for this request
        start_time: ISO 8601 timestamp when the request started
        end_time: ISO 8601 timestamp when the request completed
        messages: List of message dictionaries sent to the model
        usage: Token usage statistics for this call
        response_message: The response message dictionary from the model
    """

    request_id: str = Field(description="Unique identifier for this request")
    start_time: str = Field(description="ISO 8601 timestamp when request started")
    end_time: str = Field(description="ISO 8601 timestamp when request completed")
    messages: list[dict[str, Any]] = Field(
        description="List of message dictionaries sent to the model",
        default_factory=list,
    )
    usage: ModelCallUsage = Field(
        description="Token usage statistics for this call",
        default_factory=ModelCallUsage,
    )
    response_message: dict[str, Any] = Field(
        description="The response message dictionary from the model",
        default_factory=dict,
    )


class InferenceResult(RagBenchmarkEntry):
    answer: str
    context_ids: list[str] | None = None
    contexts: list[str] | None = None
    trajectory: Trajectory | None = None
    model_calls: list[ModelCall] | None = None
