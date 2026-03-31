"""
Cost tracking module for LiteLLM API usage.

This module provides functionality to track API costs during experiment runs
by generating unique API keys and querying usage statistics from LiteLLM proxy.
"""

import logging
import os
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field

from ragworkbench.boards.board_model import CacheMode
from ragworkbench.caching.tracking_key_cache import TrackingKeyCache

logger = logging.getLogger(__name__)


class UsageData(BaseModel):
    """
    Base data model for API usage statistics.

    This model contains the core usage and cost metrics that are common
    to both per-model and aggregated usage tracking.

    Attributes:
        total_cost: Total cost in USD
        total_tokens: Total number of tokens used (prompt + completion)
        prompt_tokens: Number of tokens in prompts
        completion_tokens: Number of tokens in completions
        requests: Number of API requests made
    """

    total_cost: float = Field(default=0.0, description="Total cost in USD", ge=0.0)
    total_tokens: int = Field(default=0, description="Total tokens used", ge=0)
    prompt_tokens: int = Field(default=0, description="Prompt tokens used", ge=0)
    completion_tokens: int = Field(
        default=0, description="Completion tokens used", ge=0
    )
    requests: int = Field(default=0, description="Number of requests made", ge=0)


class ModelUsageData(UsageData):
    """
    Data model for API usage statistics for a single model.

    Extends UsageData with model identification to track usage
    for a specific model independently.

    Attributes:
        model: The model name/identifier
        (inherits all UsageData fields)
    """

    model: str = Field(description="Model name/identifier")


class AggregatedUsageData(UsageData):
    """
    Data model for aggregated API usage statistics across all models.

    Extends UsageData with tracking information and per-model breakdowns.
    This is the main data structure returned by the cost tracker.

    Attributes:
        api_key: The API key used for tracking this session
        models_used: Sorted list of model names used during the session
        per_model_usage: Dictionary mapping model names to their individual usage data
        (inherits all UsageData fields for aggregate totals)
    """

    api_key: str = Field(default="", description="The API key used for tracking")
    models_used: list[str] = Field(
        description="Sorted list of models used", default_factory=list
    )
    per_model_usage: dict[str, ModelUsageData] = Field(
        description="Per-model usage breakdown",
        default_factory=dict,
    )

    def log_summary(self, prefix: str = "") -> None:
        """
        Log a summary of the usage data including aggregate and per-model breakdowns.

        Args:
            prefix: Optional prefix to add before each log line (e.g., experiment ID)
        """
        if not self.total_cost and not self.total_tokens:
            return  # Nothing to log if no usage data

        # Log aggregate summary
        logger.info(
            f"{prefix}Total cost: ${self.total_cost:.4f}, "
            f"Total tokens: {self.total_tokens}"
        )

        # Log per-model breakdown if available
        if self.per_model_usage:
            logger.info(f"{prefix}Per-model breakdown:")
            for model, model_data in self.per_model_usage.items():
                logger.info(
                    f"{prefix}  {model}: ${model_data.total_cost:.4f}, "
                    f"{model_data.total_tokens} tokens, "
                    f"{model_data.requests} requests"
                )


class CostTracker:
    """
    Tracks API costs for a single configuration run using LiteLLM proxy.

    This class manages API key generation and cost retrieval for tracking
    the cost of API calls made during an experiment configuration.
    """

    def __init__(
        self,
        enabled: bool = False,
        litellm_proxy_url: str = "http://localhost:4000",
        cache_dir: Path | str | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ):
        """
        Initialize the cost tracker.

        Args:
            enabled: Whether cost tracking is enabled. If False, all operations are no-ops.
            litellm_proxy_url: Base URL for the LiteLLM proxy server
            cache_dir: Directory for caching tracking keys. If None, caching is disabled.
            cache_mode: Cache operation mode (on/off/refresh)

        Raises:
            ValueError: If enabled=True but LITELLM_MASTER_KEY environment variable is not set
        """
        self.enabled = enabled
        self.litellm_proxy_url = litellm_proxy_url.rstrip("/")

        # Read master key from environment variable
        self.litellm_master_key = os.getenv("LITELLM_MASTER_KEY")

        # Validate master key is set when tracking is enabled
        if self.enabled and not self.litellm_master_key:
            raise ValueError(
                "LITELLM_MASTER_KEY environment variable must be set when cost tracking is enabled. "
                "Please set the environment variable with your LiteLLM proxy master key."
            )

        self.api_key: str | None = None
        self._cost_data: AggregatedUsageData | None = None

        # Initialize cache if cache_dir is provided and cache_mode is not OFF
        self.cache: TrackingKeyCache | None = None
        if cache_dir is not None and cache_mode != CacheMode.OFF:
            self.cache = TrackingKeyCache(
                cache_dir=cache_dir,
                cache_mode=cache_mode,
            )

    def generate_tracking_key(self, experiment_id: str) -> str | None:
        """
        Generate a unique API key for this tracking session by calling LiteLLM proxy.
        Uses cache if available to avoid regenerating keys for the same session.

        Args:
            experiment_id: Unique experiment identifier for cache key.

        Returns:
            The generated API key, or None if tracking is disabled

        Raises:
            ValueError: If master key is not provided when tracking is enabled
            RuntimeError: If API key generation fails
        """
        if not self.enabled:
            return None

        # Try to get cached API key first
        if self.cache is not None:
            cached_key = self.cache.get(experiment_id)
            if cached_key:
                self.api_key = cached_key
                logger.info(
                    f"Using cached cost tracking API key for experiment {experiment_id}"
                )
                return self.api_key

        try:
            # Call LiteLLM proxy to generate a new API key
            import httpx

            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.litellm_proxy_url}/key/generate",
                    headers={
                        "Authorization": f"Bearer {self.litellm_master_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "metadata": {
                            "user": "ragworkbench",
                            "purpose": "cost_tracking",
                            "experiment_id": experiment_id,
                        },
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    self.api_key = data.get("key")

                    if not self.api_key:
                        raise RuntimeError(
                            "LiteLLM proxy returned success but no API key in response"
                        )

                    logger.info(
                        f"Generated cost tracking API key from LiteLLM for experiment {experiment_id}"
                    )

                    # Cache the generated API key
                    if self.cache is not None:
                        self.cache.add(experiment_id, self.api_key)

                    return self.api_key
                else:
                    error_msg = (
                        f"Failed to generate API key from LiteLLM proxy. "
                        f"Status: {response.status_code}, Response: {response.text}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"Failed to generate tracking API key: {e}")
            raise RuntimeError(f"Failed to generate tracking API key: {e}") from e

    async def get_usage_data(self) -> AggregatedUsageData:
        """
        Query LiteLLM proxy for usage statistics for the current API key.

        This method queries the LiteLLM proxy's /spend/logs endpoint using the master key
        to retrieve detailed usage data for the current tracking API key.

        Returns:
            AggregatedUsageData instance containing cost data with:
            - api_key: The API key used for tracking
            - total_cost: Total cost in USD (aggregate)
            - total_tokens: Total tokens used (aggregate)
            - prompt_tokens: Prompt tokens used (aggregate)
            - completion_tokens: Completion tokens used (aggregate)
            - requests: Number of requests made (aggregate)
            - models_used: List of models used
            - per_model_usage: Dictionary of per-model usage breakdowns

            Returns empty AggregatedUsageData instance if tracking is disabled or no API key is set.

        Raises:
            RuntimeError: If the HTTP request fails or if there's an error retrieving cost data.
        """
        if not self.enabled or not self.api_key:
            return AggregatedUsageData()

        try:
            # Query LiteLLM proxy for usage statistics using master key
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Query the spend/logs endpoint with master key authorization
                # and filter by the tracking API key
                headers = {"Authorization": f"Bearer {self.litellm_master_key}"}
                response = await client.get(
                    f"{self.litellm_proxy_url}/spend/logs",
                    headers=headers,
                    params={"api_key": self.api_key},
                )

                if response.status_code == 200:
                    data = response.json()

                    # Parse the response and aggregate usage data
                    cost_data = self._parse_usage_response(data)

                    logger.info(
                        f"Cost tracking complete. Total cost: ${cost_data.total_cost:.4f}, "
                        f"Total tokens: {cost_data.total_tokens}"
                    )

                    self._cost_data = cost_data
                    return cost_data
                else:
                    error_msg = (
                        f"Failed to retrieve usage data from LiteLLM proxy. "
                        f"Status: {response.status_code}, Response: {response.text}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

        except Exception as e:
            logger.error(f"Failed to retrieve cost data from LiteLLM proxy: {e}")
            raise RuntimeError(
                f"Failed to retrieve cost data from LiteLLM proxy: {e}"
            ) from e

    def _parse_usage_response(self, data: list[dict[str, Any]]) -> AggregatedUsageData:
        """
        Parse the usage response from LiteLLM proxy and aggregate statistics.

        This method processes log entries and creates both aggregate statistics
        and per-model breakdowns for detailed cost analysis.

        Args:
            data: Response data from LiteLLM proxy (list of log entries)

        Returns:
            AggregatedUsageData instance with aggregated usage statistics and per-model breakdowns
        """
        # Ensure api_key is not None before creating AggregatedUsageData
        if self.api_key is None:
            raise ValueError("API key must be set before parsing usage data")

        # Initialize result with empty per-model tracking
        result = AggregatedUsageData(api_key=self.api_key)

        # The /spend/logs endpoint returns a list of log entries directly
        logs = data if isinstance(data, list) else []

        for log_entry in logs:
            logger.debug(f"Processing log entry: {log_entry}")

            # Extract model name and usage data from log entry
            model = log_entry.get("model", "unknown")
            spend = float(log_entry.get("spend", 0))
            total_tok = int(log_entry.get("total_tokens", 0))
            prompt_tok = int(log_entry.get("prompt_tokens", 0))
            completion_tok = int(log_entry.get("completion_tokens", 0))

            # Aggregate totals
            result.total_cost += spend
            result.total_tokens += total_tok
            result.prompt_tokens += prompt_tok
            result.completion_tokens += completion_tok
            result.requests += 1

            # Initialize or update per-model data
            if model not in result.per_model_usage:
                result.per_model_usage[model] = ModelUsageData(model=model)

            model_data = result.per_model_usage[model]
            model_data.total_cost += spend
            model_data.total_tokens += total_tok
            model_data.prompt_tokens += prompt_tok
            model_data.completion_tokens += completion_tok
            model_data.requests += 1

        # Update models_used list
        result.models_used = sorted(result.per_model_usage.keys())

        return result

    def get_cost_data(self) -> AggregatedUsageData:
        """
        Get the cost data from the last tracking session.

        Returns:
            AggregatedUsageData instance containing cost data, or empty AggregatedUsageData if no data available
        """
        return self._cost_data or AggregatedUsageData()
