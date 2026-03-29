"""
Cost tracking module for LiteLLM API usage.

This module provides functionality to track API costs during experiment runs
by generating unique API keys and querying usage statistics from LiteLLM proxy.
"""

import logging
import os
import uuid
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class UsageData(BaseModel):
    """
    Data model for API usage statistics from LiteLLM proxy.

    This model encapsulates all usage and cost information for a tracking session,
    including token counts, costs, and metadata about the API calls made.

    Attributes:
        api_key: The API key used for tracking this session
        total_cost: Total cost in USD for all API calls
        total_tokens: Total number of tokens used (prompt + completion)
        prompt_tokens: Number of tokens in prompts
        completion_tokens: Number of tokens in completions
        requests: Number of API requests made
        models_used: Sorted list of model names used during the session
    """

    api_key: str = Field(default="", description="The API key used for tracking")
    total_cost: float = Field(default=0.0, description="Total cost in USD", ge=0.0)
    total_tokens: int = Field(default=0, description="Total tokens used", ge=0)
    prompt_tokens: int = Field(default=0, description="Prompt tokens used", ge=0)
    completion_tokens: int = Field(
        default=0, description="Completion tokens used", ge=0
    )
    requests: int = Field(default=0, description="Number of requests made", ge=0)
    models_used: list[str] = Field(
        description="Sorted list of models used", default_factory=list
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
    ):
        """
        Initialize the cost tracker.

        Args:
            enabled: Whether cost tracking is enabled. If False, all operations are no-ops.
            litellm_proxy_url: Base URL for the LiteLLM proxy server

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
        self._cost_data: UsageData | None = None

    def generate_tracking_key(self) -> str | None:
        """
        Generate a unique API key for this tracking session by calling LiteLLM proxy.

        Returns:
            The generated API key, or None if tracking is disabled

        Raises:
            ValueError: If master key is not provided when tracking is enabled
            RuntimeError: If API key generation fails
        """
        if not self.enabled:
            return None

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
                            "session_id": uuid.uuid4().hex[:16],
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
                        f"Generated cost tracking API key from LiteLLM: {self.api_key[:20]}..."
                    )
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

    async def get_usage_data(self) -> UsageData:
        """
        Query LiteLLM proxy for usage statistics for the current API key.

        This method queries the LiteLLM proxy's /spend/logs endpoint using the master key
        to retrieve detailed usage data for the current tracking API key.

        Returns:
            UsageData instance containing cost data with:
            - api_key: The API key used for tracking
            - total_cost: Total cost in USD
            - total_tokens: Total tokens used
            - prompt_tokens: Prompt tokens used
            - completion_tokens: Completion tokens used
            - requests: Number of requests made
            - models_used: List of models used

            Returns empty UsageData instance if tracking is disabled or no API key is set.

        Raises:
            RuntimeError: If the HTTP request fails or if there's an error retrieving cost data.
        """
        if not self.enabled or not self.api_key:
            return UsageData()

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

    def _parse_usage_response(self, data: list[dict[str, Any]]) -> UsageData:
        """
        Parse the usage response from LiteLLM proxy and aggregate statistics.

        Args:
            data: Response data from LiteLLM proxy (list of log entries)

        Returns:
            UsageData instance with aggregated usage statistics
        """
        # Initialize aggregated data
        total_cost = 0.0
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        requests = 0
        models_used = set()

        # The /spend/logs endpoint returns a list of log entries directly
        logs = data if isinstance(data, list) else []

        for log_entry in logs:
            # Aggregate cost
            if "spend" in log_entry:
                total_cost += float(log_entry["spend"])

            # Aggregate tokens
            if "total_tokens" in log_entry:
                total_tokens += int(log_entry["total_tokens"])
            if "prompt_tokens" in log_entry:
                prompt_tokens += int(log_entry["prompt_tokens"])
            if "completion_tokens" in log_entry:
                completion_tokens += int(log_entry["completion_tokens"])

            # Track models used
            if "model" in log_entry:
                models_used.add(log_entry["model"])

            requests += 1

        # Ensure api_key is not None before creating UsageData
        if self.api_key is None:
            raise ValueError("API key must be set before parsing usage data")

        return UsageData(
            api_key=self.api_key,
            total_cost=total_cost,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            requests=requests,
            models_used=sorted(models_used),
        )

    def get_cost_data(self) -> UsageData:
        """
        Get the cost data from the last tracking session.

        Returns:
            UsageData instance containing cost data, or empty UsageData if no data available
        """
        return self._cost_data or UsageData()
