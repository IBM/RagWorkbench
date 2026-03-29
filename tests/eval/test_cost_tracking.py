"""Tests for cost tracking functionality."""

import os
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from ragworkbench.eval.cost_tracking import CostTracker, UsageData


class TestCostTracker:
    """Test suite for CostTracker class."""

    def test_init_disabled(self):
        """Test initialization with cost tracking disabled."""
        tracker = CostTracker(enabled=False)
        assert tracker.enabled is False
        assert tracker.api_key is None
        assert tracker._cost_data is None

    def test_init_enabled_with_master_key(self):
        """Test initialization with cost tracking enabled and master key set."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True, litellm_proxy_url="http://test:4000")
            assert tracker.enabled is True
            assert tracker.litellm_proxy_url == "http://test:4000"
            assert tracker.litellm_master_key == "sk-test-master-key"
            assert tracker.api_key is None

    def test_init_enabled_without_master_key(self):
        """Test initialization with cost tracking enabled but no master key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(
                ValueError, match="LITELLM_MASTER_KEY environment variable must be set"
            ):
                CostTracker(enabled=True, litellm_proxy_url="http://test:4000")

    def test_generate_tracking_key_disabled(self):
        """Test that generate_tracking_key returns None when disabled."""
        tracker = CostTracker(enabled=False)
        key = tracker.generate_tracking_key()
        assert key is None
        assert tracker.api_key is None

    def test_generate_tracking_key_enabled(self):
        """Test that generate_tracking_key calls LiteLLM API and returns a valid key."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True, litellm_proxy_url="http://test:4000")

            # Mock httpx.Client for the POST request
            with patch("ragworkbench.eval.cost_tracking.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"key": "sk-generated-api-key-123"}

                mock_client_instance = MagicMock()
                mock_client_instance.__enter__.return_value = mock_client_instance
                mock_client_instance.__exit__.return_value = None
                mock_client_instance.post.return_value = mock_response
                mock_client.return_value = mock_client_instance

                key = tracker.generate_tracking_key()

            assert key == "sk-generated-api-key-123"
            assert tracker.api_key == "sk-generated-api-key-123"

            # Verify the API was called correctly
            mock_client_instance.post.assert_called_once_with(
                "http://test:4000/key/generate",
                headers={
                    "Authorization": "Bearer sk-test-master-key",
                    "Content-Type": "application/json",
                },
                json={
                    "metadata": {
                        "user": "ragworkbench",
                        "purpose": "cost_tracking",
                        "session_id": ANY,  # UUID will vary
                    },
                },
            )

    def test_generate_tracking_key_api_error(self):
        """Test that generate_tracking_key raises RuntimeError on API error."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True, litellm_proxy_url="http://test:4000")

            # Mock httpx.Client for error response
            with patch("ragworkbench.eval.cost_tracking.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_response.text = "Internal Server Error"

                mock_client_instance = MagicMock()
                mock_client_instance.__enter__.return_value = mock_client_instance
                mock_client_instance.__exit__.return_value = None
                mock_client_instance.post.return_value = mock_response
                mock_client.return_value = mock_client_instance

                with pytest.raises(
                    RuntimeError, match="Failed to generate API key from LiteLLM proxy"
                ):
                    tracker.generate_tracking_key()

    def test_generate_tracking_key_no_key_in_response(self):
        """Test that generate_tracking_key raises RuntimeError when no key in response."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True, litellm_proxy_url="http://test:4000")

            # Mock httpx.Client for response without key
            with patch("ragworkbench.eval.cost_tracking.httpx.Client") as mock_client:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"message": "success"}  # No key field

                mock_client_instance = MagicMock()
                mock_client_instance.__enter__.return_value = mock_client_instance
                mock_client_instance.__exit__.return_value = None
                mock_client_instance.post.return_value = mock_response
                mock_client.return_value = mock_client_instance

                with pytest.raises(
                    RuntimeError,
                    match="LiteLLM proxy returned success but no API key in response",
                ):
                    tracker.generate_tracking_key()

    @pytest.mark.asyncio
    async def test_get_usage_data_disabled(self):
        """Test that get_usage_data returns empty UsageData when disabled."""
        tracker = CostTracker(enabled=False)
        result = await tracker.get_usage_data()
        assert isinstance(result, UsageData)
        assert result.api_key == ""
        assert result.total_cost == 0.0
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_get_usage_data_no_api_key(self):
        """Test that get_usage_data returns empty UsageData when no API key is set."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True)
            result = await tracker.get_usage_data()
            assert isinstance(result, UsageData)
            assert result.api_key == ""
            assert result.total_cost == 0.0
            assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_get_usage_data_success(self):
        """Test successful retrieval of usage data."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True, litellm_proxy_url="http://test:4000")
            tracker.api_key = "sk-track-test123"

        # Mock response data - LiteLLM returns a list directly
        mock_response_data = [
            {
                "spend": 0.001,
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
                "model": "gpt-3.5-turbo",
            },
            {
                "spend": 0.002,
                "total_tokens": 200,
                "prompt_tokens": 100,
                "completion_tokens": 100,
                "model": "gpt-4",
            },
        ]

        # Mock httpx.AsyncClient
        with patch("ragworkbench.eval.cost_tracking.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_client_instance

            result = await tracker.get_usage_data()

        # Verify the aggregated results
        assert result.api_key == "sk-track-test123"
        assert result.total_cost == 0.003
        assert result.total_tokens == 300
        assert result.prompt_tokens == 150
        assert result.completion_tokens == 150
        assert result.requests == 2
        assert set(result.models_used) == {"gpt-3.5-turbo", "gpt-4"}

    @pytest.mark.asyncio
    async def test_get_usage_data_http_error(self):
        """Test handling of HTTP errors when retrieving usage data."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True, litellm_proxy_url="http://test:4000")
            tracker.api_key = "sk-track-test123"

        # Mock httpx.AsyncClient with error response
        with patch("ragworkbench.eval.cost_tracking.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Internal Server Error"

            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_client_instance

            # Verify RuntimeError is raised
            with pytest.raises(RuntimeError, match="Failed to retrieve usage data"):
                await tracker.get_usage_data()

    @pytest.mark.asyncio
    async def test_get_usage_data_exception(self):
        """Test handling of exceptions when retrieving usage data."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True, litellm_proxy_url="http://test:4000")
            tracker.api_key = "sk-track-test123"

        # Mock httpx.AsyncClient to raise an exception
        with patch("ragworkbench.eval.cost_tracking.httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            mock_client.return_value = mock_client_instance

            # Verify RuntimeError is raised with the original exception
            with pytest.raises(RuntimeError, match="Failed to retrieve cost data"):
                await tracker.get_usage_data()

    def test_parse_usage_response_empty(self):
        """Test parsing of empty usage response."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True)
            tracker.api_key = "sk-track-test123"

        result = tracker._parse_usage_response([])

        assert result.api_key == "sk-track-test123"
        assert result.total_cost == 0.0
        assert result.total_tokens == 0
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.requests == 0
        assert result.models_used == []

    def test_get_cost_data_no_data(self):
        """Test get_cost_data when no data has been retrieved."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True)
            result = tracker.get_cost_data()
            assert isinstance(result, UsageData)
            assert result.api_key == ""
            assert result.total_cost == 0.0
            assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_get_cost_data_after_retrieval(self):
        """Test get_cost_data after successful data retrieval."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True, litellm_proxy_url="http://test:4000")
            tracker.api_key = "sk-track-test123"

        mock_response_data = [
            {
                "spend": 0.001,
                "total_tokens": 100,
                "prompt_tokens": 50,
                "completion_tokens": 50,
                "model": "gpt-3.5-turbo",
            }
        ]

        with patch("ragworkbench.eval.cost_tracking.httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_response_data

            mock_client_instance = AsyncMock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_client_instance

            await tracker.get_usage_data()

        # Now get_cost_data should return the cached data
        result = tracker.get_cost_data()
        assert result.total_cost == 0.001
        assert result.total_tokens == 100


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("LITELLM_MASTER_KEY") or not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Integration test requires LITELLM_MASTER_KEY and RUN_INTEGRATION_TESTS=1",
)
class TestCostTrackerIntegration:
    """Integration tests for CostTracker against a real LiteLLM proxy."""

    @pytest.mark.asyncio
    async def test_full_cost_tracking_workflow(self):
        """
        Integration test: Generate key, make API call, retrieve usage data.

        Prerequisites:
        - LiteLLM proxy running at http://localhost:4000
        - LITELLM_MASTER_KEY environment variable set
        - RUN_INTEGRATION_TESTS=1 environment variable set
        - At least one model configured in LiteLLM (e.g., gpt-3.5-turbo)
        """

        import httpx

        # Initialize tracker with real master key
        tracker = CostTracker(enabled=True, litellm_proxy_url="http://localhost:4000")

        # Step 1: Generate a real API key
        api_key = tracker.generate_tracking_key()
        assert api_key is not None
        assert api_key.startswith("sk-")
        print(f"Generated API key: {api_key[:20]}...")

        # Step 2: Make a real API call using the generated key
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:4000/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "claude-opus-4-6",
                    "messages": [{"role": "user", "content": "Say 'test' in one word"}],
                    "max_tokens": 5,
                },
            )

            # Debug: Print response details if not 200
            if response.status_code != 200:
                print(f"API call failed with status {response.status_code}")
                print(f"Response body: {response.text}")
                print(f"Response headers: {dict(response.headers)}")

            assert response.status_code == 200
            response_data = response.json()
            assert "choices" in response_data
            print(
                f"API call successful: {response_data['choices'][0]['message']['content']}"
            )

        # Wait for LiteLLM proxy to write usage logs to database
        import asyncio

        await asyncio.sleep(60)

        # Step 3: Retrieve usage data
        usage_data = await tracker.get_usage_data()

        # Verify usage data
        assert isinstance(usage_data, UsageData), "Expected UsageData instance"
        assert usage_data.api_key == api_key
        assert usage_data.total_cost >= 0, "Expected non-negative cost"
        assert usage_data.total_tokens >= 0, "Expected non-negative tokens"
        assert usage_data.requests >= 0, "Expected non-negative request count"
        assert isinstance(
            usage_data.models_used, list
        ), "Expected models_used to be a list"

        print("Usage data retrieved successfully:")
        print(f"  Total cost: ${usage_data.total_cost:.6f}")
        print(f"  Total tokens: {usage_data.total_tokens}")
        print(f"  Requests: {usage_data.requests}")
        print(f"  Models used: {usage_data.models_used}")

        # Verify cached data
        cached_data = tracker.get_cost_data()
        assert cached_data == usage_data


# Made with Bob
