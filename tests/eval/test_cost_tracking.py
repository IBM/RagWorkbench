"""Tests for cost tracking functionality."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ragworkbench.api.inference_result import ModelCall, ModelCallUsage
from ragworkbench.eval.cost_tracking import (
    AggregatedUsageData,
    CostTracker,
    ModelUsageData,
)


@pytest.fixture
def sample_log_entries():
    """Shared fixture for sample log entries used across tests."""
    return [
        {
            "request_id": "req-1",
            "startTime": "2026-04-06T10:00:00.000Z",
            "endTime": "2026-04-06T10:00:05.000Z",
            "model": "gpt-4",
            "spend": 0.001,
            "total_tokens": 100,
            "prompt_tokens": 80,
            "completion_tokens": 20,
            "metadata": {
                "cost_breakdown": {
                    "input_cost": 0.0008,
                    "output_cost": 0.0002,
                    "total_cost": 0.001,
                }
            },
            "proxy_server_request": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the capital of France?"},
                ]
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The capital of France is Paris.",
                        }
                    }
                ]
            },
        },
        {
            "request_id": "req-2",
            "startTime": "2026-04-06T10:00:30.000Z",
            "endTime": "2026-04-06T10:00:40.000Z",
            "model": "gpt-4",
            "spend": 0.005,
            "total_tokens": 500,
            "prompt_tokens": 100,
            "completion_tokens": 400,
            "usage": {"reasoning_tokens": 200},
            "metadata": {
                "cost_breakdown": {
                    "input_cost": 0.001,
                    "output_cost": 0.004,
                    "total_cost": 0.005,
                }
            },
            "proxy_server_request": {
                "messages": [
                    {"role": "user", "content": "Solve this problem step by step."},
                ]
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Here's the solution...",
                        }
                    }
                ]
            },
        },
    ]


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
        key = tracker.generate_tracking_key(experiment_id="test_config")
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

                key = tracker.generate_tracking_key(experiment_id="test_config_123")

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
                        "experiment_id": "test_config_123",
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
                    tracker.generate_tracking_key(experiment_id="test_config")

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
                    tracker.generate_tracking_key(experiment_id="test_config")

    @pytest.mark.asyncio
    async def test_get_usage_data_disabled(self):
        """Test that get_usage_data returns empty AggregatedUsageData when disabled."""
        tracker = CostTracker(enabled=False)
        result = await tracker.get_usage_data()
        assert isinstance(result, AggregatedUsageData)
        assert result.api_key == ""
        assert result.total_cost == 0.0
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_get_usage_data_no_api_key(self):
        """Test that get_usage_data returns empty AggregatedUsageData when no API key is set."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True)
            result = await tracker.get_usage_data()
            assert isinstance(result, AggregatedUsageData)
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

        # Verify per-model usage data
        assert len(result.per_model_usage) == 2
        assert "gpt-3.5-turbo" in result.per_model_usage
        assert "gpt-4" in result.per_model_usage

        # Check gpt-3.5-turbo model data
        gpt35_data = result.per_model_usage["gpt-3.5-turbo"]
        assert isinstance(gpt35_data, ModelUsageData)
        assert gpt35_data.model == "gpt-3.5-turbo"
        assert gpt35_data.total_cost == 0.001
        assert gpt35_data.total_tokens == 100
        assert gpt35_data.prompt_tokens == 50
        assert gpt35_data.completion_tokens == 50
        assert gpt35_data.requests == 1

        # Check gpt-4 model data
        gpt4_data = result.per_model_usage["gpt-4"]
        assert isinstance(gpt4_data, ModelUsageData)
        assert gpt4_data.model == "gpt-4"
        assert gpt4_data.total_cost == 0.002
        assert gpt4_data.total_tokens == 200
        assert gpt4_data.prompt_tokens == 100
        assert gpt4_data.completion_tokens == 100
        assert gpt4_data.requests == 1

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

    def test_get_cost_data_no_data(self):
        """Test get_cost_data when no data has been retrieved."""
        with patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-test-master-key"}):
            tracker = CostTracker(enabled=True)
            result = tracker.get_cost_data()
            assert isinstance(result, AggregatedUsageData)
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


class TestAggregatedUsageData:
    """Test suite for AggregatedUsageData class."""

    def test_create_from_log_entries_empty(self):
        """Test creating AggregatedUsageData from empty log entries."""
        result = AggregatedUsageData.create_from_log_entries("sk-track-test123", [])

        assert result.api_key == "sk-track-test123"
        assert result.total_cost == 0.0
        assert result.total_tokens == 0
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.requests == 0
        assert result.models_used == []
        assert result.per_model_usage == {}

    def test_extract_query_from_log_entry(self):
        """Test extracting query from log entry with messages."""
        # Test with valid log entry containing messages
        log_entry = {
            "model": "gpt-4",
            "proxy_server_request": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "How old was Southwest Airlines's president in 2018?",
                    },
                ],
            },
        }

        query = AggregatedUsageData._extract_query_from_log_entry(log_entry)
        assert query == "How old was Southwest Airlines's president in 2018?"

    def test_extract_query_from_log_entry_no_user_message(self):
        """Test extracting query when no user message exists."""
        log_entry = {
            "model": "gpt-4",
            "proxy_server_request": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                ],
            },
        }

        query = AggregatedUsageData._extract_query_from_log_entry(log_entry)
        assert query == "unknown query"

    def test_extract_query_from_log_entry_no_messages(self):
        """Test extracting query when messages field is missing."""
        log_entry = {"model": "gpt-4"}

        query = AggregatedUsageData._extract_query_from_log_entry(log_entry)
        assert query == "unknown query"

    def test_parse_usage_response_with_model_calls(self):
        """Test that model_calls are populated correctly."""
        # Mock log entries with messages
        log_entries = [
            {
                "model": "gpt-4",
                "spend": 0.001,
                "total_tokens": 100,
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "proxy_server_request": {
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"},
                    ],
                },
            },
            {
                "model": "gpt-4",
                "spend": 0.002,
                "total_tokens": 150,
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "proxy_server_request": {
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"},
                    ],
                },
            },
            {
                "model": "gpt-4",
                "spend": 0.0015,
                "total_tokens": 120,
                "prompt_tokens": 90,
                "completion_tokens": 30,
                "proxy_server_request": {
                    "messages": [
                        {"role": "user", "content": "What is 2+2?"},
                    ],
                },
            },
        ]

        result = AggregatedUsageData.create_from_log_entries("test-key", log_entries)

        # Verify query_to_log_entries structure
        assert "gpt-4" in result.per_model_usage
        model_data = result.per_model_usage["gpt-4"]

        # Check that query_to_log_entries is a dictionary
        assert isinstance(model_data.query_to_log_entries, dict)

        # Check that we have two queries
        assert len(model_data.query_to_log_entries) == 2

        # Check that the first query has 2 log entries
        assert "What is the capital of France?" in model_data.query_to_log_entries
        assert (
            len(model_data.query_to_log_entries["What is the capital of France?"]) == 2
        )

        # Check that the second query has 1 log entry
        assert "What is 2+2?" in model_data.query_to_log_entries
        assert len(model_data.query_to_log_entries["What is 2+2?"]) == 1

        # Verify the log entries are stored correctly
        first_query_logs = model_data.query_to_log_entries[
            "What is the capital of France?"
        ]
        assert first_query_logs[0]["spend"] == 0.001
        assert first_query_logs[1]["spend"] == 0.002

    def test_model_usage_data_query_to_log_entries_default(self):
        """Test that query_to_log_entries has a default empty dict."""
        model_data = ModelUsageData(model="gpt-4")
        assert model_data.query_to_log_entries == {}
        assert isinstance(model_data.query_to_log_entries, dict)

    def test_log_entry_to_model_call(self, sample_log_entries):
        """Test converting a log entry to a ModelCall object."""
        log_entry = sample_log_entries[0]
        model_call = AggregatedUsageData._log_entry_to_model_call(log_entry)

        assert isinstance(model_call, ModelCall)
        assert model_call.request_id == "req-1"
        assert model_call.start_time == "2026-04-06T10:00:00.000Z"
        assert model_call.end_time == "2026-04-06T10:00:05.000Z"
        assert len(model_call.messages) == 2
        assert model_call.messages[0]["role"] == "system"
        assert model_call.messages[1]["role"] == "user"
        assert model_call.usage.total_tokens == 100
        assert model_call.usage.prompt_tokens == 80
        assert model_call.usage.completion_tokens == 20
        assert model_call.usage.reasoning_tokens == 0
        assert model_call.usage.input_cost == 0.0008
        assert model_call.usage.output_cost == 0.0002
        assert model_call.usage.total_cost == 0.001
        assert model_call.response_message["role"] == "assistant"
        assert (
            model_call.response_message["content"] == "The capital of France is Paris."
        )

    def test_log_entry_to_model_call_with_reasoning_tokens(self, sample_log_entries):
        """Test converting a log entry with reasoning tokens to a ModelCall object."""
        log_entry = sample_log_entries[1]  # Entry with reasoning tokens
        model_call = AggregatedUsageData._log_entry_to_model_call(log_entry)

        assert model_call.usage.reasoning_tokens == 200
        assert model_call.usage.total_tokens == 500
        assert model_call.usage.input_cost == 0.001
        assert model_call.usage.output_cost == 0.004
        assert model_call.usage.total_cost == 0.005
        assert model_call.request_id == "req-2"

    def test_get_model_calls_for_query(self, sample_log_entries):
        """Test retrieving model calls for a specific query."""
        # Use only the first entry
        log_entries = [sample_log_entries[0]]

        # Parse the log entries to populate cost data
        usage_data = AggregatedUsageData.create_from_log_entries(
            "test-key", log_entries
        )

        # Get model calls for the query
        model_calls = usage_data.get_model_calls_for_query(
            "What is the capital of France?"
        )

        assert len(model_calls) == 1
        assert all(isinstance(call, ModelCall) for call in model_calls)
        assert model_calls[0].request_id == "req-1"

    def test_get_model_calls_for_query_no_data(self):
        """Test retrieving model calls when no cost data exists."""
        usage_data = AggregatedUsageData()

        model_calls = usage_data.get_model_calls_for_query("Some query")

        assert model_calls == []

    def test_get_model_calls_for_query_not_found(self, sample_log_entries):
        """Test retrieving model calls for a query that doesn't exist."""
        # Use only the second entry with a different query
        log_entries = [sample_log_entries[1]]

        usage_data = AggregatedUsageData.create_from_log_entries(
            "test-key", log_entries
        )

        # Query for a different question
        model_calls = usage_data.get_model_calls_for_query(
            "What is the capital of France?"
        )

        assert model_calls == []

    def test_log_entry_to_model_call_missing_cost_breakdown(self):
        """Test converting a log entry without cost breakdown to ModelCall object."""
        log_entry = {
            "request_id": "req-3",
            "startTime": "2026-04-06T10:00:00.000Z",
            "endTime": "2026-04-06T10:00:05.000Z",
            "model": "gpt-3.5-turbo",
            "spend": 0.0005,
            "total_tokens": 50,
            "prompt_tokens": 30,
            "completion_tokens": 20,
            "proxy_server_request": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                ]
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hi there!",
                        }
                    }
                ]
            },
        }

        model_call = AggregatedUsageData._log_entry_to_model_call(log_entry)

        # Verify cost fields default to 0.0 when metadata is missing
        assert model_call.usage.input_cost == 0.0
        assert model_call.usage.output_cost == 0.0
        assert model_call.usage.total_cost == 0.0
        # Verify other fields are still populated correctly
        assert model_call.usage.total_tokens == 50
        assert model_call.usage.prompt_tokens == 30
        assert model_call.usage.completion_tokens == 20


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
        api_key = tracker.generate_tracking_key(experiment_id="integration_test_config")
        assert api_key is not None
        assert api_key.startswith("sk-")
        print(f"Generated API key: {api_key[:20]}...")

        # Define the query to use for testing
        expected_query = "Say 'test' in one word"

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
                    "messages": [{"role": "user", "content": expected_query}],
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

        print("Waiting 60 seconds for LiteLLM proxy to write usage logs to database...")
        await asyncio.sleep(60)

        # Step 3: Retrieve usage data
        usage_data = await tracker.get_usage_data()

        # Verify usage data
        assert isinstance(
            usage_data, AggregatedUsageData
        ), "Expected AggregatedUsageData instance"
        assert usage_data.api_key == api_key
        assert usage_data.total_cost >= 0, "Expected non-negative cost"
        assert usage_data.total_tokens >= 0, "Expected non-negative tokens"
        assert usage_data.requests >= 0, "Expected non-negative request count"
        assert isinstance(
            usage_data.models_used, list
        ), "Expected models_used to be a list"
        assert isinstance(
            usage_data.per_model_usage, dict
        ), "Expected per_model_usage to be a dict"

        print("Usage data retrieved successfully:")
        print(f"  Total cost: ${usage_data.total_cost:.6f}")
        print(f"  Total tokens: {usage_data.total_tokens}")
        print(f"  Requests: {usage_data.requests}")
        print(f"  Models used: {usage_data.models_used}")

        # Print per-model breakdown
        if usage_data.per_model_usage:
            print("\n  Per-model breakdown:")
            for model, model_data in usage_data.per_model_usage.items():
                print(f"    {model}:")
                print(f"      Cost: ${model_data.total_cost:.6f}")
                print(f"      Tokens: {model_data.total_tokens}")
                print(f"      Requests: {model_data.requests}")

                # Verify query_to_log_entries exist and are properly structured
                assert isinstance(
                    model_data.query_to_log_entries, dict
                ), f"Expected query_to_log_entries to be a dict for model {model}"
                assert (
                    len(model_data.query_to_log_entries) > 0
                ), f"Expected at least one query in query_to_log_entries for model {model}"

                # Verify that the expected query is present
                assert expected_query in model_data.query_to_log_entries, (
                    f"Expected query '{expected_query}' not found in query_to_log_entries. "
                    f"Available queries: {list(model_data.query_to_log_entries.keys())}"
                )

                # Print query_to_log_entries breakdown
                print("      Model calls by query:")
                for query, log_entries in model_data.query_to_log_entries.items():
                    print(
                        f"        Query: '{query[:50]}...' - {len(log_entries)} call(s)"
                    )
                    assert isinstance(
                        log_entries, list
                    ), f"Expected log_entries to be a list for query '{query}'"
                    assert (
                        len(log_entries) > 0
                    ), f"Expected at least one log entry for query '{query}'"

        # Step 4: Test conversion to ModelCall objects using expected_query
        model_calls = usage_data.get_model_calls_for_query(expected_query)

        # Verify ModelCall objects were created
        assert isinstance(
            model_calls, list
        ), "Expected get_model_calls_for_query to return a list"
        assert (
            len(model_calls) > 0
        ), f"Expected at least one ModelCall for query '{expected_query}'"

        print(f"\n  ModelCall objects for query '{expected_query}':")
        for i, model_call in enumerate(model_calls):
            print(f"    Call {i + 1}:")
            print(f"      Request ID: {model_call.request_id}")
            print(f"      Start time: {model_call.start_time}")
            print(f"      End time: {model_call.end_time}")
            print(f"      Messages: {len(model_call.messages)} message(s)")
            print(f"      Response message: {model_call.response_message}")
            if model_call.usage:
                print(f"      Total tokens: {model_call.usage.total_tokens}")
                print(f"      Prompt tokens: {model_call.usage.prompt_tokens}")
                print(f"      Completion tokens: {model_call.usage.completion_tokens}")

            # Verify ModelCall structure
            assert model_call.request_id is not None, "Expected request_id to be set"
            assert model_call.start_time is not None, "Expected start_time to be set"
            assert model_call.end_time is not None, "Expected end_time to be set"
            assert (
                model_call.response_message is not None
            ), "Expected response_message to be set"
            assert isinstance(
                model_call.usage, ModelCallUsage
            ), "Expected usage to be ModelCallUsage instance"
            assert (
                model_call.usage.total_tokens >= 0
            ), "Expected non-negative total_tokens"

        # Verify cached data
        cached_data = tracker.get_cost_data()
        assert cached_data == usage_data
