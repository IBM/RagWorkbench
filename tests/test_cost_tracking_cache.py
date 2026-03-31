"""
Tests for cost tracking cache functionality.
"""

import tempfile
from unittest.mock import MagicMock, patch

from ragworkbench.boards.board_model import CacheMode
from ragworkbench.caching.tracking_key_cache import TrackingKeyCache
from ragworkbench.eval.cost_tracking import CostTracker


class TestTrackingKeyCache:
    """Test the TrackingKeyCache class."""

    def test_cache_initialization(self):
        """Test that cache can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TrackingKeyCache(
                cache_dir=tmpdir,
                cache_mode=CacheMode.ON,
            )
            assert cache is not None
            assert cache.cache_mode == CacheMode.ON

    def test_cache_add_and_get(self):
        """Test adding and retrieving from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TrackingKeyCache(
                cache_dir=tmpdir,
                cache_mode=CacheMode.ON,
            )

            experiment_id = "test_config_123"
            api_key = "sk-test-key-abc123"

            # Add to cache
            cache.add(experiment_id, api_key)

            # Retrieve from cache
            cached_key = cache.get(experiment_id)
            assert cached_key == api_key

    def test_cache_miss(self):
        """Test cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TrackingKeyCache(
                cache_dir=tmpdir,
                cache_mode=CacheMode.ON,
            )

            # Try to get non-existent key
            cached_key = cache.get("non_existent_config")
            assert cached_key is None


class TestCostTrackerWithCache:
    """Test CostTracker with caching enabled."""

    @patch.dict("os.environ", {"LITELLM_MASTER_KEY": "test-master-key"})
    def test_cost_tracker_with_cache_disabled(self):
        """Test that CostTracker works without cache."""
        tracker = CostTracker(
            enabled=True,
            cache_dir=None,  # No cache
        )
        assert tracker.cache is None

    @patch.dict("os.environ", {"LITELLM_MASTER_KEY": "test-master-key"})
    def test_cost_tracker_with_cache_enabled(self):
        """Test that CostTracker initializes cache when cache_dir is provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = CostTracker(
                enabled=True,
                cache_dir=tmpdir,
                cache_mode=CacheMode.ON,
            )
            assert tracker.cache is not None
            assert isinstance(tracker.cache, TrackingKeyCache)

    @patch.dict("os.environ", {"LITELLM_MASTER_KEY": "test-master-key"})
    @patch("httpx.Client")
    def test_generate_tracking_key_uses_cache(self, mock_client):
        """Test that generate_tracking_key uses cache on second call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"key": "sk-generated-key-123"}
            mock_client.return_value.__enter__.return_value.post.return_value = (
                mock_response
            )

            experiment_id = "test_config_456"

            # First tracker - should generate key
            tracker1 = CostTracker(
                enabled=True,
                cache_dir=tmpdir,
                cache_mode=CacheMode.ON,
            )
            key1 = tracker1.generate_tracking_key(experiment_id=experiment_id)
            assert key1 == "sk-generated-key-123"
            assert mock_client.return_value.__enter__.return_value.post.call_count == 1

            # Second tracker with same experiment_id - should use cache
            tracker2 = CostTracker(
                enabled=True,
                cache_dir=tmpdir,
                cache_mode=CacheMode.ON,
            )
            key2 = tracker2.generate_tracking_key(experiment_id=experiment_id)
            assert key2 == "sk-generated-key-123"
            # Should still be 1 because second call used cache
            assert mock_client.return_value.__enter__.return_value.post.call_count == 1
