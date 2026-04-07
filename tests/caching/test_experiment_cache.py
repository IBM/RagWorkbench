"""
Tests for ExperimentCache.

Focuses on ExperimentCache-specific functionality:
- ExperimentResult serialization/deserialization
- Experiment ID as cache key
- Integration with ExperimentResult model
"""

import json
import tempfile
from pathlib import Path

import pytest

from ragworkbench.api.experiment_result import ExperimentResult
from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.boards.board_model import CacheMode
from ragworkbench.caching.abstract_file_system_cache import AbstractFileSystemCache
from ragworkbench.caching.experiment_cache import ExperimentCache
from ragworkbench.eval.cost_tracking import AggregatedUsageData


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def cache_instance(temp_cache_dir):
    """Create ExperimentCache instance."""
    AbstractFileSystemCache.cache_path_to_contents.clear()
    return ExperimentCache(
        cache_dir=temp_cache_dir,
        cache_mode=CacheMode.ON,
    )


@pytest.fixture
def sample_experiment_result():
    """Create a sample ExperimentResult."""
    inference_result = InferenceResult(
        question_id="q1",
        question="What is the capital of France?",
        answer="Paris",
        ground_truth_answers=["Paris"],
        is_answerable=True,
    )

    return ExperimentResult(
        experiment_id="exp_001",
        config_name="test_config",
        dataset_name="test_dataset",
        inference_results=[inference_result],
        evaluation_results={
            "accuracy": {
                "per_question": {"q1": {"score": 1.0}},
                "statistics": {"mean": 1.0, "std": 0.0},
            }
        },
        cost_data=AggregatedUsageData(
            total_cost=0.05,
            total_tokens=100,
            prompt_tokens=50,
            completion_tokens=50,
            requests=1,
        ),
    )


class TestExperimentCacheSpecific:
    """Tests specific to ExperimentCache functionality."""

    def test_serialization_of_experiment_result(
        self, cache_instance, sample_experiment_result
    ):
        """Test that ExperimentResult is correctly serialized to JSON."""
        json_str = cache_instance._content_to_json(sample_experiment_result)
        data = json.loads(json_str)

        # Verify ExperimentResult structure
        assert "experiment_id" in data
        assert "inference_results" in data
        assert "evaluation_results" in data
        assert "cost_data" in data
        assert data["experiment_id"] == "exp_001"

    def test_deserialization_of_experiment_result(
        self, temp_cache_dir, cache_instance, sample_experiment_result
    ):
        """Test that JSON is correctly deserialized to ExperimentResult."""
        json_str = cache_instance._content_to_json(sample_experiment_result)
        json_file = temp_cache_dir / "test_exp.json"
        json_file.write_text(json_str, encoding="utf-8")

        loaded_result = cache_instance._read_content(json_file)

        assert isinstance(loaded_result, ExperimentResult)
        assert loaded_result.experiment_id == sample_experiment_result.experiment_id
        assert len(loaded_result.inference_results) == 1
        assert loaded_result.cost_data.total_cost == 0.05

    def test_experiment_id_as_cache_key(self, cache_instance, sample_experiment_result):
        """Test that experiment_id is used as the cache key."""
        experiment_id = "exp_test_123"
        sample_experiment_result.experiment_id = experiment_id

        cache_instance.add(experiment_id, sample_experiment_result)
        retrieved = cache_instance.get(experiment_id)

        assert retrieved is not None
        assert retrieved.experiment_id == experiment_id

    def test_round_trip_with_full_experiment_result(
        self, cache_instance, sample_experiment_result
    ):
        """Test complete round-trip preserves all ExperimentResult fields."""
        experiment_id = sample_experiment_result.experiment_id

        cache_instance.add(experiment_id, sample_experiment_result)
        loaded = cache_instance.get(experiment_id)

        # Verify all major fields
        assert loaded.experiment_id == sample_experiment_result.experiment_id
        assert loaded.config_name == sample_experiment_result.config_name
        assert loaded.dataset_name == sample_experiment_result.dataset_name
        assert len(loaded.inference_results) == len(
            sample_experiment_result.inference_results
        )
        assert loaded.evaluation_results == sample_experiment_result.evaluation_results
        assert (
            loaded.cost_data.total_cost == sample_experiment_result.cost_data.total_cost
        )

    def test_multiple_experiments_cached(self, cache_instance):
        """Test caching multiple different experiments."""
        experiments = []
        for i in range(3):
            exp = ExperimentResult(
                experiment_id=f"exp_{i}",
                inference_results=[],
                evaluation_results={},
            )
            experiments.append(exp)
            cache_instance.add(exp.experiment_id, exp)

        # Verify all can be retrieved
        for exp in experiments:
            retrieved = cache_instance.get(exp.experiment_id)
            assert retrieved is not None
            assert retrieved.experiment_id == exp.experiment_id

    def test_experiment_result_with_complex_evaluation_results(self, cache_instance):
        """Test caching ExperimentResult with complex evaluation data."""
        exp_result = ExperimentResult(
            experiment_id="exp_complex",
            inference_results=[],
            evaluation_results={
                "metric1": {
                    "per_question": {"q1": {"score": 0.9}, "q2": {"score": 0.8}},
                    "statistics": {"mean": 0.85, "std": 0.05},
                },
                "metric2": {
                    "per_question": {"q1": {"score": 0.95}, "q2": {"score": 0.75}},
                    "statistics": {"mean": 0.85, "std": 0.1},
                },
            },
        )

        cache_instance.add(exp_result.experiment_id, exp_result)
        retrieved = cache_instance.get(exp_result.experiment_id)

        assert retrieved is not None
        assert "metric1" in retrieved.evaluation_results
        assert "metric2" in retrieved.evaluation_results
        assert retrieved.evaluation_results["metric1"]["statistics"]["mean"] == 0.85
