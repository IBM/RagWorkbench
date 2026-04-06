import pytest

from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.eval.evaluator import Evaluator
from ragworkbench.eval.metric_models import load_metric_definitions


class TestToolUseCountMetric:
    """Test the workbench.tool_use_count metric."""

    @pytest.fixture
    def metric_definition(self):
        """Load the tool_use_count metric definition."""
        config = load_metric_definitions()
        return config.get_metric_definition("workbench.tool_use_count")

    @pytest.fixture
    def sample_inference_results(self):
        """Create sample inference results with different trajectory lengths."""
        return [
            InferenceResult(
                question_id="q1",
                question="What is the capital of France?",
                answer="Paris",
                ground_truth_answers=["Paris"],
                ground_truths_context_ids=[],
                trajectory=[
                    {"tool": "search", "query": "capital of France"},
                    {"tool": "retrieve", "doc_id": "doc1"},
                ],
            ),
            InferenceResult(
                question_id="q2",
                question="What is 2+2?",
                answer="4",
                ground_truth_answers=["4"],
                ground_truths_context_ids=[],
                trajectory=[
                    {"tool": "calculator", "expression": "2+2"},
                ],
            ),
            InferenceResult(
                question_id="q3",
                question="Who wrote Hamlet?",
                answer="Shakespeare",
                ground_truth_answers=["Shakespeare", "William Shakespeare"],
                ground_truths_context_ids=[],
                trajectory=None,  # No trajectory
            ),
            InferenceResult(
                question_id="q4",
                question="What is the speed of light?",
                answer="299,792,458 m/s",
                ground_truth_answers=["299792458 m/s"],
                ground_truths_context_ids=[],
                trajectory=[],  # Empty trajectory
            ),
        ]

    def test_metric_definition_exists(self, metric_definition):
        """Test that the metric definition is properly loaded."""
        assert metric_definition is not None
        assert metric_definition.metric_name == "workbench.tool_use_count"
        assert metric_definition.metric_id == "workbench.tool_use_count"
        assert metric_definition.vendor == "workbench"
        assert "trajectory" in metric_definition.metric_fields

    def test_tool_use_count_computation(
        self, metric_definition, sample_inference_results
    ):
        """Test that tool use counts are computed correctly."""
        evaluator = Evaluator(metric_definition=metric_definition)

        # Run the metric
        results = evaluator.run_metrics(sample_inference_results)

        # Verify results
        assert "q1" in results
        assert "q2" in results
        assert "q3" in results
        assert "q4" in results

        # Check tool counts
        assert results["q1"]["workbench.tool_use_count"] == 2.0  # 2 tool uses
        assert results["q2"]["workbench.tool_use_count"] == 1.0  # 1 tool use
        assert results["q3"]["workbench.tool_use_count"] == 0.0  # None trajectory
        assert results["q4"]["workbench.tool_use_count"] == 0.0  # Empty trajectory

    def test_tool_use_count_statistics(
        self, metric_definition, sample_inference_results
    ):
        """Test that statistics are computed correctly for tool use counts."""
        evaluator = Evaluator(metric_definition=metric_definition)

        # Run the metric
        per_question_results = evaluator.run_metrics(sample_inference_results)

        # Compute statistics
        stats = evaluator.compute_stats_from_per_question_results(per_question_results)

        # Verify statistics exist
        assert "workbench.tool_use_count" in stats

        metric_stats = stats["workbench.tool_use_count"]

        # Check that all expected keys are present
        assert "mean" in metric_stats
        assert "ci_low" in metric_stats
        assert "ci_high" in metric_stats
        assert "coverage" in metric_stats

        # Verify mean is correct: (2 + 1 + 0 + 0) / 4 = 0.75
        assert metric_stats["mean"] == 0.75

        # Coverage should be 1.0 (all values are finite)
        assert metric_stats["coverage"] == 1.0

    def test_empty_inference_results(self, metric_definition):
        """Test behavior with empty inference results list."""
        evaluator = Evaluator(metric_definition=metric_definition)

        results = evaluator.run_metrics([])

        assert results == {}

    def test_single_inference_result(self, metric_definition):
        """Test with a single inference result."""
        evaluator = Evaluator(metric_definition=metric_definition)

        single_result = [
            InferenceResult(
                question_id="q1",
                question="Test question",
                answer="Test answer",
                ground_truth_answers=["Test"],
                ground_truths_context_ids=[],
                trajectory=[
                    {"tool": "tool1"},
                    {"tool": "tool2"},
                    {"tool": "tool3"},
                ],
            )
        ]

        results = evaluator.run_metrics(single_result)

        assert "q1" in results
        assert results["q1"]["workbench.tool_use_count"] == 3.0


class TestUsageMetric:
    """Test the workbench.usage metric."""

    @pytest.fixture
    def metric_definition(self):
        """Load the usage metric definition."""
        config = load_metric_definitions()
        return config.get_metric_definition("workbench.usage")

    @pytest.fixture
    def sample_inference_results_with_tokens(self):
        """Create sample inference results with model calls and token usage."""
        from ragworkbench.api.inference_result import ModelCall, ModelCallUsage

        return [
            InferenceResult(
                question_id="q1",
                question="What is the capital of France?",
                answer="Paris",
                ground_truth_answers=["Paris"],
                ground_truths_context_ids=[],
                model_calls=[
                    ModelCall(
                        request_id="req1",
                        start_time="2024-01-01T00:00:00Z",
                        end_time="2024-01-01T00:00:01Z",
                        messages=[],
                        usage=ModelCallUsage(
                            total_tokens=100,
                            prompt_tokens=60,
                            completion_tokens=40,
                        ),
                    ),
                    ModelCall(
                        request_id="req2",
                        start_time="2024-01-01T00:00:02Z",
                        end_time="2024-01-01T00:00:03Z",
                        messages=[],
                        usage=ModelCallUsage(
                            total_tokens=150,
                            prompt_tokens=90,
                            completion_tokens=60,
                        ),
                    ),
                ],
            ),
            InferenceResult(
                question_id="q2",
                question="What is 2+2?",
                answer="4",
                ground_truth_answers=["4"],
                ground_truths_context_ids=[],
                model_calls=[
                    ModelCall(
                        request_id="req3",
                        start_time="2024-01-01T00:00:04Z",
                        end_time="2024-01-01T00:00:05Z",
                        messages=[],
                        usage=ModelCallUsage(
                            total_tokens=50,
                            prompt_tokens=30,
                            completion_tokens=20,
                        ),
                    ),
                ],
            ),
            InferenceResult(
                question_id="q3",
                question="Who wrote Hamlet?",
                answer="Shakespeare",
                ground_truth_answers=["Shakespeare"],
                ground_truths_context_ids=[],
                model_calls=None,  # No model calls
            ),
            InferenceResult(
                question_id="q4",
                question="What is the speed of light?",
                answer="299,792,458 m/s",
                ground_truth_answers=["299792458 m/s"],
                ground_truths_context_ids=[],
                model_calls=[],  # Empty model calls
            ),
        ]

    def test_metric_definition_exists(self, metric_definition):
        """Test that the metric definition is properly loaded."""
        assert metric_definition is not None
        assert metric_definition.metric_name == "workbench.usage"
        assert metric_definition.metric_id == "workbench.usage"
        assert metric_definition.vendor == "workbench"
        assert "model_calls" in metric_definition.metric_fields
        assert "sub_scores" in metric_definition.metric_params
        assert metric_definition.metric_params["sub_scores"] == [
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
        ]

    def test_token_counts_computation(
        self, metric_definition, sample_inference_results_with_tokens
    ):
        """Test that token counts are computed correctly."""
        evaluator = Evaluator(metric_definition=metric_definition)

        # Run the metric
        results = evaluator.run_metrics(sample_inference_results_with_tokens)

        # Verify results exist for all questions
        assert "q1" in results
        assert "q2" in results
        assert "q3" in results
        assert "q4" in results

        # Check q1: 2 model calls (100+150=250 total, 60+90=150 prompt, 40+60=100 completion)
        assert results["q1"]["total_tokens"] == 250.0
        assert results["q1"]["prompt_tokens"] == 150.0
        assert results["q1"]["completion_tokens"] == 100.0

        # Check q2: 1 model call (50 total, 30 prompt, 20 completion)
        assert results["q2"]["total_tokens"] == 50.0
        assert results["q2"]["prompt_tokens"] == 30.0
        assert results["q2"]["completion_tokens"] == 20.0

        # Check q3: No model calls
        assert results["q3"]["total_tokens"] == 0.0
        assert results["q3"]["prompt_tokens"] == 0.0
        assert results["q3"]["completion_tokens"] == 0.0

        # Check q4: Empty model calls
        assert results["q4"]["total_tokens"] == 0.0
        assert results["q4"]["prompt_tokens"] == 0.0
        assert results["q4"]["completion_tokens"] == 0.0

    def test_token_counts_statistics(
        self, metric_definition, sample_inference_results_with_tokens
    ):
        """Test that statistics are computed correctly for token counts."""
        evaluator = Evaluator(metric_definition=metric_definition)

        # Run the metric
        per_question_results = evaluator.run_metrics(
            sample_inference_results_with_tokens
        )

        # Compute statistics
        stats = evaluator.compute_stats_from_per_question_results(per_question_results)

        # Verify statistics exist for all sub-scores (prefixed with metric name)
        assert "workbench.usage.total_tokens" in stats
        assert "workbench.usage.prompt_tokens" in stats
        assert "workbench.usage.completion_tokens" in stats

        # Check total_tokens statistics: (250 + 50 + 0 + 0) / 4 = 75.0
        assert stats["workbench.usage.total_tokens"]["mean"] == 75.0
        assert stats["workbench.usage.total_tokens"]["coverage"] == 1.0

        # Check prompt_tokens statistics: (150 + 30 + 0 + 0) / 4 = 45.0
        assert stats["workbench.usage.prompt_tokens"]["mean"] == 45.0
        assert stats["workbench.usage.prompt_tokens"]["coverage"] == 1.0

        # Check completion_tokens statistics: (100 + 20 + 0 + 0) / 4 = 30.0
        assert stats["workbench.usage.completion_tokens"]["mean"] == 30.0
        assert stats["workbench.usage.completion_tokens"]["coverage"] == 1.0

    def test_empty_inference_results(self, metric_definition):
        """Test behavior with empty inference results list."""
        evaluator = Evaluator(metric_definition=metric_definition)

        results = evaluator.run_metrics([])

        assert results == {}


# Made with Bob
