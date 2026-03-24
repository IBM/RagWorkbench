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


# Made with Bob
