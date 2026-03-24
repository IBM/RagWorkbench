"""
Test the Evaluator class with a single metric.

This test module validates the Evaluator workflow including:
- Creating an Evaluator instance with a metric definition
- Running metrics on inference results
- Computing statistics from per-question results
"""

import shutil
from pathlib import Path

import pytest

from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
)
from ragworkbench.eval.evaluator import Evaluator
from ragworkbench.eval.metric_models import load_metric_definitions


def create_inference_result(index: int, question: str, answer: str) -> InferenceResult:
    """
    Generate an inference result from an index, question, and answer.

    Args:
        index: The index/ID for the question
        question: The question text
        answer: The answer text

    Returns:
        InferenceResult with the provided data
    """
    return InferenceResult(
        question_id=f"q{index}",
        question=question,
        ground_truth_answers=[answer],
        ground_truths_context_ids=[GroundTruthContextId(document_id=f"doc{index}")],
        is_answerable=True,
        answer=answer,
        context_ids=[f"doc{index}"],
    )


@pytest.fixture
def sample_inference_results():
    """Provide sample inference results for testing."""
    questions_and_answers = [
        ("Who is the president of the United States?", "Joe Biden"),
        ("What is the capital of France?", "Paris"),
        ("What is the largest planet in our solar system?", "Jupiter"),
        ("What is the speed of light?", "299,792,458 meters per second"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("What is the chemical symbol for gold?", "Au"),
        ("What year did World War II end?", "1945"),
        ("What is the tallest mountain in the world?", "Mount Everest"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the smallest prime number?", "2"),
        ("What is the capital of Japan?", "Tokyo"),
        ("Who invented the telephone?", "Alexander Graham Bell"),
        ("What is the largest ocean on Earth?", "Pacific Ocean"),
        ("What is the boiling point of water in Celsius?", "100 degrees"),
        ("Who was the first person to walk on the moon?", "Neil Armstrong"),
        ("What is the currency of the United Kingdom?", "Pound Sterling"),
        ("What is the square root of 144?", "12"),
        ("Who wrote '1984'?", "George Orwell"),
        ("What is the longest river in the world?", "Nile River"),
        ("What is the atomic number of carbon?", "6"),
    ]

    return [
        create_inference_result(i + 1, question, answer)
        for i, (question, answer) in enumerate(questions_and_answers)
    ]


@pytest.mark.skip(reason="Test takes >1 second (44.75s)")
def test_evaluator_runs_single_metric(sample_inference_results):
    """Test that Evaluator can run a single LLMaaJ metric and produce results."""
    # Delete inference_engine_cache before the run
    cache_path = Path("inference_engine_cache")
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"\nDeleted inference_engine_cache at: {cache_path.absolute()}")

    # Load metric definitions
    config = load_metric_definitions()

    # Get a single metric definition (LLM as a Judge - llmaaj_llama)
    metric_name = "unitxt.answer_correctness.llmaaj_llama"
    metric_def = config.get_metric_definition(metric_name)

    # Create Evaluator instance
    evaluator = Evaluator(
        metric_definition=metric_def,
        rag_benchmark=None,
        rag_corpus=None,
        cache_dir=None,
    )

    max_instances_to_test = 1
    sample_inference_results = sample_inference_results[:max_instances_to_test]
    # Run metrics on the inference results
    per_question_scores = evaluator.run_metrics(sample_inference_results)

    # Print per-question scores
    print("\n" + "=" * 80)
    print("PER-QUESTION METRIC SCORES:")
    print("=" * 80)
    for question_id, scores in per_question_scores.items():
        print(f"\n{question_id}:")
        for score_name, score_value in scores.items():
            print(f"  {score_name}: {score_value}")

    # Verify we got results for the expected number of questions
    assert len(per_question_scores) == max_instances_to_test
    for i in range(1, max_instances_to_test + 1):
        assert f"q{i}" in per_question_scores

    # Verify each question has scores
    for _question_id, scores in per_question_scores.items():
        assert isinstance(scores, dict)
        assert len(scores) > 0
        # Verify all scores are floats
        for score_name, score_value in scores.items():
            assert isinstance(score_name, str)
            assert isinstance(score_value, (int, float))

    # Compute statistics from per-question results
    stats = evaluator.compute_stats_from_per_question_results(per_question_scores)

    # Print statistics
    print("\n" + "=" * 80)
    print("METRIC STATISTICS:")
    print("=" * 80)
    for metric_name, metric_stats in stats.items():
        print(f"\n{metric_name}:")
        for stat_name, stat_value in metric_stats.items():
            print(f"  {stat_name}: {stat_value}")
    print("=" * 80 + "\n")

    # Verify statistics structure
    assert isinstance(stats, dict)
    assert len(stats) > 0

    # Each metric should have statistics
    for _metric_name, metric_stats in stats.items():
        assert isinstance(metric_stats, dict)
        # Statistics should include mean, std, etc.
        assert "mean" in metric_stats
        assert isinstance(metric_stats["mean"], (int, float))


# Made with Bob
