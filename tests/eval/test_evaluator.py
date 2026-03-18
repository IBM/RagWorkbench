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


@pytest.fixture
def sample_inference_results():
    """Provide sample inference results for testing."""
    return [
        InferenceResult(
            question_id="q1",
            question="Who is the president of the United States?",
            ground_truth_answers=["Joe Biden"],
            ground_truths_context_ids=[GroundTruthContextId(document_id="doc1")],
            is_answerable=True,
            answer="Joe Biden",
            context_ids=["doc1"],
        ),
        InferenceResult(
            question_id="q2",
            question="What is the capital of France?",
            ground_truth_answers=["Paris"],
            ground_truths_context_ids=[GroundTruthContextId(document_id="doc2")],
            is_answerable=True,
            answer="Paris",
            context_ids=["doc2"],
        ),
        InferenceResult(
            question_id="q3",
            question="What is the largest planet in our solar system?",
            ground_truth_answers=["Jupiter"],
            ground_truths_context_ids=[GroundTruthContextId(document_id="doc3")],
            is_answerable=True,
            answer="Saturn",
            context_ids=["doc3"],
        ),
    ]


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
    
    # Run metrics on the inference results
    per_question_scores = evaluator.run_metrics(sample_inference_results)
    
    # Print per-question scores
    print("\n" + "="*80)
    print("PER-QUESTION METRIC SCORES:")
    print("="*80)
    for question_id, scores in per_question_scores.items():
        print(f"\n{question_id}:")
        for score_name, score_value in scores.items():
            print(f"  {score_name}: {score_value}")
    
    # Verify we got results for all questions
    assert len(per_question_scores) == 3
    assert "q1" in per_question_scores
    assert "q2" in per_question_scores
    assert "q3" in per_question_scores
    
    # Verify each question has scores
    for question_id, scores in per_question_scores.items():
        assert isinstance(scores, dict)
        assert len(scores) > 0
        # Verify all scores are floats
        for score_name, score_value in scores.items():
            assert isinstance(score_name, str)
            assert isinstance(score_value, (int, float))
    
    # Compute statistics from per-question results
    stats = evaluator.compute_stats_from_per_question_results(per_question_scores)
    
    # Print statistics
    print("\n" + "="*80)
    print("METRIC STATISTICS:")
    print("="*80)
    for metric_name, metric_stats in stats.items():
        print(f"\n{metric_name}:")
        for stat_name, stat_value in metric_stats.items():
            print(f"  {stat_name}: {stat_value}")
    print("="*80 + "\n")
    
    # Verify statistics structure
    assert isinstance(stats, dict)
    assert len(stats) > 0
    
    # Each metric should have statistics
    for metric_name, metric_stats in stats.items():
        assert isinstance(metric_stats, dict)
        # Statistics should include mean, std, etc.
        assert "mean" in metric_stats
        assert isinstance(metric_stats["mean"], (int, float))

# Made with Bob
