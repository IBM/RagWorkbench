"""
Benchmark-related pytest fixtures.

This module provides fixtures for creating sample benchmark entries,
ground truth context IDs, and RagBenchmark instances for testing.
"""

import pytest

from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmark,
    RagBenchmarkEntry,
)


@pytest.fixture
def sample_ground_truth_context_ids() -> list[GroundTruthContextId]:
    """
    Create a list of sample GroundTruthContextId instances.

    Returns:
        List of 5 GroundTruthContextId instances with various configurations.
    """
    return [
        GroundTruthContextId(document_id="doc_0", page=1),
        GroundTruthContextId(document_id="doc_1", page=2, table_id="table_1"),
        GroundTruthContextId(document_id="doc_2"),
        GroundTruthContextId(document_id="doc_3", page=5),
        GroundTruthContextId(document_id="doc_4", table_id="table_2"),
    ]


@pytest.fixture
def sample_benchmark_entries(
    sample_ground_truth_context_ids: list[GroundTruthContextId],
) -> list[RagBenchmarkEntry]:
    """
    Create a list of sample RagBenchmarkEntry instances.

    Includes both answerable and unanswerable questions with various
    ground truth configurations.

    Args:
        sample_ground_truth_context_ids: Fixture providing context IDs.

    Returns:
        List of 6 RagBenchmarkEntry instances (4 answerable, 2 unanswerable).
    """
    return [
        # Answerable questions
        RagBenchmarkEntry(
            question_id="q_0",
            question="What is the content of document 0?",
            ground_truth_answers=["Content of document 0"],
            ground_truths_context_ids=[sample_ground_truth_context_ids[0]],
            is_answerable=True,
        ),
        RagBenchmarkEntry(
            question_id="q_1",
            question="What is in the table of document 1?",
            ground_truth_answers=["Table data", "Data from table"],
            ground_truths_context_ids=[sample_ground_truth_context_ids[1]],
            is_answerable=True,
            additional_information={"difficulty": "medium"},
        ),
        RagBenchmarkEntry(
            question_id="q_2",
            question="Describe document 2.",
            ground_truth_answers=["Document 2 description"],
            ground_truths_context_ids=[sample_ground_truth_context_ids[2]],
            is_answerable=True,
        ),
        RagBenchmarkEntry(
            question_id="q_3",
            question="What is mentioned on page 5 of document 3?",
            ground_truth_answers=["Page 5 content"],
            ground_truths_context_ids=[sample_ground_truth_context_ids[3]],
            is_answerable=True,
        ),
        # Unanswerable questions
        RagBenchmarkEntry(
            question_id="q_4",
            question="What is the weather today?",
            ground_truth_answers=None,
            ground_truths_context_ids=[],
            is_answerable=False,
        ),
        RagBenchmarkEntry(
            question_id="q_5",
            question="Who is the president?",
            ground_truth_answers=None,
            ground_truths_context_ids=[],
            is_answerable=False,
            additional_information={"category": "out_of_scope"},
        ),
    ]


@pytest.fixture
def sample_rag_benchmark(
    sample_benchmark_entries: list[RagBenchmarkEntry],
) -> RagBenchmark:
    """
    Create a sample RagBenchmark instance.

    Args:
        sample_benchmark_entries: Fixture providing benchmark entries.

    Returns:
        RagBenchmark instance with sample entries.
    """
    return RagBenchmark(benchmark_entries=sample_benchmark_entries)


@pytest.fixture
def large_benchmark_entries() -> list[RagBenchmarkEntry]:
    """
    Create a large set of benchmark entries for sampling tests.

    Returns:
        List of 15 RagBenchmarkEntry instances.
    """
    entries = []
    for i in range(15):
        # Use documents 0-9 as ground truth (10 documents)
        doc_id = f"large_doc_{i % 10}"
        entry = RagBenchmarkEntry(
            question_id=f"large_q_{i}",
            question=f"Question {i} about document {i % 10}?",
            ground_truth_answers=[f"Answer {i}"],
            ground_truths_context_ids=[GroundTruthContextId(document_id=doc_id)],
            is_answerable=True,
        )
        entries.append(entry)
    return entries
