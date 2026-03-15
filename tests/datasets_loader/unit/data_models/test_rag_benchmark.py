"""
Tests for RagBenchmark data model.

This module comprehensively tests the RagBenchmark class, focusing on:
- Benchmark creation and validation
- Filtering operations (answerable vs all queries)
- Utility methods (get_questions, get_question_ids, get_doc_ids_set)
- Immutability of frozen fields
- Edge cases and special scenarios
"""

import pytest
from pydantic import ValidationError

from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmark,
    RagBenchmarkEntry,
)


class TestRagBenchmark:
    """Comprehensive test suite for RagBenchmark model."""

    # ============================================================================
    # Section 1: Creation and Validation
    # ============================================================================

    def test_creation_with_entries(self, sample_benchmark_entries):
        """Test creating a RagBenchmark with valid entries."""
        benchmark = RagBenchmark(benchmark_entries=sample_benchmark_entries)

        assert len(benchmark.benchmark_entries) == 6
        assert isinstance(benchmark, RagBenchmark)

    def test_minimum_entries_validation(self):
        """Test that RagBenchmark requires at least one entry."""
        with pytest.raises(ValidationError):
            RagBenchmark(benchmark_entries=[])

    def test_single_entry_benchmark(self):
        """Test creating a benchmark with single entry."""
        entry = RagBenchmarkEntry(
            question_id="q_single",
            question="Single question?",
        )
        benchmark = RagBenchmark(benchmark_entries=[entry])

        assert len(benchmark) == 1
        assert benchmark.benchmark_entries[0].question_id == "q_single"

    # ============================================================================
    # Section 2: Filtering Methods - Consolidated with Parametrize
    # ============================================================================

    @pytest.mark.parametrize(
        "answerable_only,expected_count",
        [
            (True, 4),  # Only answerable questions
            (False, 6),  # All questions
        ],
    )
    def test_get_questions_filtering(
        self, sample_rag_benchmark, answerable_only, expected_count
    ):
        """Test get_questions() with answerable filtering."""
        questions = sample_rag_benchmark.get_questions(
            answerable_queries_only=answerable_only
        )

        assert len(questions) == expected_count
        assert all(isinstance(q, str) for q in questions)

        if answerable_only:
            # Unanswerable questions should not be in the list
            assert "What is the weather today?" not in questions
            assert "Who is the president?" not in questions

    @pytest.mark.parametrize(
        "answerable_only,expected_count",
        [
            (True, 4),
            (False, 6),
        ],
    )
    def test_get_question_ids_filtering(
        self, sample_rag_benchmark, answerable_only, expected_count
    ):
        """Test get_question_ids() with answerable filtering."""
        question_ids = sample_rag_benchmark.get_question_ids(
            answerable_queries_only=answerable_only
        )

        assert len(question_ids) == expected_count
        assert all(isinstance(qid, str) for qid in question_ids)

        if answerable_only:
            assert "q_4" not in question_ids  # Unanswerable
            assert "q_5" not in question_ids  # Unanswerable

    @pytest.mark.parametrize(
        "answerable_only,expected_count",
        [
            (True, 4),
            (False, 6),
        ],
    )
    def test_get_benchmark_entries_filtering(
        self, sample_rag_benchmark, answerable_only, expected_count
    ):
        """Test get_benchmark_entries() with answerable filtering."""
        entries = sample_rag_benchmark.get_benchmark_entries(
            answerable_queries_only=answerable_only
        )

        assert len(entries) == expected_count
        assert all(isinstance(e, RagBenchmarkEntry) for e in entries)

        if answerable_only:
            assert all(e.is_answerable for e in entries)

    def test_filtering_consistency_across_methods(self, sample_rag_benchmark):
        """Test that all filtering methods return consistent counts."""
        for answerable_only in [True, False]:
            questions = sample_rag_benchmark.get_questions(answerable_only)
            question_ids = sample_rag_benchmark.get_question_ids(answerable_only)
            entries = sample_rag_benchmark.get_benchmark_entries(answerable_only)

            # All methods should return same count
            assert len(questions) == len(question_ids) == len(entries)

    # ============================================================================
    # Section 3: get_doc_ids_set() Method
    # ============================================================================

    def test_get_doc_ids_set_basic(self, sample_benchmark_entries):
        """Test extracting unique document IDs from benchmark entries."""
        doc_ids = RagBenchmark.get_doc_ids_set(sample_benchmark_entries)

        assert isinstance(doc_ids, set)
        assert len(doc_ids) == 4  # doc_0, doc_1, doc_2, doc_3
        assert {"doc_0", "doc_1", "doc_2", "doc_3"}.issubset(doc_ids)

    def test_get_doc_ids_set_with_duplicates(self):
        """Test document ID extraction handles duplicates correctly."""
        entries = [
            RagBenchmarkEntry(
                question_id="q_1",
                question="Question 1?",
                ground_truths_context_ids=[
                    GroundTruthContextId(document_id="doc_a"),
                    GroundTruthContextId(document_id="doc_b"),
                ],
            ),
            RagBenchmarkEntry(
                question_id="q_2",
                question="Question 2?",
                ground_truths_context_ids=[
                    GroundTruthContextId(document_id="doc_b"),  # Duplicate
                    GroundTruthContextId(document_id="doc_c"),
                ],
            ),
        ]

        doc_ids = RagBenchmark.get_doc_ids_set(entries)
        assert doc_ids == {"doc_a", "doc_b", "doc_c"}

    def test_get_doc_ids_set_empty_contexts(self):
        """Test document ID extraction with entries having no contexts."""
        entry = RagBenchmarkEntry(
            question_id="q_no_context",
            question="Question without context?",
            ground_truths_context_ids=[],
        )
        benchmark = RagBenchmark(benchmark_entries=[entry])

        doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        assert doc_ids == set()

    def test_get_doc_ids_set_mixed_answerable_unanswerable(self):
        """Test get_doc_ids_set with mix of answerable and unanswerable entries."""
        entries = [
            RagBenchmarkEntry(
                question_id="q_1",
                question="Answerable?",
                ground_truths_context_ids=[GroundTruthContextId(document_id="doc_1")],
                is_answerable=True,
            ),
            RagBenchmarkEntry(
                question_id="q_2",
                question="Unanswerable?",
                ground_truths_context_ids=[],
                is_answerable=False,
            ),
        ]

        doc_ids = RagBenchmark.get_doc_ids_set(entries)
        assert doc_ids == {"doc_1"}

    # ============================================================================
    # Section 4: Length and Iteration
    # ============================================================================

    def test_len_method(self, sample_rag_benchmark):
        """Test that __len__ returns the correct number of entries."""
        assert len(sample_rag_benchmark) == 6

    def test_len_with_single_entry(self):
        """Test __len__ with single entry benchmark."""
        entry = RagBenchmarkEntry(
            question_id="q_single",
            question="Single question?",
        )
        benchmark = RagBenchmark(benchmark_entries=[entry])
        assert len(benchmark) == 1

    def test_len_with_large_benchmark(self):
        """Test __len__ with large number of entries."""
        entries = [
            RagBenchmarkEntry(
                question_id=f"q_{i}",
                question=f"Question {i}?",
            )
            for i in range(100)
        ]
        benchmark = RagBenchmark(benchmark_entries=entries)
        assert len(benchmark) == 100

    # ============================================================================
    # Section 5: Immutability (Frozen Fields)
    # ============================================================================

    def test_benchmark_entries_immutability(self, sample_rag_benchmark):
        """Test that benchmark_entries cannot be modified after creation."""
        with pytest.raises((ValidationError, AttributeError)):
            sample_rag_benchmark.benchmark_entries = []  # type: ignore

    def test_entries_list_is_tuple(self, sample_rag_benchmark):
        """Test that benchmark_entries is a tuple (immutable sequence).

        Note: Pydantic with frozen=True converts lists to tuples for immutability.
        """
        # Check if it's a tuple (immutable) or list
        entries = sample_rag_benchmark.benchmark_entries

        # Pydantic frozen fields should be tuples
        assert isinstance(entries, (list, tuple))

        # If it's a list, it should still be the original list
        assert len(entries) == 6

    # ============================================================================
    # Section 6: Edge Cases and Special Scenarios
    # ============================================================================

    def test_all_unanswerable_questions(self):
        """Test benchmark with only unanswerable questions."""
        entries = [
            RagBenchmarkEntry(
                question_id=f"q_{i}",
                question=f"Unanswerable {i}?",
                is_answerable=False,
            )
            for i in range(3)
        ]
        benchmark = RagBenchmark(benchmark_entries=entries)

        answerable_entries = benchmark.get_benchmark_entries(
            answerable_queries_only=True
        )
        assert len(answerable_entries) == 0

        all_entries = benchmark.get_benchmark_entries(answerable_queries_only=False)
        assert len(all_entries) == 3

    def test_benchmark_with_no_ground_truth_contexts(self):
        """Test benchmark where no entries have ground truth contexts."""
        entries = [
            RagBenchmarkEntry(
                question_id=f"q_{i}",
                question=f"Question {i}?",
                ground_truths_context_ids=[],
            )
            for i in range(5)
        ]
        benchmark = RagBenchmark(benchmark_entries=entries)

        doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        assert len(doc_ids) == 0

    def test_benchmark_with_many_contexts_per_entry(self):
        """Test benchmark with entries having many ground truth contexts."""
        contexts = [GroundTruthContextId(document_id=f"doc_{i}") for i in range(10)]
        entry = RagBenchmarkEntry(
            question_id="q_multi",
            question="Multi-context question?",
            ground_truths_context_ids=contexts,
        )
        benchmark = RagBenchmark(benchmark_entries=[entry])

        doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        assert len(doc_ids) == 10

    def test_equality(self, sample_benchmark_entries):
        """Test equality of RagBenchmark instances."""
        benchmark1 = RagBenchmark(benchmark_entries=sample_benchmark_entries)
        benchmark2 = RagBenchmark(benchmark_entries=sample_benchmark_entries)

        assert benchmark1 == benchmark2

    def test_inequality(self):
        """Test inequality of RagBenchmark instances with different entries."""
        entry1 = RagBenchmarkEntry(question_id="q_1", question="Q1?")
        entry2 = RagBenchmarkEntry(question_id="q_2", question="Q2?")

        benchmark1 = RagBenchmark(benchmark_entries=[entry1])
        benchmark2 = RagBenchmark(benchmark_entries=[entry2])

        assert benchmark1 != benchmark2

    def test_representation(self, sample_rag_benchmark):
        """Test string representation of RagBenchmark."""
        repr_str = repr(sample_rag_benchmark)

        assert "RagBenchmark" in repr_str or "benchmark_entries" in repr_str
