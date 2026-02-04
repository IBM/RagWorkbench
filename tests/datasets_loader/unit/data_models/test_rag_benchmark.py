"""
Tests for RagBenchmark data model.

This module tests the RagBenchmark class focusing on business logic,
filtering operations, and utility methods.
"""

import pytest
from pydantic import ValidationError

from ragbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmark,
    RagBenchmarkEntry,
)


class TestRagBenchmark:
    """Test suite for RagBenchmark model."""

    def test_minimum_entries_validation(self):
        """Test that RagBenchmark requires at least one entry."""
        with pytest.raises(ValidationError):
            RagBenchmark(benchmark_entries=[])

    def test_filtering_answerable_queries(self, sample_rag_benchmark):
        """Test filtering for answerable vs all queries across all methods."""
        # Test get_questions
        all_questions = sample_rag_benchmark.get_questions(
            answerable_queries_only=False
        )
        answerable_questions = sample_rag_benchmark.get_questions(
            answerable_queries_only=True
        )

        assert len(all_questions) == 6
        assert len(answerable_questions) == 4
        assert "What is the weather today?" in all_questions
        assert "What is the weather today?" not in answerable_questions

        # Test get_question_ids
        all_ids = sample_rag_benchmark.get_question_ids(answerable_queries_only=False)
        answerable_ids = sample_rag_benchmark.get_question_ids(
            answerable_queries_only=True
        )

        assert len(all_ids) == 6
        assert len(answerable_ids) == 4
        assert "q_4" in all_ids and "q_4" not in answerable_ids

        # Test get_benchmark_entries
        all_entries = sample_rag_benchmark.get_benchmark_entries(
            answerable_queries_only=False
        )
        answerable_entries = sample_rag_benchmark.get_benchmark_entries(
            answerable_queries_only=True
        )

        assert len(all_entries) == 6
        assert len(answerable_entries) == 4
        assert all(e.is_answerable for e in answerable_entries)

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
                ground_truth_context_ids=[
                    GroundTruthContextId(document_id="doc_a"),
                    GroundTruthContextId(document_id="doc_b"),
                ],
            ),
            RagBenchmarkEntry(
                question_id="q_2",
                question="Question 2?",
                ground_truth_context_ids=[
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
            ground_truth_context_ids=[],
        )
        benchmark = RagBenchmark(benchmark_entries=[entry])

        doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        assert doc_ids == set()

    def test_len_method(self, sample_rag_benchmark):
        """Test that __len__ returns the correct number of entries."""
        assert len(sample_rag_benchmark) == 6

        # Test with single entry
        single_entry = RagBenchmarkEntry(
            question_id="q_single",
            question="Single question?",
        )
        single_benchmark = RagBenchmark(benchmark_entries=[single_entry])
        assert len(single_benchmark) == 1
