"""
Comprehensive tests for RAG benchmark data models.

This module tests the GroundTruthContextId, RagBenchmarkEntry, and RagBenchmark
classes, including validation, immutability, filtering, and utility methods.
"""

import pytest
from pydantic import ValidationError

from src.datasets.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmark,
    RagBenchmarkEntry,
)


class TestGroundTruthContextId:
    """Test suite for GroundTruthContextId model."""

    def test_valid_creation_with_all_fields(self):
        """Test creating a GroundTruthContextId with all fields provided."""
        context_id = GroundTruthContextId(
            document_id="doc_123", page=5, table_id="table_1"
        )

        assert context_id.document_id == "doc_123"
        assert context_id.page == 5
        assert context_id.table_id == "table_1"

    def test_valid_creation_minimal_fields(self):
        """Test creating a GroundTruthContextId with only required fields."""
        context_id = GroundTruthContextId(document_id="doc_456")

        assert context_id.document_id == "doc_456"
        assert context_id.page is None
        assert context_id.table_id is None

    def test_immutability_frozen_fields(self):
        """Test that frozen fields cannot be modified after creation."""
        context_id = GroundTruthContextId(document_id="doc_789", page=3)

        with pytest.raises(ValidationError):
            context_id.document_id = "new_doc"

        with pytest.raises(ValidationError):
            context_id.page = 10

    def test_page_validation_positive(self):
        """Test that page number must be >= 1 when provided."""
        # Valid page numbers
        context_id = GroundTruthContextId(document_id="doc_1", page=1)
        assert context_id.page == 1

        context_id = GroundTruthContextId(document_id="doc_2", page=100)
        assert context_id.page == 100

        # Invalid page numbers
        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="doc_3", page=0)

        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="doc_4", page=-1)

    def test_document_id_required_and_non_empty(self):
        """Test that document_id is required and cannot be empty."""
        # Missing document_id
        with pytest.raises(ValidationError):
            GroundTruthContextId()  # type: ignore

        # Empty document_id
        with pytest.raises(ValidationError):
            GroundTruthContextId(document_id="")


class TestRagBenchmarkEntry:
    """Test suite for RagBenchmarkEntry model."""

    def test_valid_creation_with_all_fields(self):
        """Test creating a RagBenchmarkEntry with all fields provided."""
        context_id = GroundTruthContextId(document_id="doc_1", page=2)
        entry = RagBenchmarkEntry(
            question_id="q_1",
            question="What is the answer?",
            ground_truth_answers=["Answer 1", "Answer 2"],
            ground_truth_context_ids=[context_id],
            is_answerable=True,
            additional_information={"category": "factual", "difficulty": "easy"},
        )

        assert entry.question_id == "q_1"
        assert entry.question == "What is the answer?"
        assert entry.ground_truth_answers == ["Answer 1", "Answer 2"]
        assert len(entry.ground_truth_context_ids) == 1
        assert entry.is_answerable is True
        assert entry.additional_information == {
            "category": "factual",
            "difficulty": "easy",
        }

    def test_valid_creation_minimal_fields(self):
        """Test creating a RagBenchmarkEntry with only required fields."""
        entry = RagBenchmarkEntry(
            question_id="q_2",
            question="Another question?",
        )

        assert entry.question_id == "q_2"
        assert entry.question == "Another question?"
        assert entry.ground_truth_answers is None
        assert entry.ground_truth_context_ids == []
        assert entry.is_answerable is True  # Default value
        assert entry.additional_information is None

    def test_immutability_frozen_fields(self):
        """Test that frozen fields cannot be modified after creation."""
        entry = RagBenchmarkEntry(
            question_id="q_3",
            question="Test question?",
        )

        with pytest.raises(ValidationError):
            entry.question_id = "new_q_3"

        with pytest.raises(ValidationError):
            entry.question = "New question?"

        with pytest.raises(ValidationError):
            entry.is_answerable = False

    def test_unanswerable_question(self):
        """Test creating an unanswerable question entry."""
        entry = RagBenchmarkEntry(
            question_id="q_4",
            question="What is the weather today?",
            ground_truth_answers=None,
            ground_truth_context_ids=[],
            is_answerable=False,
        )

        assert entry.is_answerable is False
        assert entry.ground_truth_answers is None
        assert entry.ground_truth_context_ids == []

    def test_multiple_ground_truth_contexts(self):
        """Test entry with multiple ground truth context references."""
        contexts = [
            GroundTruthContextId(document_id="doc_1", page=1),
            GroundTruthContextId(document_id="doc_2", page=3),
            GroundTruthContextId(document_id="doc_3"),
        ]
        entry = RagBenchmarkEntry(
            question_id="q_5",
            question="Multi-document question?",
            ground_truth_context_ids=contexts,
        )

        assert len(entry.ground_truth_context_ids) == 3
        assert entry.ground_truth_context_ids[0].document_id == "doc_1"
        assert entry.ground_truth_context_ids[1].document_id == "doc_2"
        assert entry.ground_truth_context_ids[2].document_id == "doc_3"


class TestRagBenchmark:
    """Test suite for RagBenchmark model."""

    def test_creation_with_entries(self):
        """Test creating a RagBenchmark with valid entries."""
        # Create entries directly to avoid fixture dependency issues
        entries = [
            RagBenchmarkEntry(
                question_id=f"q_{i}",
                question=f"Test question {i}?",
                ground_truth_answers=[f"Answer {i}"],
                ground_truth_context_ids=[GroundTruthContextId(document_id=f"doc_{i}")],
                is_answerable=True,
            )
            for i in range(6)
        ]
        benchmark = RagBenchmark(benchmark_entries=entries)

        assert len(benchmark.benchmark_entries) == 6
        assert isinstance(benchmark, RagBenchmark)

    def test_minimum_entries_validation(self):
        """Test that RagBenchmark requires at least one entry."""
        with pytest.raises(ValidationError):
            RagBenchmark(benchmark_entries=[])

    def test_get_questions_all(self, sample_rag_benchmark):
        """Test retrieving all questions without filtering."""
        questions = sample_rag_benchmark.get_questions(answerable_queries_only=False)

        assert len(questions) == 6
        assert "What is the content of document 0?" in questions
        assert "What is the weather today?" in questions

    def test_get_questions_answerable_only(self, sample_rag_benchmark):
        """Test retrieving only answerable questions."""
        questions = sample_rag_benchmark.get_questions(answerable_queries_only=True)

        assert len(questions) == 4
        assert "What is the content of document 0?" in questions
        assert "What is the weather today?" not in questions
        assert "Who is the president?" not in questions

    def test_get_question_ids_all(self, sample_rag_benchmark):
        """Test retrieving all question IDs without filtering."""
        question_ids = sample_rag_benchmark.get_question_ids(
            answerable_queries_only=False
        )

        assert len(question_ids) == 6
        assert "q_0" in question_ids
        assert "q_4" in question_ids
        assert "q_5" in question_ids

    def test_get_question_ids_answerable_only(self, sample_rag_benchmark):
        """Test retrieving only answerable question IDs."""
        question_ids = sample_rag_benchmark.get_question_ids(
            answerable_queries_only=True
        )

        assert len(question_ids) == 4
        assert "q_0" in question_ids
        assert "q_1" in question_ids
        assert "q_4" not in question_ids
        assert "q_5" not in question_ids

    def test_get_benchmark_entries_all(self, sample_rag_benchmark):
        """Test retrieving all benchmark entries without filtering."""
        entries = sample_rag_benchmark.get_benchmark_entries(
            answerable_queries_only=False
        )

        assert len(entries) == 6
        assert all(isinstance(e, RagBenchmarkEntry) for e in entries)

    def test_get_benchmark_entries_answerable_only(self, sample_rag_benchmark):
        """Test retrieving only answerable benchmark entries."""
        entries = sample_rag_benchmark.get_benchmark_entries(
            answerable_queries_only=True
        )

        assert len(entries) == 4
        assert all(e.is_answerable for e in entries)

    def test_get_doc_ids_set_static_method(self, sample_benchmark_entries):
        """Test extracting unique document IDs from benchmark entries."""
        doc_ids = RagBenchmark.get_doc_ids_set(sample_benchmark_entries)

        assert isinstance(doc_ids, set)
        assert len(doc_ids) == 4  # doc_0, doc_1, doc_2, doc_3
        assert "doc_0" in doc_ids
        assert "doc_1" in doc_ids
        assert "doc_2" in doc_ids
        assert "doc_3" in doc_ids

    def test_get_doc_ids_set_with_multiple_contexts(self):
        """Test document ID extraction with entries having multiple contexts."""
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
                    GroundTruthContextId(document_id="doc_b"),
                    GroundTruthContextId(document_id="doc_c"),
                ],
            ),
        ]

        doc_ids = RagBenchmark.get_doc_ids_set(entries)

        assert len(doc_ids) == 3
        assert doc_ids == {"doc_a", "doc_b", "doc_c"}

    def test_len_method(self, sample_rag_benchmark):
        """Test that __len__ returns the correct number of entries."""
        assert len(sample_rag_benchmark) == 6

    def test_len_method_with_single_entry(self):
        """Test __len__ with a benchmark containing a single entry."""
        entry = RagBenchmarkEntry(
            question_id="q_single",
            question="Single question?",
        )
        benchmark = RagBenchmark(benchmark_entries=[entry])

        assert len(benchmark) == 1

    def test_immutability_frozen_benchmark_entries(self, sample_rag_benchmark):
        """Test that benchmark_entries field is frozen."""
        with pytest.raises(ValidationError):
            sample_rag_benchmark.benchmark_entries = []

    def test_mixed_answerable_unanswerable_filtering(self):
        """Test filtering with a mix of answerable and unanswerable questions."""
        entries = [
            RagBenchmarkEntry(
                question_id=f"q_{i}",
                question=f"Question {i}?",
                is_answerable=(i % 2 == 0),  # Even indices are answerable
            )
            for i in range(10)
        ]
        benchmark = RagBenchmark(benchmark_entries=entries)

        all_questions = benchmark.get_questions(answerable_queries_only=False)
        answerable_questions = benchmark.get_questions(answerable_queries_only=True)

        assert len(all_questions) == 10
        assert len(answerable_questions) == 5

    def test_empty_ground_truth_contexts(self):
        """Test benchmark entries with no ground truth contexts."""
        entry = RagBenchmarkEntry(
            question_id="q_no_context",
            question="Question without context?",
            ground_truth_context_ids=[],
        )
        benchmark = RagBenchmark(benchmark_entries=[entry])

        doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        assert len(doc_ids) == 0
        assert doc_ids == set()
