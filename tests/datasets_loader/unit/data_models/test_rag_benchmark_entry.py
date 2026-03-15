"""
Tests for RagBenchmarkEntry data model.

This module comprehensively tests the RagBenchmarkEntry class, focusing on:
- Entry creation with various configurations
- Field validation and defaults
- Answerable vs unanswerable questions
- Ground truth answers and contexts
- Immutability of frozen fields
- Edge cases and special scenarios
"""

import pytest
from pydantic import ValidationError

from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)


class TestRagBenchmarkEntry:
    """Test suite for RagBenchmarkEntry model."""

    def test_creation_with_all_fields(self):
        """Test creating a RagBenchmarkEntry with all optional fields."""
        context_id = GroundTruthContextId(document_id="doc_1", page=2)
        entry = RagBenchmarkEntry(
            question_id="q_1",
            question="What is the answer?",
            ground_truth_answers=["Answer 1", "Answer 2"],
            ground_truths_context_ids=[context_id],
            is_answerable=True,
            additional_information={"category": "factual", "difficulty": "easy"},
        )

        assert entry.question_id == "q_1"
        assert entry.question == "What is the answer?"
        assert entry.ground_truth_answers == ["Answer 1", "Answer 2"]
        assert len(entry.ground_truths_context_ids) == 1
        assert entry.is_answerable is True
        assert entry.additional_information == {
            "category": "factual",
            "difficulty": "easy",
        }

    def test_creation_with_minimal_fields(self):
        """Test creating a RagBenchmarkEntry with only required fields and defaults."""
        entry = RagBenchmarkEntry(
            question_id="q_2",
            question="Another question?",
        )

        assert entry.question_id == "q_2"
        assert entry.question == "Another question?"
        assert entry.ground_truth_answers is None
        assert entry.ground_truths_context_ids == []
        assert entry.is_answerable is True  # Default value
        assert entry.additional_information is None

    def test_unanswerable_question(self):
        """Test creating an unanswerable question entry."""
        entry = RagBenchmarkEntry(
            question_id="q_4",
            question="What is the weather today?",
            ground_truth_answers=None,
            ground_truths_context_ids=[],
            is_answerable=False,
        )

        assert entry.is_answerable is False
        assert entry.ground_truth_answers is None
        assert entry.ground_truths_context_ids == []

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
            ground_truths_context_ids=contexts,
        )

        assert len(entry.ground_truths_context_ids) == 3
        assert entry.ground_truths_context_ids[0].document_id == "doc_1"
        assert entry.ground_truths_context_ids[1].document_id == "doc_2"
        assert entry.ground_truths_context_ids[2].document_id == "doc_3"

    # ============================================================================
    # Section 4: Ground Truth Answers
    # ============================================================================

    def test_empty_answers_list_vs_none(self):
        """Test difference between empty list and None for ground_truth_answers."""
        entry_with_none = RagBenchmarkEntry(
            question_id="q_9",
            question="Question with None answers?",
            ground_truth_answers=None,
        )
        entry_with_empty = RagBenchmarkEntry(
            question_id="q_10",
            question="Question with empty answers?",
            ground_truth_answers=[],
        )

        assert entry_with_none.ground_truth_answers is None
        assert entry_with_empty.ground_truth_answers == []

    def test_single_answer(self):
        """Test entry with single ground truth answer."""
        entry = RagBenchmarkEntry(
            question_id="q_11",
            question="What is the capital?",
            ground_truth_answers=["Paris"],
        )

        assert entry.ground_truth_answers == ["Paris"]
        assert entry.ground_truth_answers is not None
        assert len(entry.ground_truth_answers) == 1

    def test_multiple_answers(self):
        """Test entry with multiple acceptable answers."""
        entry = RagBenchmarkEntry(
            question_id="q_12",
            question="What is 2+2?",
            ground_truth_answers=["4", "four", "Four", "IV"],
        )

        assert entry.ground_truth_answers is not None
        assert len(entry.ground_truth_answers) == 4
        assert "4" in entry.ground_truth_answers

    # ============================================================================
    # Section 5: Additional Information
    # ============================================================================

    def test_additional_information_with_various_types(self):
        """Test additional_information with various data types."""
        entry = RagBenchmarkEntry(
            question_id="q_13",
            question="Test question?",
            additional_information={
                "category": "factual",
                "difficulty": 3,
                "tags": ["science", "history"],
                "is_verified": True,
            },
        )

        assert entry.additional_information is not None
        assert entry.additional_information["category"] == "factual"
        assert entry.additional_information["difficulty"] == 3
        assert entry.additional_information["is_verified"] is True

    # ============================================================================
    # Section 6: Immutability (Frozen Fields)
    # ============================================================================

    def test_question_id_immutability(self):
        """Test that question_id cannot be modified after creation."""
        entry = RagBenchmarkEntry(question_id="q_1", question="Test?")

        with pytest.raises((ValidationError, AttributeError)):
            entry.question_id = "q_2"  # type: ignore

    def test_question_immutability(self):
        """Test that question cannot be modified after creation."""
        entry = RagBenchmarkEntry(question_id="q_1", question="Original?")

        with pytest.raises((ValidationError, AttributeError)):
            entry.question = "Modified?"  # type: ignore

    def test_is_answerable_immutability(self):
        """Test that is_answerable cannot be modified after creation."""
        entry = RagBenchmarkEntry(
            question_id="q_1", question="Test?", is_answerable=True
        )

        with pytest.raises((ValidationError, AttributeError)):
            entry.is_answerable = False  # type: ignore

    # ============================================================================
    # Section 7: Edge Cases
    # ============================================================================

    def test_very_long_question(self):
        """Test entry with very long question text."""
        long_question = "What is " + "very " * 1000 + "long?"
        entry = RagBenchmarkEntry(question_id="q_long", question=long_question)

        assert len(entry.question) > 5000

    def test_unicode_in_question(self):
        """Test entry with Unicode characters in question."""
        entry = RagBenchmarkEntry(
            question_id="q_unicode",
            question="什么是人工智能？ 🤖",
        )

        assert "什么" in entry.question
        assert "🤖" in entry.question

    def test_equality(self):
        """Test equality of RagBenchmarkEntry instances."""
        entry1 = RagBenchmarkEntry(
            question_id="q_1",
            question="Test?",
            ground_truth_answers=["Answer"],
        )
        entry2 = RagBenchmarkEntry(
            question_id="q_1",
            question="Test?",
            ground_truth_answers=["Answer"],
        )

        assert entry1 == entry2

    def test_inequality(self):
        """Test inequality of RagBenchmarkEntry instances."""
        entry1 = RagBenchmarkEntry(question_id="q_1", question="Test?")
        entry2 = RagBenchmarkEntry(question_id="q_2", question="Test?")

        assert entry1 != entry2
