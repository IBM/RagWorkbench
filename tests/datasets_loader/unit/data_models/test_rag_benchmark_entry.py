"""
Tests for RagBenchmarkEntry data model.

This module tests the RagBenchmarkEntry class focusing on business logic
and various entry configurations.
"""

from ragbench.datasets_loader.data_models.rag_benchmark import (
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

    def test_creation_with_minimal_fields(self):
        """Test creating a RagBenchmarkEntry with only required fields and defaults."""
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
