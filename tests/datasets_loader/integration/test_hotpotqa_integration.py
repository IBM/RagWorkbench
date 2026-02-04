"""
Integration tests for HotpotQA data loader.

This module contains integration tests that load real HotpotQA data from HuggingFace
and verify data integrity, including multi-hop reasoning support.
"""

import pytest

from ragbench.datasets_loader.hotpot_qa_data_loader import HotpotQaDataLoader
from tests.datasets_loader.helpers.integration_test_helpers import (
    IntegrationTestHelpers as helpers,
)


@pytest.mark.integration
class TestHotpotQAIntegration:
    """Integration tests for HotpotQA data loader with real data."""

    @pytest.mark.parametrize("split", ["train", "validation"])
    def test_ground_truth_documents_exist_in_corpus(self, split):
        """
        Test that all ground-truth doc_ids in benchmark exist in corpus.

        This test verifies data integrity by ensuring that every document
        referenced in the benchmark's ground truth context IDs actually
        exists in the loaded corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.
        """
        # Load HotpotQA data
        loader = HotpotQaDataLoader(split=split)

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify all ground-truth documents exist in corpus
        helpers.assert_ground_truth_documents_exist(corpus, benchmark, split)

    def test_multi_hop_questions_have_multiple_contexts(self):
        """
        Test that multi-hop questions reference multiple documents.

        HotpotQA is specifically designed for multi-hop reasoning, so many
        questions should require information from multiple documents.
        """
        # Load HotpotQA data
        loader = HotpotQaDataLoader(split="train")
        benchmark = loader.get_benchmark()

        # Count entries with multiple ground truth contexts
        multi_context_entries = [
            entry
            for entry in benchmark.benchmark_entries
            if len(entry.ground_truth_context_ids) > 1
        ]

        assert len(multi_context_entries) > 0, (
            "HotpotQA should have questions requiring multiple documents "
            "for multi-hop reasoning"
        )

    def test_document_ids_are_unique(self):
        """
        Test that all document IDs in the corpus are unique.

        This test verifies data integrity by ensuring no duplicate document IDs.
        """
        # Load HotpotQA data
        loader = HotpotQaDataLoader(split="train")
        corpus = loader.get_corpus()

        # Verify uniqueness
        helpers.assert_document_ids_unique(corpus)

    def test_question_ids_are_unique(self):
        """
        Test that all question IDs in the benchmark are unique.

        This test verifies data integrity by ensuring no duplicate question IDs.
        """
        # Load HotpotQA data
        loader = HotpotQaDataLoader(split="train")
        benchmark = loader.get_benchmark()

        # Verify uniqueness
        helpers.assert_question_ids_unique(benchmark)

    def test_documents_have_content(self):
        """
        Test that documents have non-empty content.

        Verifies that Wikipedia paragraphs are properly loaded with content.
        """
        # Load HotpotQA data
        loader = HotpotQaDataLoader(split="train")
        corpus = loader.get_corpus()

        # Verify documents have content
        helpers.assert_documents_have_content(corpus, sample_size=20)

    def test_entries_have_answers(self):
        """
        Test that all benchmark entries have non-empty answers.

        Verifies that every question has at least one answer.
        """
        # Load HotpotQA data
        loader = HotpotQaDataLoader(split="train")
        benchmark = loader.get_benchmark()

        # Verify all entries have answers
        helpers.assert_entries_have_answers(benchmark)
