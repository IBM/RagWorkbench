"""
Integration tests for KramaBench data loader.

This module contains integration tests that load real KramaBench data from HuggingFace
and verify data integrity.
"""

import pytest

from ragbench.datasets_loader.kramabench_data_loader import KramabenchDataLoader
from tests.datasets_loader.helpers.integration_test_helpers import (
    IntegrationTestHelpers as helpers,
)


@pytest.mark.integration
class TestKramaBenchIntegration:
    """Integration tests for KramaBench data loader with real data."""

    def test_ground_truth_documents_exist_in_corpus(self):
        """
        Test that all ground-truth doc_ids in benchmark exist in corpus.

        This test verifies data integrity by ensuring that every document
        referenced in the benchmark's ground truth context IDs actually
        exists in the loaded corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.
        """
        # Load KramaBench data
        loader = KramabenchDataLoader()

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify all ground-truth documents exist in corpus
        helpers.assert_ground_truth_documents_exist(corpus, benchmark)

    def test_corpus_and_benchmark_not_empty(self):
        """
        Test that both corpus and benchmark contain data.

        Verifies that the loader successfully loads both documents and questions.
        """
        # Load KramaBench data
        loader = KramabenchDataLoader()

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify both are loaded
        helpers.assert_corpus_not_empty(corpus)
        helpers.assert_benchmark_not_empty(benchmark)

    def test_document_ids_are_unique(self):
        """
        Test that all document IDs in the corpus are unique.

        This test verifies data integrity by ensuring no duplicate document IDs.
        """
        # Load KramaBench data
        loader = KramabenchDataLoader()
        corpus = loader.get_corpus()

        # Verify uniqueness
        helpers.assert_document_ids_unique(corpus)

    def test_question_ids_are_unique(self):
        """
        Test that all question IDs in the benchmark are unique.

        This test verifies data integrity by ensuring no duplicate question IDs.
        """
        # Load KramaBench data
        loader = KramabenchDataLoader()
        benchmark = loader.get_benchmark()

        # Verify uniqueness
        helpers.assert_question_ids_unique(benchmark)

    def test_documents_have_content(self):
        """
        Test that documents have non-empty content.

        Verifies that documents are properly loaded with content.
        """
        # Load KramaBench data
        loader = KramabenchDataLoader()
        corpus = loader.get_corpus()

        # Verify documents have content
        helpers.assert_documents_have_content(corpus, sample_size=20)

    def test_entries_have_answers(self):
        """
        Test that all benchmark entries have non-empty answers.

        Verifies that every question has at least one answer.
        """
        # Load KramaBench data
        loader = KramabenchDataLoader()
        benchmark = loader.get_benchmark()

        # Verify all entries have answers
        helpers.assert_entries_have_answers(benchmark)
