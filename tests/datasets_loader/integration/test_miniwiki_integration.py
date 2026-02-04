"""
Integration tests for Mini Wikipedia data loader.

This module contains integration tests that load real Mini Wikipedia data from HuggingFace
and verify data integrity and functionality.
"""

import pytest

from ragbench.datasets_loader.miniwiki_data_loader import MiniWikiDataLoader
from tests.datasets_loader.helpers.integration_test_helpers import (
    IntegrationTestHelpers as helpers,
)


@pytest.mark.integration
class TestMiniWikiIntegration:
    """Integration tests for Mini Wikipedia data loader with real data."""

    def test_load_documents(self):
        """
        Test that documents can be loaded from the Mini Wikipedia corpus.

        This test verifies that:
        - Documents can be loaded successfully
        - The corpus is not empty
        - Each document has required attributes
        """
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")

        # Get corpus
        corpus = loader.get_corpus()

        # Verify corpus is not empty and has content
        helpers.assert_corpus_not_empty(corpus)
        helpers.assert_documents_have_content(corpus, sample_size=5)

        # Verify document structure (Mini Wiki specific checks)
        for doc in corpus.documents[:5]:
            assert doc.mime_type == "text/plain", "Document should be text/plain"
            content = doc.stream.read()
            assert isinstance(content, bytes), "Document content should be bytes"
            doc.stream.seek(0)  # Reset for potential future reads

    def test_load_benchmark_entries(self):
        """
        Test that benchmark entries can be loaded from the Mini Wikipedia dataset.

        This test verifies that:
        - Benchmark entries can be loaded successfully
        - Each entry has required attributes
        - Questions and answers are properly formatted
        """
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")

        # Get benchmark
        benchmark = loader.get_benchmark()

        # Verify benchmark is not empty and has answers
        helpers.assert_benchmark_not_empty(benchmark)
        helpers.assert_entries_have_answers(benchmark)
        helpers.assert_entries_are_answerable(benchmark)

        # Verify ground truth context IDs is empty (Mini Wiki specific)
        for entry in benchmark.benchmark_entries[:5]:
            assert (
                len(entry.ground_truth_context_ids) == 0
            ), "Mini Wiki dataset does not have ground truth context IDs"

    def test_corpus_and_benchmark_loaded_together(self):
        """
        Test that both corpus and benchmark can be loaded together.

        This test verifies that:
        - Both corpus and benchmark can be loaded in the same loader instance
        - Data is consistent between calls
        """
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")

        # Get both corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify both are loaded
        helpers.assert_corpus_not_empty(corpus)
        helpers.assert_benchmark_not_empty(benchmark)

        # Verify data consistency - calling again should return same data
        corpus2 = loader.get_corpus()
        benchmark2 = loader.get_benchmark()

        assert len(corpus.documents) == len(
            corpus2.documents
        ), "Corpus should be consistent"
        assert len(benchmark.benchmark_entries) == len(
            benchmark2.benchmark_entries
        ), "Benchmark should be consistent"

    def test_document_ids_are_unique(self):
        """
        Test that all document IDs in the corpus are unique.

        This test verifies data integrity by ensuring no duplicate document IDs.
        """
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")
        corpus = loader.get_corpus()

        # Verify uniqueness
        helpers.assert_document_ids_unique(corpus)

    def test_question_ids_are_unique(self):
        """
        Test that all question IDs in the benchmark are unique.

        This test verifies data integrity by ensuring no duplicate question IDs.
        """
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")
        benchmark = loader.get_benchmark()

        # Verify uniqueness
        helpers.assert_question_ids_unique(benchmark)

    def test_answers_are_not_empty(self):
        """
        Test that all benchmark entries have non-empty answers.

        This test verifies that every question has at least one answer.
        """
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")
        benchmark = loader.get_benchmark()

        # Verify all entries have answers
        helpers.assert_entries_have_answers(benchmark)
