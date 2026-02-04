"""
Integration tests for WatsonX DocsQA data loader.

This module contains integration tests that load real WatsonX DocsQA data from HuggingFace
and verify data integrity, particularly ensuring that all ground-truth documents
referenced in benchmark entries exist in the corpus.
"""

import pytest

from ragbench.datasets_loader.watsonx_data_loader import WatsonxDocsQADataLoader
from tests.datasets_loader.helpers.integration_test_helpers import (
    IntegrationTestHelpers as helpers,
)


@pytest.mark.integration
class TestWatsonxDocsQAIntegration:
    """Integration tests for WatsonX DocsQA data loader with real data."""

    def test_load_watsonx_docs_qa_dataset(self):
        """
        Test that WatsonX DocsQA dataset can be loaded successfully.

        This test verifies that the loader can successfully load the dataset
        from HuggingFace and that the data has the expected structure.
        """
        # Load WatsonX DocsQA data
        loader = WatsonxDocsQADataLoader()

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify we have data
        helpers.assert_corpus_not_empty(corpus)
        helpers.assert_benchmark_not_empty(benchmark)

        # Verify expected dataset size (approximately)
        # The dataset should have around 1,144 documents and 75 Q&A pairs
        assert len(corpus) > 1000, f"Expected ~1,144 documents, got {len(corpus)}"
        assert len(benchmark) > 70, f"Expected ~75 questions, got {len(benchmark)}"

    @pytest.mark.parametrize("split", [None, "train", "test"])
    def test_ground_truth_documents_exist_in_corpus(self, split):
        """
        Test that all ground-truth doc_ids in benchmark exist in corpus.

        This test verifies data integrity by ensuring that every document
        referenced in the benchmark's ground truth context IDs actually
        exists in the loaded corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.
        """
        # Load WatsonX DocsQA data
        loader = WatsonxDocsQADataLoader(split=split)

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify all ground-truth documents exist in corpus
        helpers.assert_ground_truth_documents_exist(corpus, benchmark, split)

    def test_document_metadata_structure(self):
        """
        Test that documents have the expected metadata structure.

        Verifies that each document has the required metadata fields
        (title and url) as specified in the implementation.
        """
        # Load WatsonX DocsQA data
        loader = WatsonxDocsQADataLoader()
        corpus = loader.get_corpus()

        # Verify documents have required metadata fields
        helpers.assert_documents_have_metadata(
            corpus, required_fields=["title", "url"], sample_size=10
        )

    def test_benchmark_entry_structure(self):
        """
        Test that benchmark entries have the expected structure.

        Verifies that each benchmark entry has:
        - A question ID
        - A question text
        - At least one ground truth answer
        - Exactly one ground truth context ID (single document reference)
        - is_answerable set to True
        """
        # Load WatsonX DocsQA data
        loader = WatsonxDocsQADataLoader()
        benchmark = loader.get_benchmark()

        # Verify entries have answers and are answerable
        helpers.assert_entries_have_answers(benchmark)
        helpers.assert_entries_are_answerable(benchmark)

        # Verify single document reference (WatsonX-specific requirement)
        helpers.assert_entries_have_ground_truth_contexts(benchmark, expected_count=1)

    def test_document_content_not_empty(self):
        """
        Test that documents have non-empty content.

        Verifies that the document content is properly loaded and not empty.
        """
        # Load WatsonX DocsQA data
        loader = WatsonxDocsQADataLoader()
        corpus = loader.get_corpus()

        # Verify documents have content
        helpers.assert_documents_have_content(corpus, sample_size=20)
