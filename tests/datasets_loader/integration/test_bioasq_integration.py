"""
Integration tests for BioASQ data loader.

This module contains integration tests that load real BioASQ data from HuggingFace
and verify data integrity, particularly ensuring that all ground-truth documents
referenced in benchmark entries exist in the corpus.
"""

from typing import Literal

import pytest

from ragbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus
from tests.datasets_loader.helpers.integration_test_helpers import (
    IntegrationTestHelpers as helpers,
)


@pytest.fixture(scope="class")
def bioasq_train_loader() -> BioasqDataLoader:
    """
    Class-scoped fixture that loads BioASQ train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return BioasqDataLoader(split="train")


@pytest.fixture(scope="class")
def bioasq_test_loader() -> BioasqDataLoader:
    """
    Class-scoped fixture that loads BioASQ test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return BioasqDataLoader(split="test")


@pytest.mark.integration
class TestBioASQIntegration:
    """Integration tests for BioASQ data loader with real data."""

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_ground_truth_documents_exist_in_corpus(
        self, split: Literal["train", "test"], request: pytest.FixtureRequest
    ) -> None:
        """
        Test that all ground-truth doc_ids in benchmark exist in corpus.

        This test verifies data integrity by ensuring that every document
        referenced in the benchmark's ground truth context IDs actually
        exists in the loaded corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.
        """
        # Get the appropriate loader fixture based on split parameter
        loader: BioasqDataLoader = request.getfixturevalue(f"bioasq_{split}_loader")

        # Get corpus and benchmark
        corpus: RagCorpus = loader.get_corpus()
        benchmark: RagBenchmark = loader.get_benchmark()

        # Verify all ground-truth documents exist in corpus
        helpers.assert_ground_truth_documents_exist(corpus, benchmark, split)

    def test_document_ids_are_unique(
        self, bioasq_train_loader: BioasqDataLoader
    ) -> None:
        """
        Test that all document IDs in the corpus are unique.

        This test verifies data integrity by ensuring no duplicate document IDs.
        """
        # Get corpus from shared loader
        corpus: RagCorpus = bioasq_train_loader.get_corpus()

        # Verify uniqueness
        helpers.assert_document_ids_unique(corpus)

    def test_question_ids_are_unique(
        self, bioasq_train_loader: BioasqDataLoader
    ) -> None:
        """
        Test that all question IDs in the benchmark are unique.

        This test verifies data integrity by ensuring no duplicate question IDs.
        """
        # Get benchmark from shared loader
        benchmark: RagBenchmark = bioasq_train_loader.get_benchmark()

        # Verify uniqueness
        helpers.assert_question_ids_unique(benchmark)

    def test_documents_have_content(
        self, bioasq_train_loader: BioasqDataLoader
    ) -> None:
        """
        Test that documents have non-empty content.

        Verifies that biomedical documents are properly loaded with content.
        """
        # Get corpus from shared loader
        corpus: RagCorpus = bioasq_train_loader.get_corpus()

        # Verify documents have content
        helpers.assert_documents_have_content(corpus, sample_size=20)

    def test_entries_have_answers(self, bioasq_train_loader: BioasqDataLoader) -> None:
        """
        Test that all benchmark entries have non-empty answers.

        Verifies that every biomedical question has at least one answer.
        """
        # Get benchmark from shared loader
        benchmark: RagBenchmark = bioasq_train_loader.get_benchmark()

        # Verify all entries have answers
        helpers.assert_entries_have_answers(benchmark)
