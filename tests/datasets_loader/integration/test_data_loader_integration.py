"""
Unified integration tests for BioASQ and CLAP-NQ data loaders.

This module contains integration tests that load real data from HuggingFace
and verify data integrity, particularly ensuring that all ground-truth documents
referenced in benchmark entries exist in the corpus.

The tests are parameterized to run against both BioasqDataLoader and ClapNqDataLoader,
ensuring consistent behavior and data integrity across both datasets.
"""

from typing import Literal

import pytest

from ragbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragbench.datasets_loader.clap_nq_data_loader import ClapNqDataLoader
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


@pytest.fixture(scope="class")
def clap_nq_train_loader() -> ClapNqDataLoader:
    """
    Class-scoped fixture that loads CLAP-NQ train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return ClapNqDataLoader(split="train")


@pytest.fixture(scope="class")
def clap_nq_test_loader() -> ClapNqDataLoader:
    """
    Class-scoped fixture that loads CLAP-NQ test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return ClapNqDataLoader(split="test")


@pytest.mark.integration
@pytest.mark.parametrize(
    "loader_name",
    [
        "bioasq",
        "clap_nq",
    ],
)
class TestDataLoaderIntegration:
    """
    Unified integration tests for BioASQ and CLAP-NQ data loaders with real data.

    This test class is parameterized to run all tests against both BioasqDataLoader
    and ClapNqDataLoader, ensuring consistent behavior and data integrity across
    both datasets.

    The parameterization allows us to:
    - Eliminate code duplication between similar test files
    - Ensure consistent test coverage across different loaders
    - Easily extend to additional loaders in the future
    - Maintain efficient testing with class-scoped fixtures
    """

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_ground_truth_documents_exist_in_corpus(
        self,
        split: Literal["train", "test"],
        loader_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        Test that all ground-truth doc_ids in benchmark exist in corpus.

        This test verifies data integrity by ensuring that every document
        referenced in the benchmark's ground truth context IDs actually
        exists in the loaded corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.

        Args:
            split: The dataset split to test ('train' or 'test')
            loader_name: Name of the loader being tested (for fixture lookup)
            request: Pytest fixture request object for dynamic fixture access
        """
        # Get the appropriate loader fixture based on loader_name and split
        fixture_name = f"{loader_name}_{split}_loader"
        loader: RagDataLoader = request.getfixturevalue(fixture_name)

        # Get corpus and benchmark
        corpus: RagCorpus = loader.get_corpus()
        benchmark: RagBenchmark = loader.get_benchmark()

        # Verify all ground-truth documents exist in corpus
        helpers.assert_ground_truth_documents_exist(corpus, benchmark, split)

    def test_document_ids_are_unique(
        self,
        loader_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        Test that all document IDs in the corpus are unique.

        This test verifies data integrity by ensuring no duplicate document IDs.

        Args:
            loader_name: Name of the loader being tested (for fixture lookup)
            request: Pytest fixture request object for dynamic fixture access
        """
        # Get the train loader fixture for this loader
        fixture_name = f"{loader_name}_train_loader"
        loader: RagDataLoader = request.getfixturevalue(fixture_name)

        # Get corpus from shared loader
        corpus: RagCorpus = loader.get_corpus()

        # Verify uniqueness
        helpers.assert_document_ids_unique(corpus)

    def test_question_ids_are_unique(
        self,
        loader_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        Test that all question IDs in the benchmark are unique.

        This test verifies data integrity by ensuring no duplicate question IDs.

        Args:
            loader_name: Name of the loader being tested (for fixture lookup)
            request: Pytest fixture request object for dynamic fixture access
        """
        # Get the train loader fixture for this loader
        fixture_name = f"{loader_name}_train_loader"
        loader: RagDataLoader = request.getfixturevalue(fixture_name)

        # Get benchmark from shared loader
        benchmark: RagBenchmark = loader.get_benchmark()

        # Verify uniqueness
        helpers.assert_question_ids_unique(benchmark)

    def test_documents_have_content(
        self,
        loader_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        Test that documents have non-empty content.

        Verifies that documents are properly loaded with content.

        Args:
            loader_name: Name of the loader being tested (for fixture lookup)
            request: Pytest fixture request object for dynamic fixture access
        """
        # Get the train loader fixture for this loader
        fixture_name = f"{loader_name}_train_loader"
        loader: RagDataLoader = request.getfixturevalue(fixture_name)

        # Get corpus from shared loader
        corpus: RagCorpus = loader.get_corpus()

        # Verify documents have content
        helpers.assert_documents_have_content(corpus, sample_size=20)

    def test_entries_have_answers(
        self,
        loader_name: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """
        Test that all benchmark entries have non-empty answers.

        Verifies that every question has at least one answer.

        Args:
            loader_name: Name of the loader being tested (for fixture lookup)
            request: Pytest fixture request object for dynamic fixture access
        """
        # Get the train loader fixture for this loader
        fixture_name = f"{loader_name}_train_loader"
        loader: RagDataLoader = request.getfixturevalue(fixture_name)

        # Get benchmark from shared loader
        benchmark: RagBenchmark = loader.get_benchmark()

        # Verify all entries have answers
        helpers.assert_entries_have_answers(benchmark)
