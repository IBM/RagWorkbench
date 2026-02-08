"""
Unified integration tests for multiple data loaders.

This module contains integration tests that load real data from various sources
and verify data integrity, particularly ensuring that all ground-truth documents
referenced in benchmark entries exist in the corpus.

The tests are parameterized to run against all available datasets using the
DataLoaderFactory, ensuring consistent behavior and data integrity across
all datasets.
"""

from typing import Any, Literal

import pytest

from ragbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragbench.datasets_loader.data_loader_factory import DataLoaderFactory
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus
from tests.datasets_loader.helpers.integration_test_helpers import (
    IntegrationTestHelpers as helpers,
)

# List of all available dataset names for parameterization
DATASET_NAMES = [
    "bioasq",
    "clap_nq",
    "da_code",
    "dabstep",
    "hotpot_qa",
    "kramabench",
    "miniwiki",
    "mldr",
    "narrative_qa",
    "office_qa",
    "qasper",
    "secque",
    "watsonx",
]


def _create_loader_fixture(dataset_name: str, split: Literal["train", "test"]) -> Any:
    """
    Factory function to create loader fixtures dynamically using DataLoaderFactory.

    This eliminates the need for 28 nearly identical fixture definitions.
    Each loader is instantiated via the factory with only the split parameter.

    Args:
        dataset_name: The dataset name to load
        split: Dataset split ('train' or 'test')

    Returns:
        A fixture function that creates the appropriate loader instance
    """

    @pytest.fixture(scope="class")
    def loader_fixture() -> RagDataLoader:
        return DataLoaderFactory.create_loader(dataset_name=dataset_name, split=split)

    return loader_fixture


# Dynamically create all loader fixtures
for dataset_name in DATASET_NAMES:
    for split_value in ["train", "test"]:
        fixture_name = f"{dataset_name}_{split_value}_loader"
        # Cast to Literal type for type checker
        split_literal: Literal["train", "test"] = split_value  # type: ignore[assignment]
        globals()[fixture_name] = _create_loader_fixture(dataset_name, split_literal)


@pytest.mark.integration
@pytest.mark.parametrize(
    "loader_name",
    [
        "bioasq",
        "clap_nq",
        "da_code",
        "dabstep",
        "hotpot_qa",
        "kramabench",
        "miniwiki",
        "mldr",
        "narrative_qa",
        "office_qa",
        "qasper",
        "secque",
        "watsonx",
    ],
)
class TestDataLoaderIntegration:
    """
    Unified integration tests for multiple data loaders with real data.

    This test class is parameterized to run all tests against BioasqDataLoader,
    ClapNqDataLoader, DaCodeDataLoader, DabStepDataLoader, HotpotQaDataLoader,
    KramabenchDataLoader, MiniWikiDataLoader, MLDRDataLoader, NarrativeQaDataLoader,
    OfficeQADataLoader, QasperQaDataLoader, SecqueDataLoader, and WatsonxDocsQADataLoader,
    ensuring consistent behavior and data integrity across all datasets.

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

        Note:
            DabStep and MiniWiki have empty ground_truth_context_ids, so this
            test is skipped for them.

        Args:
            split: The dataset split to test ('train' or 'test')
            loader_name: Name of the loader being tested (for fixture lookup)
            request: Pytest fixture request object for dynamic fixture access
        """
        # Skip for DabStep and MiniWiki as they have empty ground_truth_context_ids
        if loader_name in ["dabstep", "miniwiki"]:
            pytest.skip(f"{loader_name} has empty ground_truth_context_ids")

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

        Note:
            DA-Code stores answers in additional_information['gold_answer_documents']
            rather than ground_truth_answers, so this test is skipped for DA-Code.

        Args:
            loader_name: Name of the loader being tested (for fixture lookup)
            request: Pytest fixture request object for dynamic fixture access
        """
        # Skip for DA-Code as it stores answers differently
        if loader_name == "da_code":
            pytest.skip(
                "DA-Code stores answers in additional_information, not ground_truth_answers"
            )

        # Get the train loader fixture for this loader
        fixture_name = f"{loader_name}_train_loader"
        loader: RagDataLoader = request.getfixturevalue(fixture_name)

        # Get benchmark from shared loader
        benchmark: RagBenchmark = loader.get_benchmark()

        # Verify all entries have answers
        helpers.assert_entries_have_answers(benchmark)
