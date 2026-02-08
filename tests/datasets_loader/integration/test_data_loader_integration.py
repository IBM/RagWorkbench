"""
Unified integration tests for multiple data loaders.

This module contains integration tests that load real data from various sources
and verify data integrity, particularly ensuring that all ground-truth documents
referenced in benchmark entries exist in the corpus.

The tests are parameterized to run against BioasqDataLoader, ClapNqDataLoader,
DaCodeDataLoader, DabStepDataLoader, HotpotQaDataLoader, KramabenchDataLoader,
MiniWikiDataLoader, MLDRDataLoader, NarrativeQaDataLoader, OfficeQADataLoader,
QasperQaDataLoader, SecqueDataLoader, and WatsonxDocsQADataLoader, ensuring
consistent behavior and data integrity across all datasets.
"""

from typing import Any, Literal

import pytest

from ragbench.datasets_loader import KramabenchDataLoader
from ragbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragbench.datasets_loader.clap_nq_data_loader import ClapNqDataLoader
from ragbench.datasets_loader.da_code_data_loader import DaCodeDataLoader
from ragbench.datasets_loader.dabstep_data_loader import DabStepDataLoader
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus
from ragbench.datasets_loader.hotpot_qa_data_loader import HotpotQaDataLoader
from ragbench.datasets_loader.miniwiki_data_loader import MiniWikiDataLoader
from ragbench.datasets_loader.mldr_data_loader import MLDRDataLoader
from ragbench.datasets_loader.narrative_qa_data_loader import NarrativeQaDataLoader
from ragbench.datasets_loader.office_qa_data_loader import OfficeQADataLoader
from ragbench.datasets_loader.qasper_data_loader import QasperQaDataLoader
from ragbench.datasets_loader.secque_data_loader import SecqueDataLoader
from ragbench.datasets_loader.watsonx_data_loader import WatsonxDocsQADataLoader
from tests.datasets_loader.helpers.integration_test_helpers import (
    IntegrationTestHelpers as helpers,
)

# Mapping of loader names to their corresponding classes
LOADER_CLASSES: dict[str, type[RagDataLoader]] = {
    "bioasq": BioasqDataLoader,
    "clap_nq": ClapNqDataLoader,
    "da_code": DaCodeDataLoader,
    "dabstep": DabStepDataLoader,
    "hotpot_qa": HotpotQaDataLoader,
    "kramabench": KramabenchDataLoader,
    "miniwiki": MiniWikiDataLoader,
    "mldr": MLDRDataLoader,
    "narrative_qa": NarrativeQaDataLoader,
    "office_qa": OfficeQADataLoader,
    "qasper": QasperQaDataLoader,
    "secque": SecqueDataLoader,
    "watsonx": WatsonxDocsQADataLoader,
}


def _create_loader_fixture(loader_class: type[RagDataLoader], split: str) -> Any:
    """
    Factory function to create loader fixtures dynamically.

    This eliminates the need for 28 nearly identical fixture definitions.
    Each loader is instantiated with only the split parameter.

    Args:
        loader_class: The loader class to instantiate
        split: Dataset split ('train' or 'test')

    Returns:
        A fixture function that creates the appropriate loader instance
    """

    @pytest.fixture(scope="class")
    def loader_fixture() -> RagDataLoader:
        return loader_class(split=split)  # type: ignore[call-arg]

    return loader_fixture


# Dynamically create all loader fixtures
for loader_name, loader_class in LOADER_CLASSES.items():
    for split in ["train", "test"]:
        fixture_name = f"{loader_name}_{split}_loader"
        globals()[fixture_name] = _create_loader_fixture(loader_class, split)


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
