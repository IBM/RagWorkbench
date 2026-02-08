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

from typing import Literal

import pytest

from ragbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragbench.datasets_loader.clap_nq_data_loader import ClapNqDataLoader
from ragbench.datasets_loader.da_code_data_loader import DaCodeDataLoader
from ragbench.datasets_loader.dabstep_data_loader import DabStepDataLoader
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus
from ragbench.datasets_loader.hotpot_qa_data_loader import HotpotQaDataLoader
from ragbench.datasets_loader.kramabench_data_loader import KramabenchDataLoader
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


@pytest.fixture(scope="class")
def da_code_train_loader() -> DaCodeDataLoader:
    """
    Class-scoped fixture that loads DA-Code train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return DaCodeDataLoader(split="train")


@pytest.fixture(scope="class")
def da_code_test_loader() -> DaCodeDataLoader:
    """
    Class-scoped fixture that loads DA-Code test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return DaCodeDataLoader(split="test")


@pytest.fixture(scope="class")
def dabstep_train_loader() -> DabStepDataLoader:
    """
    Class-scoped fixture that loads DabStep train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return DabStepDataLoader(split="train")


@pytest.fixture(scope="class")
def dabstep_test_loader() -> DabStepDataLoader:
    """
    Class-scoped fixture that loads DabStep test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return DabStepDataLoader(split="test")


@pytest.fixture(scope="class")
def hotpot_qa_train_loader() -> HotpotQaDataLoader:
    """
    Class-scoped fixture that loads HotpotQA train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return HotpotQaDataLoader(split="train")


@pytest.fixture(scope="class")
def hotpot_qa_test_loader() -> HotpotQaDataLoader:
    """
    Class-scoped fixture that loads HotpotQA test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return HotpotQaDataLoader(split="test")


@pytest.fixture(scope="class")
def kramabench_train_loader() -> KramabenchDataLoader:
    """
    Class-scoped fixture that loads Kramabench train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return KramabenchDataLoader(split="train")


@pytest.fixture(scope="class")
def kramabench_test_loader() -> KramabenchDataLoader:
    """
    Class-scoped fixture that loads Kramabench test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return KramabenchDataLoader(split="test")


@pytest.fixture(scope="class")
def miniwiki_train_loader() -> MiniWikiDataLoader:
    """
    Class-scoped fixture that loads MiniWiki train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return MiniWikiDataLoader(split="train")


@pytest.fixture(scope="class")
def miniwiki_test_loader() -> MiniWikiDataLoader:
    """
    Class-scoped fixture that loads MiniWiki test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return MiniWikiDataLoader(split="test")


@pytest.fixture(scope="class")
def mldr_train_loader() -> MLDRDataLoader:
    """
    Class-scoped fixture that loads MLDR train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return MLDRDataLoader(split="train")


@pytest.fixture(scope="class")
def mldr_test_loader() -> MLDRDataLoader:
    """
    Class-scoped fixture that loads MLDR test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return MLDRDataLoader(split="test")


@pytest.fixture(scope="class")
def narrative_qa_train_loader() -> NarrativeQaDataLoader:
    """
    Class-scoped fixture that loads NarrativeQA train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return NarrativeQaDataLoader(split="train")


@pytest.fixture(scope="class")
def narrative_qa_test_loader() -> NarrativeQaDataLoader:
    """
    Class-scoped fixture that loads NarrativeQA test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return NarrativeQaDataLoader(split="test")


@pytest.fixture(scope="class")
def office_qa_train_loader() -> OfficeQADataLoader:
    """
    Class-scoped fixture that loads OfficeQA train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return OfficeQADataLoader(split="train")


@pytest.fixture(scope="class")
def office_qa_test_loader() -> OfficeQADataLoader:
    """
    Class-scoped fixture that loads OfficeQA test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return OfficeQADataLoader(split="test")


@pytest.fixture(scope="class")
def qasper_train_loader() -> QasperQaDataLoader:
    """
    Class-scoped fixture that loads Qasper train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return QasperQaDataLoader(split="train")


@pytest.fixture(scope="class")
def qasper_test_loader() -> QasperQaDataLoader:
    """
    Class-scoped fixture that loads Qasper test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return QasperQaDataLoader(split="test")


@pytest.fixture(scope="class")
def secque_train_loader() -> SecqueDataLoader:
    """
    Class-scoped fixture that loads Secque train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return SecqueDataLoader(split="train")


@pytest.fixture(scope="class")
def secque_test_loader() -> SecqueDataLoader:
    """
    Class-scoped fixture that loads Secque test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return SecqueDataLoader(split="test")


@pytest.fixture(scope="class")
def watsonx_train_loader() -> WatsonxDocsQADataLoader:
    """
    Class-scoped fixture that loads WatsonX DocsQA train data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return WatsonxDocsQADataLoader(split="train")


@pytest.fixture(scope="class")
def watsonx_test_loader() -> WatsonxDocsQADataLoader:
    """
    Class-scoped fixture that loads WatsonX DocsQA test data once for all tests.

    This fixture is shared across all test methods in the class to avoid
    the expensive data loading operation multiple times.
    """
    return WatsonxDocsQADataLoader(split="test")


@pytest.mark.integration
@pytest.mark.parametrize(
    "loader_name",
    [
        "bioasq",
        "clap_nq",
        # TODO
        # "da_code",
        "dabstep",
        "hotpot_qa",
        # TODO!
        # "kramabench",
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
