"""
Comprehensive tests for RagDataLoader abstract base class.

This module tests the RagDataLoader class including sampling logic,
initialization, and integration between corpus and benchmark components.
Uses a concrete mock implementation for testing abstract methods.
"""

from io import BytesIO
from typing import Literal

from datasets.datasets_utils import get_benchmark_split
from src.datasets.abstract_data_loader import RagDataLoader
from src.datasets.data_models.data_sampling_params import DataSamplingParams
from src.datasets.data_models.dataset_names import DatasetName
from src.datasets.data_models.document_object import DocumentObject
from src.datasets.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmark,
    RagBenchmarkEntry,
)
from src.datasets.data_models.rag_corpus import RagCorpus


class MockRagDataLoader(RagDataLoader):
    """
    Concrete implementation of RagDataLoader for testing purposes.

    This mock loader provides fixed test data for documents and benchmark entries,
    allowing us to test the abstract base class functionality.
    """

    def __init__(
        self,
        dataset_name: DatasetName = DatasetName.AI_ARXIV,
        split: Literal["train", "test"] | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        num_docs: int = 20,
        num_questions: int = 15,
    ):
        """
        Initialize mock data loader with configurable test data size.

        Args:
            dataset_name: Dataset identifier.
            split: Dataset split.
            sampling_params: Sampling parameters.
            num_docs: Number of documents to generate.
            num_questions: Number of questions to generate.
        """
        self.num_docs = num_docs
        self.num_questions = num_questions
        super().__init__(dataset_name, split, sampling_params)

    def _get_documents(self) -> list[DocumentObject]:
        """Generate mock documents for testing."""
        documents = []
        for i in range(self.num_docs):
            doc = DocumentObject(
                name=f"mock_doc_{i}",
                stream=BytesIO(f"Content of mock document {i}".encode()),
                mime_type="application/pdf",
                metadata={"index": i, "category": f"cat_{i % 3}"},
            )
            documents.append(doc)
        return documents

    def _get_benchmark_entries(
        self, split: Literal["train", "test"] | None
    ) -> list[RagBenchmarkEntry]:
        """Generate mock benchmark entries for testing."""
        entries = []
        # Use first 10 documents as ground truth (or fewer if num_docs < 10)
        num_gt_docs = min(10, self.num_docs)

        for i in range(self.num_questions):
            # Cycle through ground truth documents
            doc_id = f"mock_doc_{i % num_gt_docs}"
            entry = RagBenchmarkEntry(
                question_id=f"mock_q_{i}",
                question=f"Mock question {i} about document {i % num_gt_docs}?",
                ground_truth_answers=[f"Mock answer {i}"],
                ground_truth_context_ids=[GroundTruthContextId(document_id=doc_id)],
                is_answerable=True,
            )
            entries.append(entry)
        # We have to handle the split case:
        return get_benchmark_split(entries, split)


class TestRagDataLoaderSampling:
    """Test suite for RagDataLoader sampling functionality."""

    def test_question_sampling_with_limit(self):
        """Test that question sampling respects the question_limit parameter."""
        sampling_params = DataSamplingParams(question_limit=5, seed=42)
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=15
        )

        benchmark = loader.get_benchmark()
        assert len(benchmark) == 5

    def test_question_sampling_no_limit(self):
        """Test that all questions are included when no limit is specified."""
        sampling_params = DataSamplingParams(question_limit=None)
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=15
        )

        benchmark = loader.get_benchmark()
        assert len(benchmark) == 15

    def test_question_sampling_limit_exceeds_available(self):
        """Test behavior when question_limit exceeds available questions."""
        sampling_params = DataSamplingParams(question_limit=100, seed=42)
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=15
        )

        benchmark = loader.get_benchmark()
        # Should return all available questions (15), not fail
        assert len(benchmark) == 15

    def test_document_sampling_with_factor(self):
        """Test document sampling with document_factor parameter."""
        # With 15 questions using 10 unique docs as ground truth,
        # and document_factor=2, we expect: 10 GT docs + 20 non-relevant = 30 total
        # But we only have 20 docs total, so: 10 GT + 10 non-relevant = 20
        sampling_params = DataSamplingParams(
            question_limit=15, document_factor=2, seed=42
        )
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=15
        )

        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Get ground truth document IDs
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)

        # All ground truth documents should be in corpus
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        assert gt_doc_ids.issubset(corpus_doc_ids)

        # Total documents should be GT + (factor * GT), capped by available docs
        expected_max = len(gt_doc_ids) + (2 * len(gt_doc_ids))
        assert len(corpus) <= expected_max
        assert len(corpus) <= 20  # Can't exceed total available

    def test_document_sampling_preserves_ground_truth(self):
        """Test that document sampling always includes all ground truth documents."""
        sampling_params = DataSamplingParams(
            question_limit=10, document_factor=1, seed=42
        )
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=15
        )

        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Get all ground truth document IDs from benchmark
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)

        # Get all document IDs from corpus
        corpus_doc_ids = {doc.name for doc in corpus.documents}

        # All ground truth documents must be in corpus
        assert gt_doc_ids.issubset(corpus_doc_ids)

    def test_sampling_reproducibility_with_seed(self):
        """Test that same seed produces identical sampling results."""
        sampling_params_1 = DataSamplingParams(
            question_limit=8, document_factor=2, seed=123
        )
        sampling_params_2 = DataSamplingParams(
            question_limit=8, document_factor=2, seed=123
        )

        loader_1 = MockRagDataLoader(
            sampling_params=sampling_params_1, num_docs=20, num_questions=15
        )
        loader_2 = MockRagDataLoader(
            sampling_params=sampling_params_2, num_docs=20, num_questions=15
        )

        # Get question IDs from both loaders
        questions_1 = loader_1.get_benchmark().get_question_ids()
        questions_2 = loader_2.get_benchmark().get_question_ids()

        # Should be identical with same seed
        assert questions_1 == questions_2

        # Get document names from both loaders
        docs_1 = {doc.name for doc in loader_1.get_corpus().documents}
        docs_2 = {doc.name for doc in loader_2.get_corpus().documents}

        # Should be identical with same seed
        assert docs_1 == docs_2

    def test_sampling_different_results_with_different_seeds(self):
        """Test that different seeds produce different sampling results."""
        sampling_params_1 = DataSamplingParams(
            question_limit=8, document_factor=2, seed=111
        )
        sampling_params_2 = DataSamplingParams(
            question_limit=8, document_factor=2, seed=222
        )

        loader_1 = MockRagDataLoader(
            sampling_params=sampling_params_1, num_docs=20, num_questions=15
        )
        loader_2 = MockRagDataLoader(
            sampling_params=sampling_params_2, num_docs=20, num_questions=15
        )

        questions_1 = set(loader_1.get_benchmark().get_question_ids())
        questions_2 = set(loader_2.get_benchmark().get_question_ids())

        # With different seeds, results should likely differ
        # (not guaranteed but highly probable with sufficient data)
        assert questions_1 != questions_2

    def test_combined_question_and_document_sampling(self):
        """Test that question and document sampling work together correctly."""
        sampling_params = DataSamplingParams(
            question_limit=5, document_factor=3, seed=42
        )
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=15
        )

        benchmark = loader.get_benchmark()
        corpus = loader.get_corpus()

        # Check question sampling
        assert len(benchmark) == 5

        # Check document sampling includes ground truth
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        assert gt_doc_ids.issubset(corpus_doc_ids)

        # Check document count is reasonable
        expected_max = len(gt_doc_ids) + (3 * len(gt_doc_ids))
        assert len(corpus) <= expected_max


class TestRagDataLoaderInitialization:
    """Test suite for RagDataLoader initialization."""

    def test_initialization_with_default_sampling(self):
        """Test initialization with default sampling parameters (no sampling)."""
        loader = MockRagDataLoader(num_docs=10, num_questions=8)

        # With default params (no sampling), all data should be included
        assert len(loader.get_benchmark()) == 8
        assert len(loader.get_corpus()) == 10

    def test_split_handling_train(self):
        """Test initialization with 'train' split."""
        loader = MockRagDataLoader(split="train", num_docs=15, num_questions=10)

        assert loader.split == "train"
        assert isinstance(loader.get_benchmark(), RagBenchmark)
        assert isinstance(loader.get_corpus(), RagCorpus)

    def test_split_handling_test(self):
        """Test initialization with 'test' split."""
        loader = MockRagDataLoader(split="test", num_docs=15, num_questions=10)

        assert loader.split == "test"
        assert isinstance(loader.get_benchmark(), RagBenchmark)
        assert isinstance(loader.get_corpus(), RagCorpus)

    def test_split_handling_none(self):
        """Test initialization with None split (full dataset)."""
        loader = MockRagDataLoader(split=None, num_docs=15, num_questions=10)

        assert loader.split is None
        assert isinstance(loader.get_benchmark(), RagBenchmark)
        assert isinstance(loader.get_corpus(), RagCorpus)

    def test_corpus_and_benchmark_creation(self):
        """Test that corpus and benchmark are properly created during initialization."""
        loader = MockRagDataLoader(num_docs=12, num_questions=8)

        # Check that instances are created
        assert hasattr(loader, "benchmark")
        assert hasattr(loader, "rag_corpus")
        assert isinstance(loader.benchmark, RagBenchmark)
        assert isinstance(loader.rag_corpus, RagCorpus)

        # Check that they contain data
        assert len(loader.benchmark) > 0
        assert len(loader.rag_corpus) > 0


class TestRagDataLoaderMethods:
    """Test suite for RagDataLoader public methods."""

    def test_get_benchmark_returns_instance(self):
        """Test that get_benchmark() returns a RagBenchmark instance."""
        loader = MockRagDataLoader(num_docs=10, num_questions=5)
        benchmark = loader.get_benchmark()

        assert isinstance(benchmark, RagBenchmark)
        assert len(benchmark) == 5

    def test_get_corpus_returns_instance(self):
        """Test that get_corpus() returns a RagCorpus instance."""
        loader = MockRagDataLoader(num_docs=10, num_questions=5)
        corpus = loader.get_corpus()

        assert isinstance(corpus, RagCorpus)
        assert len(corpus) == 10

    def test_get_benchmark_consistency(self):
        """Test that get_benchmark() returns the same instance consistently."""
        loader = MockRagDataLoader(num_docs=10, num_questions=5)

        benchmark_1 = loader.get_benchmark()
        benchmark_2 = loader.get_benchmark()

        # Should return the same instance
        assert benchmark_1 is benchmark_2

    def test_get_corpus_consistency(self):
        """Test that get_corpus() returns the same instance consistently."""
        loader = MockRagDataLoader(num_docs=10, num_questions=5)

        corpus_1 = loader.get_corpus()
        corpus_2 = loader.get_corpus()

        # Should return the same instance
        assert corpus_1 is corpus_2

    def test_integration_end_to_end(self):
        """Test complete end-to-end workflow with the data loader."""
        sampling_params = DataSamplingParams(
            question_limit=6, document_factor=2, seed=42
        )
        loader = MockRagDataLoader(
            dataset_name=DatasetName.AI_ARXIV,
            split="test",
            sampling_params=sampling_params,
            num_docs=15,
            num_questions=10,
        )

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify basic properties
        assert len(benchmark) == 6
        assert len(corpus) > 0

        # Verify ground truth documents are in corpus
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        assert gt_doc_ids.issubset(corpus_doc_ids)

        # Verify we can get questions
        questions = benchmark.get_questions()
        assert len(questions) == 6
        assert all(isinstance(q, str) for q in questions)

        # Verify we can access documents
        first_doc = corpus[0]
        assert isinstance(first_doc, DocumentObject)
        assert first_doc.name.startswith("mock_doc_")


class TestRagDataLoaderEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_minimal_dataset_single_doc_single_question(self):
        """Test with minimal dataset (1 document, 1 question)."""
        loader = MockRagDataLoader(num_docs=1, num_questions=1)

        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        assert len(corpus) == 1
        assert len(benchmark) == 1

    def test_no_document_sampling_with_none_factor(self):
        """Test that document_factor=None means no document sampling."""
        sampling_params = DataSamplingParams(
            question_limit=5, document_factor=None, seed=42
        )
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=10
        )

        corpus = loader.get_corpus()

        # All documents should be included
        assert len(corpus) == 20

    def test_document_factor_zero(self):
        """Test document sampling with factor=0 (only ground truth docs)."""
        sampling_params = DataSamplingParams(
            question_limit=5, document_factor=0, seed=42
        )
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=10
        )

        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Should only have ground truth documents
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        corpus_doc_ids = {doc.name for doc in corpus.documents}

        assert len(corpus) == len(gt_doc_ids)
        assert corpus_doc_ids == gt_doc_ids
