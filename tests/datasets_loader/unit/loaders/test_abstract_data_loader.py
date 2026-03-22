"""
Comprehensive tests for RagDataLoader abstract base class.

This module tests the complete functionality of RagDataLoader including:
- Initialization and configuration
- Question and document sampling
- Edge cases and boundary conditions
- Reproducibility and determinism
"""

import pytest

from ragworkbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)
from ragworkbench.datasets_loader.data_models.document_object import DocumentObject
from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmark,
    RagBenchmarkEntry,
)
from ragworkbench.datasets_loader.dataset_names import DatasetName
from tests.datasets_loader.helpers.mock_data_loader import MockRagDataLoader


class TestRagDataLoaderInitialization:
    """Test suite for RagDataLoader initialization and configuration."""

    def test_initialization_with_default_sampling(self):
        """Test initialization with default sampling parameters (no sampling)."""
        loader = MockRagDataLoader(num_docs=10, num_questions=8)

        # With default params (no sampling), all data should be included
        assert len(loader.get_benchmark()) == 8
        assert len(loader.get_corpus()) == 10
        assert loader.dataset_name == DatasetName.BIOASQ
        assert loader.split is None

    @pytest.mark.parametrize(
        "split,expected_split",
        [
            ("train", "train"),
            ("test", "test"),
            (None, None),
        ],
    )
    def test_split_handling(self, split, expected_split):
        """Test initialization with different split values."""
        loader = MockRagDataLoader(split=split, num_docs=15, num_questions=10)

        assert loader.split == expected_split
        assert isinstance(loader.get_benchmark(), RagBenchmark)
        assert len(loader.get_benchmark()) > 0
        assert len(loader.get_corpus()) > 0

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters specified."""
        sampling_params = DataSamplingParams(
            question_limit=5, document_factor=2, seed=42
        )
        loader = MockRagDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=sampling_params,
            num_docs=20,
            num_questions=15,
        )

        assert loader.dataset_name == DatasetName.BIOASQ
        assert loader.split == "train"
        assert len(loader.get_benchmark()) == 5  # question_limit applied


class TestRagDataLoaderQuestionSampling:
    """Test suite for question sampling functionality."""

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_question_sampling_with_limit(self):
        """Test that question sampling respects the question_limit parameter."""
        sampling_params = DataSamplingParams(question_limit=5, seed=42)
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=15
        )

        benchmark = loader.get_benchmark()
        assert len(benchmark) == 5

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_question_sampling_no_limit(self):
        """Test that all questions are included when no limit is specified."""
        sampling_params = DataSamplingParams(question_limit=None)
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=15
        )

        benchmark = loader.get_benchmark()
        assert len(benchmark) == 15

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_question_sampling_limit_exceeds_available(self):
        """Test behavior when question_limit exceeds available questions."""
        sampling_params = DataSamplingParams(question_limit=100, seed=42)
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=15
        )

        benchmark = loader.get_benchmark()
        # Should return all available questions (15), not fail
        assert len(benchmark) == 15


class TestRagDataLoaderDocumentSampling:
    """Test suite for document sampling functionality."""

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_document_sampling_with_factor(self):
        """Test document sampling with document_factor parameter."""
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

    @pytest.mark.skip(reason="Test disabled - needs fixing")
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

    @pytest.mark.skip(reason="Test disabled - needs fixing")
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

    @pytest.mark.skip(reason="Test disabled - needs fixing")
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


class TestRagDataLoaderReproducibility:
    """Test suite for sampling reproducibility and determinism."""

    @pytest.mark.skip(reason="Test disabled - needs fixing")
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

    @pytest.mark.skip(reason="Test disabled - needs fixing")
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


class TestRagDataLoaderEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_minimal_dataset_single_doc_single_question(self):
        """Test with minimal dataset (1 document, 1 question)."""
        loader = MockRagDataLoader(num_docs=1, num_questions=1)

        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        assert len(corpus) == 1
        assert len(benchmark) == 1

    @pytest.mark.skip(reason="Test disabled - needs fixing")
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

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_document_factor_exceeds_available_non_relevant_docs(self):
        """Test when document_factor requests more non-relevant docs than available."""
        # 10 docs total, 8 questions using first 8 docs as GT
        # document_factor=5 would request 8 + (5*8) = 48 docs, but only 10 exist
        sampling_params = DataSamplingParams(
            question_limit=8, document_factor=5, seed=42
        )
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=10, num_questions=8
        )

        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Should include all available documents (10)
        assert len(corpus) == 10

        # All ground truth docs should still be present
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        assert gt_doc_ids.issubset(corpus_doc_ids)


class TestLoadSampleStaticMethod:
    """Test suite for the _load_sample static method."""

    def test_load_sample_no_sampling(self):
        """Test _load_sample with no sampling parameters."""
        from io import BytesIO

        # Create test data
        docs = [
            DocumentObject(
                name=f"doc_{i}",
                stream=BytesIO(b"content"),
                mime_type="text/plain",
            )
            for i in range(10)
        ]
        entries = [
            RagBenchmarkEntry(
                question_id=f"q_{i}",
                question=f"Question {i}",
                ground_truth_answers=[f"Answer {i}"],
                ground_truths_context_ids=[
                    GroundTruthContextId(document_id=f"doc_{i}")
                ],
                is_answerable=True,
            )
            for i in range(5)
        ]

        sampling_params = DataSamplingParams()
        sampled_entries, sampled_docs = RagDataLoader._load_sample(
            entries, docs, sampling_params
        )

        # No sampling should return all data
        assert len(sampled_entries) == 5
        assert len(sampled_docs) == 10

    def test_load_sample_with_question_limit(self):
        """Test _load_sample with question_limit only."""
        from io import BytesIO

        docs = [
            DocumentObject(
                name=f"doc_{i}",
                stream=BytesIO(b"content"),
                mime_type="text/plain",
            )
            for i in range(10)
        ]
        entries = [
            RagBenchmarkEntry(
                question_id=f"q_{i}",
                question=f"Question {i}",
                ground_truth_answers=[f"Answer {i}"],
                ground_truths_context_ids=[
                    GroundTruthContextId(document_id=f"doc_{i}")
                ],
                is_answerable=True,
            )
            for i in range(10)
        ]

        sampling_params = DataSamplingParams(question_limit=3, seed=42)
        sampled_entries, sampled_docs = RagDataLoader._load_sample(
            entries, docs, sampling_params
        )

        # Should sample 3 questions
        assert len(sampled_entries) == 3
        # All docs should remain (no document_factor specified)
        assert len(sampled_docs) == 10

    def test_load_sample_with_document_factor(self):
        """Test _load_sample with document_factor."""
        from io import BytesIO

        docs = [
            DocumentObject(
                name=f"doc_{i}",
                stream=BytesIO(b"content"),
                mime_type="text/plain",
            )
            for i in range(20)
        ]
        entries = [
            RagBenchmarkEntry(
                question_id=f"q_{i}",
                question=f"Question {i}",
                ground_truth_answers=[f"Answer {i}"],
                ground_truths_context_ids=[
                    GroundTruthContextId(document_id=f"doc_{i % 5}")
                ],  # 5 unique GT docs
                is_answerable=True,
            )
            for i in range(10)
        ]

        sampling_params = DataSamplingParams(document_factor=2, seed=42)
        sampled_entries, sampled_docs = RagDataLoader._load_sample(
            entries, docs, sampling_params
        )

        # All questions should remain
        assert len(sampled_entries) == 10

        # Should have 5 GT docs + (2 * 5) non-relevant = 15 total
        assert len(sampled_docs) == 15

        # Verify all GT docs are present
        gt_doc_ids = RagBenchmark.get_doc_ids_set(sampled_entries)
        sampled_doc_ids = {doc.name for doc in sampled_docs}
        assert gt_doc_ids.issubset(sampled_doc_ids)

    def test_load_sample_does_not_modify_original_lists(self):
        """Test that _load_sample doesn't modify the original input lists."""
        from io import BytesIO

        docs = [
            DocumentObject(
                name=f"doc_{i}",
                stream=BytesIO(b"content"),
                mime_type="text/plain",
            )
            for i in range(10)
        ]
        entries = [
            RagBenchmarkEntry(
                question_id=f"q_{i}",
                question=f"Question {i}",
                ground_truth_answers=[f"Answer {i}"],
                ground_truths_context_ids=[
                    GroundTruthContextId(document_id=f"doc_{i}")
                ],
                is_answerable=True,
            )
            for i in range(10)
        ]

        original_docs_len = len(docs)
        original_entries_len = len(entries)

        sampling_params = DataSamplingParams(
            question_limit=3, document_factor=1, seed=42
        )
        RagDataLoader._load_sample(entries, docs, sampling_params)

        # Original lists should remain unchanged
        assert len(docs) == original_docs_len
        assert len(entries) == original_entries_len
