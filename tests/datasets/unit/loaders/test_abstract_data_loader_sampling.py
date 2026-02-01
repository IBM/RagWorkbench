"""
Tests for RagDataLoader sampling functionality.

This module tests the sampling logic of the RagDataLoader abstract base class,
including question sampling, document sampling, and reproducibility.
"""

from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from tests.datasets.helpers.mock_data_loader import MockRagDataLoader


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
