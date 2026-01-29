"""
Tests for RagDataLoader public methods and edge cases.

This module tests the public methods of RagDataLoader and various
edge cases and boundary conditions.
"""

from ragbench.datasets.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets.data_models.rag_benchmark import (
    RagBenchmark,
)
from ragbench.datasets.data_models.rag_corpus import RagCorpus
from tests.datasets.helpers.mock_data_loader import MockRagDataLoader


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

    def test_end_to_end_integration_workflow(self):
        """Test complete end-to-end workflow with data loader."""
        sampling_params = DataSamplingParams(
            question_limit=5, document_factor=2, seed=42
        )
        loader = MockRagDataLoader(
            sampling_params=sampling_params, num_docs=20, num_questions=10
        )

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify basic properties
        assert len(benchmark) == 5
        assert len(corpus) > 0

        # Verify ground truth documents are in corpus
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        assert gt_doc_ids.issubset(corpus_doc_ids)

        # Verify we can get questions
        questions = benchmark.get_questions()
        assert len(questions) == 5


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
