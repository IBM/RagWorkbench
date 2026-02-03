"""
Integration tests for CLAP-NQ data loader.

This module contains integration tests that load real CLAP-NQ data from HuggingFace
and verify data integrity, particularly ensuring that all ground-truth documents
referenced in benchmark entries exist in the corpus.
"""

import pytest

from ragbench.datasets_loader.clap_nq_data_loader import ClapNqDataLoader
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark


@pytest.mark.integration
class TestClapNqIntegration:
    """Integration tests for CLAP-NQ data loader with real data."""

    def test_ground_truth_documents_exist_in_corpus(self):
        """
        Test that all ground-truth doc_ids in benchmark exist in corpus.

        This test verifies data integrity by ensuring that every document
        referenced in the benchmark's ground truth context IDs actually
        exists in the loaded corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.
        """
        # Load CLAP-NQ data (using train split for faster testing)
        loader = ClapNqDataLoader(split="train")

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Extract all document IDs from corpus
        corpus_doc_ids = {doc.name for doc in corpus.documents}

        # Extract all ground-truth document IDs from benchmark
        benchmark_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)

        # Verify all ground-truth documents exist in corpus
        missing_docs = benchmark_doc_ids - corpus_doc_ids

        assert len(missing_docs) == 0, (
            f"Found {len(missing_docs)} ground-truth documents missing from corpus: "
            f"{sorted(missing_docs)[:10]}..."  # Show first 10 missing docs
        )

        # Additional verification: ensure we have data
        assert len(corpus_doc_ids) > 0, "Corpus should not be empty"
        assert len(benchmark_doc_ids) > 0, "Benchmark should reference documents"

    def test_ground_truth_documents_exist_in_corpus_test_split(self):
        """
        Test ground-truth document validation for test split.

        Verifies that the test split (validation in HuggingFace) also maintains
        data integrity between corpus and benchmark.
        """
        # Load CLAP-NQ test split
        loader = ClapNqDataLoader(split="test")

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Extract document IDs
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        benchmark_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)

        # Verify all ground-truth documents exist in corpus
        missing_docs = benchmark_doc_ids - corpus_doc_ids

        assert len(missing_docs) == 0, (
            f"Found {len(missing_docs)} ground-truth documents missing from corpus "
            f"in test split: {sorted(missing_docs)[:10]}..."
        )
