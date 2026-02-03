"""
Integration tests for RealMM data loader.

This module contains integration tests that load real RealMM data from HuggingFace
and verify data integrity, particularly ensuring that all ground-truth documents
referenced in benchmark entries exist in the corpus.

Tests are parametrized to run against all 4 RealMM dataset variants.
"""

import pytest

from ragbench.datasets_loader.data_models.dataset_names import DatasetName
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.real_mm_rag_data_loader import RealMMRagDataLoader

# All RealMM dataset variants to test
REAL_MM_DATASETS = [
    DatasetName.REAL_MM_FIN_SLIDES,
    DatasetName.REAL_MM_FIN_REPORT,
    DatasetName.REAL_MM_TECH_REPORT,
    DatasetName.REAL_MM_TECH_SLIDES,
]


@pytest.mark.integration
class TestRealMMIntegration:
    """Integration tests for RealMM data loader with real data."""

    @pytest.mark.parametrize("dataset_name", REAL_MM_DATASETS)
    def test_ground_truth_documents_exist_in_corpus(self, dataset_name):
        """
        Test that all ground-truth doc_ids in benchmark exist in corpus.

        This test verifies data integrity by ensuring that every document
        (image) referenced in the benchmark's ground truth context IDs actually
        exists in the loaded corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.

        Args:
            dataset_name: RealMM dataset variant to test (parametrized).
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(dataset_name=dataset_name, split=None)

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
            f"[{dataset_name.value}] Found {len(missing_docs)} ground-truth documents "
            f"missing from corpus: {sorted(missing_docs)[:10]}..."
        )

        # Additional verification: ensure we have data
        assert (
            len(corpus_doc_ids) > 0
        ), f"[{dataset_name.value}] Corpus should not be empty"
        assert (
            len(benchmark_doc_ids) > 0
        ), f"[{dataset_name.value}] Benchmark should reference documents"

    @pytest.mark.parametrize("dataset_name", REAL_MM_DATASETS)
    def test_ground_truth_documents_exist_in_corpus_train_split(self, dataset_name):
        """
        Test ground-truth document validation for train split.

        Verifies that the train split (created from test split using get_benchmark_split)
        also maintains data integrity between corpus and benchmark.

        Args:
            dataset_name: RealMM dataset variant to test (parametrized).
        """
        # Load RealMM train split
        loader = RealMMRagDataLoader(dataset_name=dataset_name, split="train")

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Extract document IDs
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        benchmark_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)

        # Verify all ground-truth documents exist in corpus
        missing_docs = benchmark_doc_ids - corpus_doc_ids

        assert len(missing_docs) == 0, (
            f"[{dataset_name.value}] Found {len(missing_docs)} ground-truth documents "
            f"missing from corpus in train split: {sorted(missing_docs)[:10]}..."
        )

    @pytest.mark.parametrize("dataset_name", REAL_MM_DATASETS)
    def test_ground_truth_documents_exist_in_corpus_test_split(self, dataset_name):
        """
        Test ground-truth document validation for test split.

        Verifies that the test split (created from test split using get_benchmark_split)
        also maintains data integrity between corpus and benchmark.

        Args:
            dataset_name: RealMM dataset variant to test (parametrized).
        """
        # Load RealMM test split
        loader = RealMMRagDataLoader(dataset_name=dataset_name, split="test")

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Extract document IDs
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        benchmark_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)

        # Verify all ground-truth documents exist in corpus
        missing_docs = benchmark_doc_ids - corpus_doc_ids

        assert len(missing_docs) == 0, (
            f"[{dataset_name.value}] Found {len(missing_docs)} ground-truth documents "
            f"missing from corpus in test split: {sorted(missing_docs)[:10]}..."
        )

    @pytest.mark.parametrize("dataset_name", REAL_MM_DATASETS)
    def test_load_documents(self, dataset_name):
        """
        Test that documents can be loaded from the RealMM corpus.

        This test verifies that:
        - Documents can be loaded successfully
        - The corpus is not empty
        - Each document has required attributes
        - Documents are images with appropriate MIME types

        Args:
            dataset_name: RealMM dataset variant to test (parametrized).
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(dataset_name=dataset_name, split=None)

        # Get corpus
        corpus = loader.get_corpus()

        # Verify corpus is not empty
        assert (
            len(corpus.documents) > 0
        ), f"[{dataset_name.value}] Corpus should not be empty"

        # Verify document structure
        for doc in corpus.documents[:5]:  # Check first 5 documents
            assert (
                doc.name
            ), f"[{dataset_name.value}] Document should have a name (image filename)"
            assert (
                doc.stream
            ), f"[{dataset_name.value}] Document should have content stream"
            assert doc.mime_type.startswith(
                "image/"
            ), f"[{dataset_name.value}] Document should be an image, got {doc.mime_type}"

            # Verify content can be read
            content = doc.stream.read()
            assert (
                len(content) > 0
            ), f"[{dataset_name.value}] Document content should not be empty"
            assert isinstance(
                content, bytes
            ), f"[{dataset_name.value}] Document content should be bytes"

    @pytest.mark.parametrize("dataset_name", REAL_MM_DATASETS)
    def test_load_benchmark_entries(self, dataset_name):
        """
        Test that benchmark entries can be loaded from the RealMM dataset.

        This test verifies that:
        - Benchmark entries can be loaded successfully
        - Each entry has required attributes
        - Questions and answers are properly formatted
        - Ground truth context IDs reference image filenames

        Args:
            dataset_name: RealMM dataset variant to test (parametrized).
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(dataset_name=dataset_name, split=None)

        # Get benchmark
        benchmark = loader.get_benchmark()

        # Verify benchmark is not empty
        assert (
            len(benchmark.benchmark_entries) > 0
        ), f"[{dataset_name.value}] Benchmark should not be empty"

        # Verify benchmark entry structure
        for entry in benchmark.benchmark_entries[:5]:  # Check first 5 entries
            assert (
                entry.question_id
            ), f"[{dataset_name.value}] Entry should have a question ID"
            assert (
                entry.question
            ), f"[{dataset_name.value}] Entry should have a question"
            assert (
                len(entry.ground_truth_answers) > 0
            ), f"[{dataset_name.value}] Entry should have answers"
            assert (
                entry.is_answerable
            ), f"[{dataset_name.value}] All RealMM questions should be answerable"

            # Verify ground truth context IDs exist and reference images
            assert (
                len(entry.ground_truth_context_ids) > 0
            ), f"[{dataset_name.value}] Entry should have ground truth context IDs"

            for context_id in entry.ground_truth_context_ids:
                assert (
                    context_id.document_id
                ), f"[{dataset_name.value}] Context ID should have document_id"
                # Verify it looks like an image filename
                assert any(
                    context_id.document_id.endswith(ext)
                    for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
                ), f"[{dataset_name.value}] Document ID should be an image filename: {context_id.document_id}"

    @pytest.mark.parametrize("dataset_name", REAL_MM_DATASETS)
    def test_document_ids_are_unique(self, dataset_name):
        """
        Test that all document IDs in the corpus are unique.

        This test verifies data integrity by ensuring no duplicate document IDs.

        Args:
            dataset_name: RealMM dataset variant to test (parametrized).
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(dataset_name=dataset_name, split=None)

        # Get corpus
        corpus = loader.get_corpus()

        # Extract all document IDs
        doc_ids = [doc.name for doc in corpus.documents]

        # Verify uniqueness
        unique_doc_ids = set(doc_ids)
        assert len(doc_ids) == len(
            unique_doc_ids
        ), f"[{dataset_name.value}] Found {len(doc_ids) - len(unique_doc_ids)} duplicate document IDs"

    @pytest.mark.parametrize("dataset_name", REAL_MM_DATASETS)
    def test_question_ids_are_unique(self, dataset_name):
        """
        Test that all question IDs in the benchmark are unique.

        This test verifies data integrity by ensuring no duplicate question IDs.

        Args:
            dataset_name: RealMM dataset variant to test (parametrized).
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(dataset_name=dataset_name, split=None)

        # Get benchmark
        benchmark = loader.get_benchmark()

        # Extract all question IDs
        question_ids = [entry.question_id for entry in benchmark.benchmark_entries]

        # Verify uniqueness
        unique_question_ids = set(question_ids)
        assert len(question_ids) == len(
            unique_question_ids
        ), f"[{dataset_name.value}] Found {len(question_ids) - len(unique_question_ids)} duplicate question IDs"

    @pytest.mark.parametrize("dataset_name", REAL_MM_DATASETS)
    def test_answers_are_not_empty(self, dataset_name):
        """
        Test that all benchmark entries have non-empty answers.

        This test verifies that every question has at least one answer.

        Args:
            dataset_name: RealMM dataset variant to test (parametrized).
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(dataset_name=dataset_name, split=None)

        # Get benchmark
        benchmark = loader.get_benchmark()

        # Verify all entries have answers
        for entry in benchmark.benchmark_entries:
            assert (
                len(entry.ground_truth_answers) > 0
            ), f"[{dataset_name.value}] Question {entry.question_id} has no answers"
            assert all(
                answer.strip() for answer in entry.ground_truth_answers
            ), f"[{dataset_name.value}] Question {entry.question_id} has empty answer strings"
