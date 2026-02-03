"""
Integration tests for RealMM data loader.

This module contains integration tests that load real RealMM data from HuggingFace
and verify data integrity, particularly ensuring that all ground-truth documents
referenced in benchmark entries exist in the corpus.
"""

import pytest

from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.real_mm_rag_data_loader import RealMMRagDataLoader


@pytest.mark.integration
class TestRealMMIntegration:
    """Integration tests for RealMM data loader with real data."""

    def test_ground_truth_documents_exist_in_corpus(self):
        """
        Test that all ground-truth doc_ids in benchmark exist in corpus.

        This test verifies data integrity by ensuring that every document
        (image) referenced in the benchmark's ground truth context IDs actually
        exists in the loaded corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.
        """
        # Load RealMM data (no split parameter as dataset only has test split)
        loader = RealMMRagDataLoader(split=None)

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

    def test_ground_truth_documents_exist_in_corpus_train_split(self):
        """
        Test ground-truth document validation for train split.

        Verifies that the train split (created from test split using get_benchmark_split)
        also maintains data integrity between corpus and benchmark.
        """
        # Load RealMM train split
        loader = RealMMRagDataLoader(split="train")

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
            f"in train split: {sorted(missing_docs)[:10]}..."
        )

    def test_ground_truth_documents_exist_in_corpus_test_split(self):
        """
        Test ground-truth document validation for test split.

        Verifies that the test split (created from test split using get_benchmark_split)
        also maintains data integrity between corpus and benchmark.
        """
        # Load RealMM test split
        loader = RealMMRagDataLoader(split="test")

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

    def test_load_documents(self):
        """
        Test that documents can be loaded from the RealMM corpus.

        This test verifies that:
        - Documents can be loaded successfully
        - The corpus is not empty
        - Each document has required attributes
        - Documents are images with appropriate MIME types
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(split=None)

        # Get corpus
        corpus = loader.get_corpus()

        # Verify corpus is not empty
        assert len(corpus.documents) > 0, "Corpus should not be empty"

        # Verify document structure
        for doc in corpus.documents[:5]:  # Check first 5 documents
            assert doc.name, "Document should have a name (image filename)"
            assert doc.stream, "Document should have content stream"
            assert doc.mime_type.startswith(
                "image/"
            ), f"Document should be an image, got {doc.mime_type}"

            # Verify content can be read
            content = doc.stream.read()
            assert len(content) > 0, "Document content should not be empty"
            assert isinstance(content, bytes), "Document content should be bytes"

    def test_load_benchmark_entries(self):
        """
        Test that benchmark entries can be loaded from the RealMM dataset.

        This test verifies that:
        - Benchmark entries can be loaded successfully
        - Each entry has required attributes
        - Questions and answers are properly formatted
        - Ground truth context IDs reference image filenames
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(split=None)

        # Get benchmark
        benchmark = loader.get_benchmark()

        # Verify benchmark is not empty
        assert len(benchmark.benchmark_entries) > 0, "Benchmark should not be empty"

        # Verify benchmark entry structure
        for entry in benchmark.benchmark_entries[:5]:  # Check first 5 entries
            assert entry.question_id, "Entry should have a question ID"
            assert entry.question, "Entry should have a question"
            assert len(entry.ground_truth_answers) > 0, "Entry should have answers"
            assert entry.is_answerable, "All RealMM questions should be answerable"

            # Verify ground truth context IDs exist and reference images
            assert (
                len(entry.ground_truth_context_ids) > 0
            ), "Entry should have ground truth context IDs"

            for context_id in entry.ground_truth_context_ids:
                assert context_id.document_id, "Context ID should have document_id"
                # Verify it looks like an image filename
                assert any(
                    context_id.document_id.endswith(ext)
                    for ext in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]
                ), f"Document ID should be an image filename: {context_id.document_id}"

    def test_document_ids_are_unique(self):
        """
        Test that all document IDs in the corpus are unique.

        This test verifies data integrity by ensuring no duplicate document IDs.
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(split=None)

        # Get corpus
        corpus = loader.get_corpus()

        # Extract all document IDs
        doc_ids = [doc.name for doc in corpus.documents]

        # Verify uniqueness
        unique_doc_ids = set(doc_ids)
        assert len(doc_ids) == len(
            unique_doc_ids
        ), f"Found {len(doc_ids) - len(unique_doc_ids)} duplicate document IDs"

    def test_question_ids_are_unique(self):
        """
        Test that all question IDs in the benchmark are unique.

        This test verifies data integrity by ensuring no duplicate question IDs.
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(split=None)

        # Get benchmark
        benchmark = loader.get_benchmark()

        # Extract all question IDs
        question_ids = [entry.question_id for entry in benchmark.benchmark_entries]

        # Verify uniqueness
        unique_question_ids = set(question_ids)
        assert len(question_ids) == len(
            unique_question_ids
        ), f"Found {len(question_ids) - len(unique_question_ids)} duplicate question IDs"

    def test_answers_are_not_empty(self):
        """
        Test that all benchmark entries have non-empty answers.

        This test verifies that every question has at least one answer.
        """
        # Load RealMM data
        loader = RealMMRagDataLoader(split=None)

        # Get benchmark
        benchmark = loader.get_benchmark()

        # Verify all entries have answers
        for entry in benchmark.benchmark_entries:
            assert (
                len(entry.ground_truth_answers) > 0
            ), f"Question {entry.question_id} has no answers"
            assert all(
                answer.strip() for answer in entry.ground_truth_answers
            ), f"Question {entry.question_id} has empty answer strings"
