"""
Integration tests for Mini Wikipedia data loader.

This module contains integration tests that load real Mini Wikipedia data from HuggingFace
and verify data integrity and functionality.
"""

import pytest

from ragbench.datasets_loader.miniwiki_data_loader import MiniWikiDataLoader


@pytest.mark.integration
class TestMiniWikiIntegration:
    """Integration tests for Mini Wikipedia data loader with real data."""

    def test_load_documents(self):
        """
        Test that documents can be loaded from the Mini Wikipedia corpus.

        This test verifies that:
        - Documents can be loaded successfully
        - The corpus is not empty
        - Each document has required attributes
        """
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")

        # Get corpus
        corpus = loader.get_corpus()

        # Verify corpus is not empty
        assert len(corpus.documents) > 0, "Corpus should not be empty"

        # Verify document structure
        for doc in corpus.documents[:5]:  # Check first 5 documents
            assert doc.name, "Document should have a name (ID)"
            assert doc.stream, "Document should have content stream"
            assert doc.mime_type == "text/plain", "Document should be text/plain"

            # Verify content can be read
            content = doc.stream.read()
            assert len(content) > 0, "Document content should not be empty"
            assert isinstance(content, bytes), "Document content should be bytes"

    def test_load_benchmark_entries(self):
        """
        Test that benchmark entries can be loaded from the Mini Wikipedia dataset.

        This test verifies that:
        - Benchmark entries can be loaded successfully
        - Each entry has required attributes
        - Questions and answers are properly formatted
        """
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")

        # Get benchmark
        benchmark = loader.get_benchmark()

        # Verify benchmark is not empty
        assert len(benchmark.benchmark_entries) > 0, "Benchmark should not be empty"

        # Verify benchmark entry structure
        for entry in benchmark.benchmark_entries[:5]:  # Check first 5 entries
            assert entry.question_id, "Entry should have a question ID"
            assert entry.question, "Entry should have a question"
            assert len(entry.ground_truth_answers) > 0, "Entry should have answers"
            assert entry.is_answerable, "All Mini Wiki questions should be answerable"

            # Verify ground truth context IDs is empty (not available in dataset)
            assert (
                len(entry.ground_truth_context_ids) == 0
            ), "Mini Wiki dataset does not have ground truth context IDs"

    def test_corpus_and_benchmark_loaded_together(self):
        """
        Test that both corpus and benchmark can be loaded together.

        This test verifies that:
        - Both corpus and benchmark can be loaded in the same loader instance
        - Data is consistent between calls
        """
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")

        # Get both corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify both are loaded
        assert len(corpus.documents) > 0, "Corpus should be loaded"
        assert len(benchmark.benchmark_entries) > 0, "Benchmark should be loaded"

        # Verify data consistency - calling again should return same data
        corpus2 = loader.get_corpus()
        benchmark2 = loader.get_benchmark()

        assert len(corpus.documents) == len(
            corpus2.documents
        ), "Corpus should be consistent"
        assert len(benchmark.benchmark_entries) == len(
            benchmark2.benchmark_entries
        ), "Benchmark should be consistent"

    def test_document_ids_are_unique(self):
        """
        Test that all document IDs in the corpus are unique.

        This test verifies data integrity by ensuring no duplicate document IDs.
        """
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")

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
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")

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
        # Load Mini Wikipedia data
        loader = MiniWikiDataLoader(split="train")

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
