"""
Integration tests for WatsonX DocsQA data loader.

This module contains integration tests that load real WatsonX DocsQA data from HuggingFace
and verify data integrity, particularly ensuring that all ground-truth documents
referenced in benchmark entries exist in the corpus.
"""

import pytest

from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.watsonx_data_loader import WatsonxDocsQADataLoader


@pytest.mark.integration
class TestWatsonxDocsQAIntegration:
    """Integration tests for WatsonX DocsQA data loader with real data."""

    def test_load_watsonx_docs_qa_dataset(self):
        """
        Test that WatsonX DocsQA dataset can be loaded successfully.

        This test verifies that the loader can successfully load the dataset
        from HuggingFace and that the data has the expected structure.
        """
        # Load WatsonX DocsQA data
        loader = WatsonxDocsQADataLoader()

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify we have data
        assert len(corpus) > 0, "Corpus should not be empty"
        assert len(benchmark) > 0, "Benchmark should not be empty"

        # Verify expected dataset size (approximately)
        # The dataset should have around 1,144 documents and 75 Q&A pairs
        assert len(corpus) > 1000, f"Expected ~1,144 documents, got {len(corpus)}"
        assert len(benchmark) > 70, f"Expected ~75 questions, got {len(benchmark)}"

    def test_ground_truth_documents_exist_in_corpus(self):
        """
        Test that all ground-truth doc_ids in benchmark exist in corpus.

        This test verifies data integrity by ensuring that every document
        referenced in the benchmark's ground truth context IDs actually
        exists in the loaded corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.
        """
        # Load WatsonX DocsQA data
        loader = WatsonxDocsQADataLoader()

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

        Verifies that the train split maintains data integrity between
        corpus and benchmark.
        """
        # Load WatsonX DocsQA train split
        loader = WatsonxDocsQADataLoader(split="train")

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

        Verifies that the test split also maintains data integrity between
        corpus and benchmark.
        """
        # Load WatsonX DocsQA test split
        loader = WatsonxDocsQADataLoader(split="test")

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

    def test_document_metadata_structure(self):
        """
        Test that documents have the expected metadata structure.

        Verifies that each document has the required metadata fields
        (title and url) as specified in the implementation.
        """
        # Load WatsonX DocsQA data
        loader = WatsonxDocsQADataLoader()
        corpus = loader.get_corpus()

        # Check first few documents for metadata
        for doc in corpus.documents[:10]:
            assert (
                "title" in doc.metadata
            ), f"Document {doc.name} missing 'title' in metadata"
            assert (
                "url" in doc.metadata
            ), f"Document {doc.name} missing 'url' in metadata"

            # Verify metadata values are strings
            assert isinstance(doc.metadata["title"], str), "Title should be a string"
            assert isinstance(doc.metadata["url"], str), "URL should be a string"

    def test_benchmark_entry_structure(self):
        """
        Test that benchmark entries have the expected structure.

        Verifies that each benchmark entry has:
        - A question ID
        - A question text
        - At least one ground truth answer
        - Exactly one ground truth context ID (single document reference)
        - is_answerable set to True
        """
        # Load WatsonX DocsQA data
        loader = WatsonxDocsQADataLoader()
        benchmark = loader.get_benchmark()

        # Check first few entries
        for entry in benchmark.benchmark_entries[:10]:
            # Verify basic structure
            assert entry.question_id, "Question ID should not be empty"
            assert entry.question, "Question should not be empty"
            assert entry.ground_truth_answers, "Should have at least one answer"
            assert (
                len(entry.ground_truth_answers) > 0
            ), "Should have at least one answer"

            # Verify single document reference (as per task specification)
            assert len(entry.ground_truth_context_ids) == 1, (
                f"Expected exactly 1 ground truth document, "
                f"got {len(entry.ground_truth_context_ids)}"
            )

            # Verify all questions are answerable
            assert (
                entry.is_answerable is True
            ), "All WatsonX questions should be answerable"

    def test_document_content_not_empty(self):
        """
        Test that documents have non-empty content.

        Verifies that the document content is properly loaded and not empty.
        """
        # Load WatsonX DocsQA data
        loader = WatsonxDocsQADataLoader()
        corpus = loader.get_corpus()

        # Check that documents have content
        empty_docs = []
        for doc in corpus.documents[:20]:  # Check first 20 documents
            # Read content from stream
            content = doc.stream.read()
            if len(content) == 0:
                empty_docs.append(doc.name)
            # Reset stream for potential future reads
            doc.stream.seek(0)

        assert (
            len(empty_docs) == 0
        ), f"Found {len(empty_docs)} documents with empty content: {empty_docs}"
