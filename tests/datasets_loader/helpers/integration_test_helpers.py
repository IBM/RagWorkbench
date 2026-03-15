"""
Reusable helper functions for integration tests of data loaders.

This module provides common validation functions to reduce code duplication
across integration tests for different data loaders.
"""

from ragworkbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragworkbench.datasets_loader.data_models.rag_corpus import RagCorpus


class IntegrationTestHelpers:
    """Collection of reusable integration test helper methods."""

    @staticmethod
    def assert_ground_truth_documents_exist(
        corpus: RagCorpus,
        benchmark: RagBenchmark,
        split_name: str | None = None,
    ) -> None:
        """
        Verify all ground-truth doc_ids in benchmark exist in corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.

        Args:
            corpus: The corpus containing documents
            benchmark: The benchmark containing questions and ground truth
            split_name: Optional name of the split being tested (for error messages)

        Raises:
            AssertionError: If any ground-truth documents are missing from corpus
        """
        # Extract all document IDs from corpus
        corpus_doc_ids = {doc.name for doc in corpus.documents}

        # Extract all ground-truth document IDs from benchmark
        benchmark_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)

        # Verify all ground-truth documents exist in corpus
        missing_docs = benchmark_doc_ids - corpus_doc_ids

        split_msg = f" in {split_name} split" if split_name else ""
        assert len(missing_docs) == 0, (
            f"Found {len(missing_docs)} ground-truth documents missing from corpus"
            f"{split_msg}: {sorted(missing_docs)[:10]}..."
        )

        # Additional verification: ensure we have data
        assert len(corpus_doc_ids) > 0, "Corpus should not be empty"
        assert len(benchmark_doc_ids) > 0, "Benchmark should reference documents"

    @staticmethod
    def assert_corpus_not_empty(corpus: RagCorpus, min_docs: int = 1) -> None:
        """
        Verify corpus has minimum number of documents.

        Args:
            corpus: The corpus to validate
            min_docs: Minimum number of documents expected

        Raises:
            AssertionError: If corpus has fewer than min_docs documents
        """
        assert len(corpus.documents) >= min_docs, (
            f"Corpus should have at least {min_docs} documents, "
            f"but has {len(corpus.documents)}"
        )

    @staticmethod
    def assert_benchmark_not_empty(
        benchmark: RagBenchmark, min_entries: int = 1
    ) -> None:
        """
        Verify benchmark has minimum number of entries.

        Args:
            benchmark: The benchmark to validate
            min_entries: Minimum number of entries expected

        Raises:
            AssertionError: If benchmark has fewer than min_entries entries
        """
        assert len(benchmark.benchmark_entries) >= min_entries, (
            f"Benchmark should have at least {min_entries} entries, "
            f"but has {len(benchmark.benchmark_entries)}"
        )

    @staticmethod
    def assert_document_ids_unique(corpus: RagCorpus) -> None:
        """
        Verify all document IDs in the corpus are unique.

        Args:
            corpus: The corpus to validate

        Raises:
            AssertionError: If duplicate document IDs are found
        """
        doc_ids = [doc.name for doc in corpus.documents]
        unique_doc_ids = set(doc_ids)

        assert len(doc_ids) == len(
            unique_doc_ids
        ), f"Found {len(doc_ids) - len(unique_doc_ids)} duplicate document IDs"

    @staticmethod
    def assert_question_ids_unique(benchmark: RagBenchmark) -> None:
        """
        Verify all question IDs in the benchmark are unique.

        Args:
            benchmark: The benchmark to validate

        Raises:
            AssertionError: If duplicate question IDs are found
        """
        question_ids = [entry.question_id for entry in benchmark.benchmark_entries]
        unique_question_ids = set(question_ids)

        assert len(question_ids) == len(
            unique_question_ids
        ), f"Found {len(question_ids) - len(unique_question_ids)} duplicate question IDs"

    @staticmethod
    def assert_documents_have_content(corpus: RagCorpus, sample_size: int = 10) -> None:
        """
        Verify documents have non-empty content.

        Args:
            corpus: The corpus to validate
            sample_size: Number of documents to check (checks first N documents)

        Raises:
            AssertionError: If any sampled documents have empty content
        """
        empty_docs = []
        for doc in corpus.documents[:sample_size]:
            # Read content from stream
            content = doc.stream.read()
            if len(content) == 0:
                empty_docs.append(doc.name)
            # Reset stream for potential future reads
            doc.stream.seek(0)

        assert (
            len(empty_docs) == 0
        ), f"Found {len(empty_docs)} documents with empty content: {empty_docs}"

    @staticmethod
    def assert_documents_have_metadata(
        corpus: RagCorpus, required_fields: list[str], sample_size: int = 10
    ) -> None:
        """
        Verify documents have required metadata fields.

        Args:
            corpus: The corpus to validate
            required_fields: List of metadata field names that must be present
            sample_size: Number of documents to check (checks first N documents)

        Raises:
            AssertionError: If any required metadata fields are missing
        """
        for doc in corpus.documents[:sample_size]:
            for field in required_fields:
                assert (
                    field in doc.metadata
                ), f"Document {doc.name} missing '{field}' in metadata"

                # Verify metadata values are strings
                assert isinstance(
                    doc.metadata[field], str
                ), f"Document {doc.name} metadata field '{field}' should be a string"

    @staticmethod
    def assert_entries_have_answers(benchmark: RagBenchmark) -> None:
        """
        Verify all benchmark entries have non-empty answers.

        Args:
            benchmark: The benchmark to validate

        Raises:
            AssertionError: If any entries have empty answers
        """
        for entry in benchmark.benchmark_entries:
            assert (
                len(entry.ground_truth_answers) > 0
            ), f"Question {entry.question_id} has no answers"
            assert all(
                answer.strip() for answer in entry.ground_truth_answers
            ), f"Question {entry.question_id} has empty answer strings"

    @staticmethod
    def assert_entries_are_answerable(benchmark: RagBenchmark) -> None:
        """
        Verify all benchmark entries are marked as answerable.

        Args:
            benchmark: The benchmark to validate

        Raises:
            AssertionError: If any entries are marked as unanswerable
        """
        unanswerable = [
            entry.question_id
            for entry in benchmark.benchmark_entries
            if not entry.is_answerable
        ]

        assert (
            len(unanswerable) == 0
        ), f"Found {len(unanswerable)} unanswerable questions: {unanswerable[:10]}"

    @staticmethod
    def assert_entries_have_ground_truth_contexts(
        benchmark: RagBenchmark, expected_count: int | None = None
    ) -> None:
        """
        Verify benchmark entries have ground truth context IDs.

        Args:
            benchmark: The benchmark to validate
            expected_count: If provided, verify each entry has exactly this many contexts

        Raises:
            AssertionError: If entries don't have expected ground truth contexts
        """
        for entry in benchmark.benchmark_entries:
            if expected_count is not None:
                assert len(entry.ground_truths_context_ids) == expected_count, (
                    f"Question {entry.question_id} expected {expected_count} "
                    f"ground truth contexts, got {len(entry.ground_truths_context_ids)}"
                )
            else:
                assert (
                    len(entry.ground_truths_context_ids) > 0
                ), f"Question {entry.question_id} has no ground truth contexts"
