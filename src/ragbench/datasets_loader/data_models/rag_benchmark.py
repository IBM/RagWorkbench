from typing import Any

from pydantic import BaseModel, Field


class GroundTruthContextId(BaseModel):
    """
    Identifies a specific context location within a document corpus.

    This class represents a reference to ground truth context that answers a question,
    pinpointing the exact location within a document (and optionally a specific page
    or table within that document).

    Attributes:
        document_id: Unique identifier for the document containing the ground truth.
        page: Optional page number within the document (1-indexed).
        table_id: Optional identifier for a specific table within the document.
    """

    document_id: str = Field(
        frozen=True,
        min_length=1,
        description="Unique identifier for the document containing the ground truth context.",
    )
    page: int | None = Field(
        frozen=True,
        default=None,
        ge=1,
        description="Optional page number within the document where the context is located (1-indexed).",
    )
    table_id: str | None = Field(
        frozen=True,
        default=None,
        description="Optional identifier for a specific table within the document.",
    )


class RagBenchmarkEntry(BaseModel):
    """
    Represents a single question-answer pair in a RAG benchmark dataset.

    Each entry contains a question, its ground truth answers, references to the
    source contexts, and metadata about answerability. This structure enables
    comprehensive evaluation of RAG systems including both retrieval and generation.

    Attributes:
        question_id: Unique identifier for this question.
        question: The question text to be answered.
        ground_truth_answers: List of acceptable answer strings, or None if not applicable.
        ground_truths_context_ids: References to document locations containing the answer.
        is_answerable: Whether the question can be answered from the corpus.
        additional_information: Optional metadata for extended analysis.
    """

    question_id: str = Field(
        frozen=True,
        description="Unique identifier for this benchmark question.",
    )
    question: str = Field(
        frozen=True,
        description="The question text to be answered by the RAG system.",
    )
    ground_truth_answers: list[str] | None = Field(
        frozen=True,
        default=None,
        description="List of acceptable ground truth answer strings. "
        "None if answers are not provided or question is unanswerable.",
    )
    ground_truths_context_ids: list[GroundTruthContextId] = Field(
        frozen=True,
        default_factory=list,
        description="References to document locations containing the ground truth context. "
        "Empty list if no specific contexts are identified.",
    )
    is_answerable: bool = Field(
        frozen=True,
        default=True,
        description="Whether this question can be answered from the document corpus. "
        "False for questions designed to test handling of unanswerable queries.",
    )
    additional_information: dict[str, Any] | None = Field(
        frozen=True,
        default=None,
        description="Optional metadata for extended analysis, such as question category, "
        "difficulty level, or domain-specific annotations.",
    )


class RagBenchmark(BaseModel):
    """
    A complete RAG benchmark dataset containing questions and ground truth data.

    This class manages a collection of benchmark entries and provides utility methods
    for filtering, querying, and analyzing the benchmark data. It ensures at least
    one entry exists and provides convenient access patterns for evaluation workflows.

    Attributes:
        benchmark_entries: List of benchmark question-answer entries.

    Example:
        >>> my_benchmark = RagBenchmark(benchmark_entries=[...])
        >>> answerable_questions = my_benchmark.get_questions(answerable_queries_only=True)
        >>> doc_ids = RagBenchmark.get_doc_ids_set(my_benchmark.benchmark_entries)
        >>> print(f"Benchmark contains {len(my_benchmark)} questions")
    """

    benchmark_entries: list[RagBenchmarkEntry] = Field(
        frozen=True,
        min_length=1,
        description="List of benchmark entries. Must contain at least one entry.",
    )

    def get_question_ids(self, answerable_queries_only: bool = False) -> list[str]:
        """
        Extract question IDs from the benchmark.

        Args:
            answerable_queries_only: If True, return only IDs of answerable questions.
                                    If False, return all question IDs.

        Returns:
            List of question ID strings.

        Example:
            >>> my_benchmark = RagBenchmark(benchmark_entries=[...])
            >>> my_benchmark.get_question_ids(answerable_queries_only=True)
            ['q1', 'q2', 'q5']
        """
        return [
            q.question_id
            for q in self.get_benchmark_entries(
                answerable_queries_only=answerable_queries_only
            )
        ]

    def get_questions(self, answerable_queries_only: bool = False) -> list[str]:
        """
        Extract question texts from the benchmark.

        Args:
            answerable_queries_only: If True, return only answerable questions.
                                    If False, return all questions.

        Returns:
            List of question text strings.

        Example:
            >>> my_benchmark = RagBenchmark(benchmark_entries=[...])
            >>> my_benchmark.get_questions(answerable_queries_only=False)
            ['What is the capital?', 'Who invented the telephone?', ...]
        """
        return [
            q.question
            for q in self.get_benchmark_entries(
                answerable_queries_only=answerable_queries_only
            )
        ]

    def get_benchmark_entries(
        self, answerable_queries_only: bool = False
    ) -> list[RagBenchmarkEntry]:
        """
        Retrieve benchmark entries with optional filtering.

        Args:
            answerable_queries_only: If True, return only entries where is_answerable=True.
                                    If False, return all entries.

        Returns:
            List of RagBenchmarkEntry objects matching the filter criteria.

        Example:
            >>> my_benchmark = RagBenchmark(benchmark_entries=[...])
            >>> entries = my_benchmark.get_benchmark_entries(answerable_queries_only=True)
            >>> print(f"Found {len(entries)} answerable questions")
        """
        return [
            e
            for e in self.benchmark_entries
            if (not answerable_queries_only or e.is_answerable)
        ]

    @staticmethod
    def get_doc_ids_set(rag_benchmark_entries: list[RagBenchmarkEntry]) -> set[str]:
        """
        Extract unique document IDs referenced in benchmark entries.

        This method collects all unique document IDs from the ground truth context
        references across all provided benchmark entries.

        Args:
            rag_benchmark_entries: List of benchmark entries to process.

        Returns:
            Set of unique document ID strings referenced in the entries.

        Example:
            >>> my_benchmark = RagBenchmark(benchmark_entries=[...])
            >>> entries = my_benchmark.get_benchmark_entries()
            >>> doc_ids = RagBenchmark.get_doc_ids_set(entries)
            >>> print(f"Benchmark references {len(doc_ids)} unique documents")
        """
        benchmark_doc_ids: set[str] = set()
        entry: RagBenchmarkEntry
        for entry in rag_benchmark_entries:
            context: GroundTruthContextId
            for context in entry.ground_truths_context_ids:
                benchmark_doc_ids.add(context.document_id)
        return benchmark_doc_ids

    def __len__(self) -> int:
        """
        Return the number of benchmark entries.

        Returns:
            Integer count of benchmark entries.

        Example:
            >>> my_benchmark = RagBenchmark(benchmark_entries=[...])
            >>> len(my_benchmark)
            150
        """
        return len(self.benchmark_entries)
