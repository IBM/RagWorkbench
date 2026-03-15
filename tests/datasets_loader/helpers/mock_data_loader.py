"""
Mock data loader for testing purposes.

This module provides a concrete implementation of RagDataLoader
that can be used in tests without requiring actual dataset files.
"""

from io import BytesIO
from pathlib import Path

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)
from ragworkbench.datasets_loader.data_models.document_object import DocumentObject
from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)
from ragworkbench.datasets_loader.dataset_names import DatasetName
from ragworkbench.datasets_loader.datasets_utils import get_benchmark_split


class MockRagDataLoader(RagDataLoader):
    """
    Concrete implementation of RagDataLoader for testing purposes.

    This mock loader provides fixed test data for documents and benchmark entries,
    allowing us to test the abstract base class functionality without requiring
    actual dataset files.

    Attributes:
        num_docs: Number of mock documents to generate.
        num_questions: Number of mock questions to generate.

    Example:
        >>> loader = MockRagDataLoader(num_docs=10, num_questions=5)
        >>> corpus = loader.get_corpus()
        >>> benchmark = loader.get_benchmark()
        >>> print(f"Loaded {len(corpus)} docs and {len(benchmark)} questions")
    """

    def __init__(
        self,
        dataset_name: DatasetName = DatasetName.BIOASQ,
        split: DatasetSplit | None = None,
        data_sampling: DataSamplingParams = DataSamplingParams(),
        num_docs: int = 20,
        num_questions: int = 15,
        cache_dir: Path | None = None,
    ):
        """
        Initialize mock data loader with configurable test data size.

        Args:
            dataset_name: Dataset identifier (default: AI_ARXIV).
            split: Dataset split ('train', 'test', or None).
            data_sampling: Sampling parameters for questions and documents.
            num_docs: Number of documents to generate (default: 20).
            num_questions: Number of questions to generate (default: 15).
        """
        self.num_docs = num_docs
        self.num_questions = num_questions
        super().__init__(dataset_name, split, data_sampling, cache_dir=cache_dir)

    def _get_documents(self) -> list[DocumentObject]:
        """
        Generate mock documents for testing.

        Returns:
            List of DocumentObject instances with mock content.
        """
        documents = []
        for i in range(self.num_docs):
            doc = DocumentObject(
                name=f"mock_doc_{i}",
                stream=BytesIO(f"Content of mock document {i}".encode()),
                mime_type="application/pdf",
                metadata={"index": i, "category": f"cat_{i % 3}"},
            )
            documents.append(doc)
        return documents

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        """
        Generate mock benchmark entries for testing.

        Args:
            split: Dataset split to generate ('train', 'test', or None).

        Returns:
            List of RagBenchmarkEntry instances with mock questions and answers.
        """
        entries = []
        # Use first 10 documents as ground truth (or fewer if num_docs < 10)
        num_gt_docs = min(10, self.num_docs)

        for i in range(self.num_questions):
            # Cycle through ground truth documents
            doc_id = f"mock_doc_{i % num_gt_docs}"
            entry = RagBenchmarkEntry(
                question_id=f"mock_q_{i}",
                question=f"Mock question {i} about document {i % num_gt_docs}?",
                ground_truth_answers=[f"Mock answer {i}"],
                ground_truths_context_ids=[GroundTruthContextId(document_id=doc_id)],
                is_answerable=True,
            )
            entries.append(entry)

        # Handle the split case
        return get_benchmark_split(entries, split)
