"""
Shared pytest fixtures for datasets tests.

This module provides reusable fixtures for testing the datasets module,
including sample documents, benchmark entries, and utility functions.
"""

from io import BytesIO
from pathlib import Path

import pytest

from datasets.data_models.data_sampling_params import DataSamplingParams
from datasets.data_models.document_object import DocumentObject
from datasets.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmark,
    RagBenchmarkEntry,
)
from datasets.data_models.rag_corpus import RagCorpus


@pytest.fixture
def sample_document_objects() -> list[DocumentObject]:
    """
    Create a list of sample DocumentObject instances for testing.

    Returns:
        List of 5 DocumentObject instances with various MIME types.
    """
    documents = []
    for i in range(5):
        doc = DocumentObject(
            name=f"doc_{i}",
            stream=BytesIO(f"Content of document {i}".encode()),
            mime_type="application/pdf",
            metadata={"index": i, "category": f"category_{i % 2}"},
        )
        documents.append(doc)
    return documents


@pytest.fixture
def sample_ground_truth_context_ids() -> list[GroundTruthContextId]:
    """
    Create a list of sample GroundTruthContextId instances.

    Returns:
        List of 5 GroundTruthContextId instances with various configurations.
    """
    return [
        GroundTruthContextId(document_id="doc_0", page=1),
        GroundTruthContextId(document_id="doc_1", page=2, table_id="table_1"),
        GroundTruthContextId(document_id="doc_2"),
        GroundTruthContextId(document_id="doc_3", page=5),
        GroundTruthContextId(document_id="doc_4", table_id="table_2"),
    ]


@pytest.fixture
def sample_benchmark_entries(
    sample_ground_truth_context_ids: list[GroundTruthContextId],
) -> list[RagBenchmarkEntry]:
    """
    Create a list of sample RagBenchmarkEntry instances.

    Includes both answerable and unanswerable questions with various
    ground truth configurations.

    Args:
        sample_ground_truth_context_ids: Fixture providing context IDs.

    Returns:
        List of 6 RagBenchmarkEntry instances (4 answerable, 2 unanswerable).
    """
    return [
        # Answerable questions
        RagBenchmarkEntry(
            question_id="q_0",
            question="What is the content of document 0?",
            ground_truth_answers=["Content of document 0"],
            ground_truth_context_ids=[sample_ground_truth_context_ids[0]],
            is_answerable=True,
        ),
        RagBenchmarkEntry(
            question_id="q_1",
            question="What is in the table of document 1?",
            ground_truth_answers=["Table data", "Data from table"],
            ground_truth_context_ids=[sample_ground_truth_context_ids[1]],
            is_answerable=True,
            additional_information={"difficulty": "medium"},
        ),
        RagBenchmarkEntry(
            question_id="q_2",
            question="Describe document 2.",
            ground_truth_answers=["Document 2 description"],
            ground_truth_context_ids=[sample_ground_truth_context_ids[2]],
            is_answerable=True,
        ),
        RagBenchmarkEntry(
            question_id="q_3",
            question="What is mentioned on page 5 of document 3?",
            ground_truth_answers=["Page 5 content"],
            ground_truth_context_ids=[sample_ground_truth_context_ids[3]],
            is_answerable=True,
        ),
        # Unanswerable questions
        RagBenchmarkEntry(
            question_id="q_4",
            question="What is the weather today?",
            ground_truth_answers=None,
            ground_truth_context_ids=[],
            is_answerable=False,
        ),
        RagBenchmarkEntry(
            question_id="q_5",
            question="Who is the president?",
            ground_truth_answers=None,
            ground_truth_context_ids=[],
            is_answerable=False,
            additional_information={"category": "out_of_scope"},
        ),
    ]


@pytest.fixture
def sample_rag_benchmark(
    sample_benchmark_entries: list[RagBenchmarkEntry],
) -> RagBenchmark:
    """
    Create a sample RagBenchmark instance.

    Args:
        sample_benchmark_entries: Fixture providing benchmark entries.

    Returns:
        RagBenchmark instance with sample entries.
    """
    return RagBenchmark(benchmark_entries=sample_benchmark_entries)


@pytest.fixture
def sample_rag_corpus(sample_document_objects: list[DocumentObject]) -> RagCorpus:
    """
    Create a sample RagCorpus instance.

    Args:
        sample_document_objects: Fixture providing document objects.

    Returns:
        RagCorpus instance with sample documents.
    """
    return RagCorpus(documents=sample_document_objects)


@pytest.fixture
def temp_export_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory for file export tests.

    Args:
        tmp_path: pytest's built-in temporary directory fixture.

    Returns:
        Path to a temporary export directory.
    """
    export_dir = tmp_path / "exports"
    export_dir.mkdir(exist_ok=True)
    return export_dir


@pytest.fixture
def sample_data_sampling_params() -> DataSamplingParams:
    """
    Create a sample DataSamplingParams instance with default values.

    Returns:
        DataSamplingParams instance with default configuration.
    """
    return DataSamplingParams()


@pytest.fixture
def sample_data_sampling_params_with_limits() -> DataSamplingParams:
    """
    Create a DataSamplingParams instance with sampling limits.

    Returns:
        DataSamplingParams with question_limit=3, document_factor=2, seed=42.
    """
    return DataSamplingParams(question_limit=3, document_factor=2, seed=42)


@pytest.fixture
def large_document_set() -> list[DocumentObject]:
    """
    Create a large set of documents for sampling tests.

    Returns:
        List of 20 DocumentObject instances.
    """
    documents = []
    for i in range(20):
        doc = DocumentObject(
            name=f"large_doc_{i}",
            stream=BytesIO(f"Content of large document {i}".encode()),
            mime_type="text/plain",
            metadata={"index": i},
        )
        documents.append(doc)
    return documents


@pytest.fixture
def large_benchmark_entries() -> list[RagBenchmarkEntry]:
    """
    Create a large set of benchmark entries for sampling tests.

    Returns:
        List of 15 RagBenchmarkEntry instances.
    """
    entries = []
    for i in range(15):
        # Use documents 0-9 as ground truth (10 documents)
        doc_id = f"large_doc_{i % 10}"
        entry = RagBenchmarkEntry(
            question_id=f"large_q_{i}",
            question=f"Question {i} about document {i % 10}?",
            ground_truth_answers=[f"Answer {i}"],
            ground_truth_context_ids=[GroundTruthContextId(document_id=doc_id)],
            is_answerable=True,
        )
        entries.append(entry)
    return entries
