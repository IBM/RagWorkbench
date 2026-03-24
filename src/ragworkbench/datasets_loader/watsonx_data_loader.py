"""
WatsonX DocsQA dataset loader implementation.

This module provides a data loader for the WatsonX DocsQA dataset from IBM Research,
which contains enterprise product documentation for RAG evaluation.

References:
    - HuggingFace: https://huggingface.co/datasets/ibm-research/watsonxDocsQA
    - Paper: https://arxiv.org/abs/2505.03452
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Literal

from datasets import (  # type: ignore[import-not-found]
    concatenate_datasets,
    load_dataset,
)

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

logger = logging.getLogger(__name__)


class WatsonxDocsQADataLoader(RagDataLoader):
    """
    Data loader for the WatsonX DocsQA dataset.

    This loader handles loading and processing of the WatsonX DocsQA dataset from
    HuggingFace, including enterprise product documentation and question-answer pairs
    with ground truth document references.

    The WatsonX DocsQA dataset includes:
    - 1,144 text and markdown documents from enterprise product documentation
    - 75 question-answer pairs with gold document labels
    - Questions crafted by subject matter experts and synthetically generated

    Attributes:
        dataset_name: Set to DatasetName.WATSONX_DOCS_QA.
        document_format: Format of documents ('text', 'markdown', or 'html').
        split: Dataset split ('train', 'test', or None for all).

    Example:
        >>> loader = WatsonxDocsQADataLoader(
        ...     document_format="text",
        ...     split="train",
        ...     sampling_params=DataSamplingParams(question_limit=50)
        ... )
        >>> corpus = loader.get_corpus()
        >>> benchmark = loader.get_benchmark()
        >>> print(f"Loaded {len(corpus)} documents and {len(benchmark)} questions")

    Note:
        The dataset is loaded from HuggingFace: ibm-research/watsonxDocsQA
        - Corpus subset: Contains doc_id, title, url, and document content
        - Question_answers subset: Contains question_id, question, correct_answer,
          and correct_answer_document_ids
    """

    def __init__(
        self,
        document_format: Literal["text", "markdown", "html"],
        split: DatasetSplit | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        """
        Initialize the WatsonX DocsQA data loader.

        Args:
            document_format: Format of documents to load ('text', 'markdown', or 'html').
                           Determines which document column to use from the corpus.
            split: Dataset split to load ('train', 'test', or None for all).
                   Note: The dataset may only have a 'train' split available.
            sampling_params: Parameters controlling question and document sampling.
                           Defaults to no sampling (all data included).
            cache_dir: Optional cache directory for dataset storage.

        Note:
            Documents are loaded from the 'corpus' subset.
            Benchmark entries are loaded from the 'question_answers' subset.
        """
        logger.info(f"Initializing WatsonxDocsQADataLoader with split='{split}'")
        self.document_format = document_format

        # Select dataset name based on document format
        dataset_name_map = {
            "text": DatasetName.WATSONX_DOCS_QA_TXT,
            "html": DatasetName.WATSONX_DOCS_QA_HTML,
            "markdown": DatasetName.WATSONX_DOCS_QA_MD,
        }
        dataset_name = dataset_name_map[document_format]
        match self.document_format:
            case "text":
                self.suffix = ".txt"
            case "html":
                self.suffix = ".html"
            case "markdown":
                self.suffix = ".md"
            case _:
                raise ValueError(f"Unsupported document format: {self.document_format}")
        super().__init__(dataset_name, split, sampling_params, cache_dir=cache_dir)

    def _get_documents(self) -> list[DocumentObject]:
        """
        Load documents from the WatsonX DocsQA corpus.

        This method loads the document corpus from the HuggingFace dataset
        "ibm-research/watsonxDocsQA" subset "corpus". Each document contains
        enterprise product documentation with metadata.

        Returns:
            List of DocumentObject instances, where each document has:
            - name: The document ID (from 'doc_id' column)
            - content: The document text (from 'document' column) as bytes
            - mime_type: Set to 'text/plain' for text documents
            - metadata: Contains 'title' and 'url' fields from the dataset

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.

        Note:
            The corpus subset contains 1,144 documents with columns:
            - doc_id: Document identifier
            - title: Document title
            - url: Source URL
            - document: Document content (text)
            - document_markdown: Markdown formatted content
            - document_html: HTML formatted content
        """
        hf_path = "ibm-research/watsonxDocsQA"
        subset = "corpus"

        logger.info(f"Loading documents from HuggingFace dataset: {hf_path}/{subset}")

        # Load the corpus from HuggingFace
        dataset = load_dataset(hf_path, subset)

        # Get the first available split (typically 'train')
        available_splits = list(dataset.keys())
        if not available_splits:
            raise Exception(f"No splits found in dataset {hf_path}/{subset}")

        split_name = available_splits[0]
        corpus_data = dataset[split_name]

        logger.info(
            f"Found {len(corpus_data)} documents in split '{split_name}' "
            f"of {hf_path}/{subset}"
        )

        # Create DocumentObject for each row
        documents: list[DocumentObject] = []
        for row in corpus_data:
            # Extract fields from the row
            doc_id = str(row["doc_id"])
            title = row.get("title", "")
            url = row.get("url", "")
            document_content = None
            mime_type = None
            match self.document_format:
                case "text":
                    document_content = row["document"]
                    mime_type = "text/plain"
                case "html":
                    document_content = row["html_document"]
                    mime_type = "text/html"
                case "markdown":
                    document_content = row["md_document"]
                    mime_type = "text/markdown"
                case _:
                    raise ValueError(
                        f"Unsupported document format: {self.document_format}"
                    )

            # Ensure values are set
            assert (
                document_content is not None and mime_type is not None
            ), f"Failed to load document content for format: {self.document_format}"

            # Create DocumentObject with metadata
            # Store title and url in metadata as specified
            doc = DocumentObject(
                name=doc_id + self.suffix,
                stream=BytesIO(document_content.encode("utf-8")),
                mime_type=mime_type,
                metadata={"title": title, "url": url},
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from WatsonX DocsQA corpus")
        return documents

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        """
        Load question-answer pairs from the WatsonX DocsQA dataset.

        This method loads WatsonX DocsQA questions with their ground truth answers
        and document references from the HuggingFace dataset
        "ibm-research/watsonxDocsQA" subset "question_answers".

        Args:
            split: Dataset split to load ('train', 'test', or None for all).

        Returns:
            List of RagBenchmarkEntry instances with questions and answers.

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.

        Note:
            The question_answers subset contains 75 Q&A pairs with columns:
            - question_id: Question identifier
            - question: The question text
            - correct_answer: Single answer string
            - correct_answer_document_ids: Single document ID (string)

            Each question has exactly one correct answer and references one document.
            The dataset has pre-defined 'train' and 'test' splits.
        """
        hf_path = "ibm-research/watsonxDocsQA"
        subset = "question_answers"

        logger.info(
            f"Loading benchmark entries from HuggingFace dataset: {hf_path}/{subset}, "
            f"split='{split}'"
        )

        # Load the question-answer dataset from HuggingFace
        dataset = load_dataset(hf_path, subset)

        # Handle split parameter
        if split == "train":
            data = dataset["train"]
            logger.info(f"Loading train split with {len(data)} entries")
        elif split == "test":
            data = dataset["test"]
            logger.info(f"Loading test split with {len(data)} entries")
        else:
            # Concatenate both train and test splits
            train_data = dataset["train"]
            test_data = dataset["test"]
            data = concatenate_datasets([train_data, test_data])
            logger.info(
                f"Loading all splits: {len(train_data)} train + {len(test_data)} test "
                f"= {len(data)} total entries"
            )

        # Create RagBenchmarkEntry for each row
        entries: list[RagBenchmarkEntry] = []
        for row in data:
            # Extract fields from the row
            question_id = str(row["question_id"])
            question = row["question"]
            correct_answer = row["correct_answer"]
            # correct_answer_document_ids contains a single document ID
            correct_answer_doc_id = (
                str(row["correct_answer_document_ids"]) + self.suffix
            )

            # Create ground truth context ID from the single document ID
            ground_truth_context_ids = [
                GroundTruthContextId(document_id=correct_answer_doc_id)
            ]

            # Create RagBenchmarkEntry
            entry = RagBenchmarkEntry(
                question_id=question_id,
                question=question,
                ground_truth_answers=[correct_answer],  # Wrap single answer in list
                ground_truths_context_ids=ground_truth_context_ids,
                is_answerable=True,  # All WatsonX DocsQA questions are answerable
                additional_information={
                    "source": "watsonx_docs_qa",
                    "dataset": "ibm-research/watsonxDocsQA",
                },
            )
            entries.append(entry)

        logger.info(
            f"Loaded {len(entries)} benchmark entries from WatsonX DocsQA dataset"
        )
        return entries
