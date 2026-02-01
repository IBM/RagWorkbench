"""
HotpotQA dataset loader implementation.

This module provides a data loader for the HotpotQA multi-hop question answering dataset.
HotpotQA requires reasoning over multiple documents to answer questions.

References:
    - Paper: https://arxiv.org/abs/1809.09600
    - HuggingFace: https://huggingface.co/datasets/hotpotqa/hotpot_qa
"""

import logging
from io import BytesIO
from typing import Literal

from datasets import (  # type: ignore[import-not-found]
    concatenate_datasets,
    load_dataset,
)

from ragbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets_loader.data_models.dataset_names import DatasetName
from ragbench.datasets_loader.data_models.document_object import DocumentObject
from ragbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)

logger = logging.getLogger(__name__)


class HotpotQaDataLoader(RagDataLoader):
    """
    Data loader for the HotpotQA multi-hop question answering dataset.

    This loader handles loading and processing of HotpotQA dataset from HuggingFace,
    including Wikipedia paragraphs and multi-hop questions with supporting facts.

    The HotpotQA dataset includes:
    - Wikipedia paragraphs as context documents
    - Multi-hop questions requiring reasoning across multiple documents
    - Supporting facts indicating which sentences are relevant
    - Question types: bridge (connecting entities) and comparison
    - Difficulty levels: easy, medium, hard

    Example:
        >>> loader = HotpotQaDataLoader(
        ...     dataset_name=DatasetName.HOTPOT_QA,
        ...     split="train",
        ...     sampling_params=DataSamplingParams(question_limit=100)
        ... )
        >>> corpus = loader.get_corpus()
        >>> benchmark = loader.get_benchmark()

    Note:
        This loader uses the HuggingFace dataset "hotpotqa/hotpot_qa" with the
        parquet conversion revision. Documents are created from context paragraphs,
        using the paragraph title as the document ID and concatenating all sentences
        as the document content.
    """

    def __init__(
        self,
        split: Literal["train", "test"] | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
    ):
        """
        Initialize the HotpotQA data loader.

        Args:
            split: Dataset split to load ('train', 'validation', 'test', or None for all).
            sampling_params: Parameters controlling question and document sampling.

        Note:
            The HotpotQA dataset has three splits: train, validation, and test.
            When split is None, all three splits are concatenated.
        """
        logger.info(f"Initializing HotpotQaDataLoader with split='{split}'")

        super().__init__(DatasetName.HOTPOT_QA, split, sampling_params)

    def _get_documents(self) -> list[DocumentObject]:
        """
        Load documents from the HotpotQA dataset.

        This method extracts all unique context paragraphs from the dataset.
        Each question in HotpotQA comes with multiple context paragraphs,
        where each paragraph has a title and a list of sentences.

        The method:
        1. Loads the dataset from HuggingFace
        2. Iterates through all questions in all splits
        3. Extracts unique paragraphs (identified by title)
        4. Concatenates sentences for each paragraph
        5. Creates DocumentObject instances

        Returns:
            List of DocumentObject instances, where each document has:
            - name: The paragraph title (document ID)
            - content: Concatenated sentences as bytes
            - mime_type: Set to 'text/plain' for text documents

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.
        """
        hf_path = "hotpotqa/hotpot_qa"
        revision = "refs/convert/parquet"

        logger.info(
            f"Loading documents from HuggingFace dataset: {hf_path} (revision: {revision})"
        )

        # Load the dataset from HuggingFace
        dataset = load_dataset(hf_path, revision=revision)

        # Dictionary to store unique documents by title
        unique_documents: dict[str, str] = {}

        # Process all splits to extract unique documents
        for split_name in ["train", "validation"]:
            if split_name not in dataset:
                logger.warning(f"Split '{split_name}' not found in dataset")
                continue

            split_data = dataset[split_name]
            logger.info(
                f"Processing {len(split_data)} entries from '{split_name}' split"
            )

            # Extract documents from each question's context
            for row in split_data:
                context = row["context"]
                # context is a dict with 'title' and 'sentences' keys
                # Each entry: {'title': [...], 'sentences': [[...], [...]]}
                titles = context["title"]
                sentences_lists = context["sentences"]

                # Process each paragraph in the context
                for title, sentences in zip(titles, sentences_lists, strict=True):
                    if title not in unique_documents:
                        # Concatenate all sentences with space separator
                        document_text = " ".join(sentences)
                        unique_documents[title] = document_text

        logger.info(f"Found {len(unique_documents)} unique documents")

        # Create DocumentObject for each unique document
        documents: list[DocumentObject] = []
        for title, text in unique_documents.items():
            doc = DocumentObject(
                name=title,
                stream=BytesIO(text.encode("utf-8")),
                mime_type="text/plain",
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from HotpotQA corpus")
        return documents

    def _get_benchmark_entries(
        self, split: Literal["train", "test"] | None
    ) -> list[RagBenchmarkEntry]:
        """
        Load question-answer pairs from the HotpotQA dataset.

        This method loads HotpotQA questions with their ground truth answers
        and context documents from the HuggingFace dataset.

        Args:
            split: Dataset split to load ('train', 'validation', 'test', or None for all).

        Returns:
            List of RagBenchmarkEntry instances with questions and answers.

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.

        Note:
            All context titles associated with each question are used as
            ground_truth_context_ids. This includes all documents provided
            in the context, not just the supporting facts.
        """
        hf_path = "hotpotqa/hotpot_qa"
        revision = "refs/convert/parquet"

        logger.info(
            f"Loading benchmark entries from HuggingFace dataset: {hf_path}, "
            f"split='{split}'"
        )

        # Load the dataset from HuggingFace
        dataset = load_dataset(hf_path, revision=revision)

        # Handle split parameter
        if split == "train":
            data = dataset["train"]
            logger.info(f"Loading train split with {len(data)} entries")
        elif split == "test":
            data = dataset["validation"]
            logger.info(f"Loading test split with {len(data)} entries")
        else:
            # Concatenate all splits
            train_data = dataset["train"]
            val_data = dataset["validation"]
            data = concatenate_datasets([train_data, val_data])
            logger.info(
                f"Loading all splits: {len(train_data)} train + {len(val_data)} test "
                f" = {len(data)} total entries"
            )

        # Create RagBenchmarkEntry for each row
        entries: list[RagBenchmarkEntry] = []
        for row in data:
            # Extract fields from the row
            question_id = str(row["id"])
            question = row["question"]
            answer = row["answer"]
            question_type = row.get("type", "unknown")
            level = row.get("level", "unknown")

            # Extract all context titles as ground truth
            # context is a dict with 'title' and 'sentences' keys
            # Format: {'title': [...], 'sentences': [[...], [...]]}
            context = row["context"]
            context_titles = context["title"]

            # Get unique titles from context (remove duplicates if any)
            unique_context_titles = list(set(context_titles))

            # Convert context titles to GroundTruthContextId objects
            ground_truth_context_ids = [
                GroundTruthContextId(document_id=title)
                for title in unique_context_titles
            ]

            # Create RagBenchmarkEntry
            entry = RagBenchmarkEntry(
                question_id=question_id,
                question=question,
                ground_truth_answers=[answer],  # Wrap single answer in list
                ground_truth_context_ids=ground_truth_context_ids,
                is_answerable=True,  # All HotpotQA questions are answerable
                additional_information={
                    "source": "hotpotqa",
                    "question_type": question_type,
                    "level": level,
                },
            )
            entries.append(entry)

        logger.info(f"Loaded {len(entries)} benchmark entries from HotpotQA dataset")
        return entries
