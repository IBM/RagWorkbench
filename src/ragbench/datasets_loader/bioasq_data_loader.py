"""
BioASQ dataset loader implementation.

References:
    - https://huggingface.co/datasets/enelpol/rag-mini-bioasq
"""

import logging
from io import BytesIO
from typing import Literal

from datasets import load_dataset

from ragbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets_loader.data_models.dataset_names import DatasetName
from ragbench.datasets_loader.data_models.document_object import DocumentObject
from ragbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)

logger = logging.getLogger(__name__)


class BioasqDataLoader(RagDataLoader):
    def __init__(
        self,
        split: Literal["train", "test"] | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
    ):
        """
        Initialize the BioASQ data loader.

        """
        logger.info(f"Initializing BioasqDataLoader with split='{split}'")

        super().__init__(DatasetName.BIOASQ, split, sampling_params)

    def _get_documents(self) -> list[DocumentObject]:
        """
        Load documents from the BioASQ text corpus.

        This method loads the text corpus from the HuggingFace dataset
        "enelpol/rag-mini-bioasq" subset "text-corpus". Each row contains
        a passage (document content) and an id (document identifier).

        Returns:
            List of DocumentObject instances, where each document has:
            - name: The document ID from the corpus
            - content: The passage text as bytes
            - mime_type: Set to 'text/plain' for text documents

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.
        """
        hf_path = "enelpol/rag-mini-bioasq"
        subset = "text-corpus"

        logger.info(f"Loading documents from HuggingFace dataset: {hf_path}/{subset}")

        # Load the text corpus from HuggingFace
        dataset = load_dataset(hf_path, subset)

        # The dataset should have a split (typically 'train')
        # Get the first available split
        split_name = "test"
        corpus_data = dataset[split_name]

        logger.info(f"Found {len(corpus_data)} documents in split '{split_name}'")

        # Create DocumentObject for each row
        documents: list[DocumentObject] = []
        for row in corpus_data:
            # Extract id and passage from the row
            doc_id = str(row["id"])  # Convert to string in case it's an integer
            passage = row["passage"]

            # Create DocumentObject with the passage content
            # Convert string content to bytes for DocumentStream
            doc = DocumentObject(
                name=doc_id,
                stream=BytesIO(passage.encode("utf-8")),
                mime_type="text/plain",
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from BioASQ corpus")
        return documents

    def _get_benchmark_entries(
        self, split: Literal["train", "test"] | None
    ) -> list[RagBenchmarkEntry]:
        """
        Load benchmark entries from the BioASQ question-answer-passage dataset.

        This method loads question-answer pairs from the HuggingFace dataset
        "enelpol/rag-mini-bioasq" subset "question-answer-passage". Each row contains
        a question, answer, id, and relevant_passage_ids (list of document IDs).

        Args:
            split: Dataset split to load ('train', 'test', or None for all splits).

        Returns:
            List of RagBenchmarkEntry instances, where each entry has:
            - question_id: The question ID from the dataset
            - question: The question text
            - ground_truth_answers: List containing the single answer string
            - ground_truth_context_ids: List of GroundTruthContextId objects
            - is_answerable: Always True for this dataset
            - additional_information: None

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.
        """
        hf_path = "enelpol/rag-mini-bioasq"
        subset = "question-answer-passages"

        logger.info(
            f"Loading benchmark entries from HuggingFace dataset: {hf_path}/{subset}"
        )

        # Load the question-answer-passage dataset from HuggingFace
        dataset = load_dataset(hf_path, subset)

        # Determine which splits to process
        entries: list[RagBenchmarkEntry] = []

        if split is None:
            # Load both train and test splits
            splits_to_process = ["train", "test"]
        else:
            # Load only the requested split
            splits_to_process = [split]

        # Process each split
        for split_name in splits_to_process:
            if split_name not in dataset:
                logger.warning(f"Split '{split_name}' not found in dataset, skipping")
                continue

            split_data = dataset[split_name]
            logger.info(
                f"Processing {len(split_data)} entries from split '{split_name}'"
            )

            # Create RagBenchmarkEntry for each row
            for row in split_data:
                # Extract fields from the row
                question_id = str(
                    row["id"]
                )  # Convert to string in case it's an integer
                question = row["question"]
                answer = row["answer"]
                relevant_passage_ids = row["relevant_passage_ids"]

                # Convert single answer string to list format
                ground_truth_answers = [answer]

                # Create GroundTruthContextId objects from relevant_passage_ids
                # Convert each doc_id to string in case they're integers
                ground_truth_context_ids = [
                    GroundTruthContextId(document_id=str(doc_id))
                    for doc_id in relevant_passage_ids
                ]

                # Create RagBenchmarkEntry
                entry = RagBenchmarkEntry(
                    question_id=question_id,
                    question=question,
                    ground_truth_answers=ground_truth_answers,
                    ground_truth_context_ids=ground_truth_context_ids,
                    is_answerable=True,
                    additional_information=None,
                )
                entries.append(entry)

        logger.info(f"Loaded {len(entries)} benchmark entries from BioASQ dataset")
        return entries
