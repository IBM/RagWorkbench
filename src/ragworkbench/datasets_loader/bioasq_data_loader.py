"""
BioASQ dataset loader implementation.

References:
    - https://huggingface.co/datasets/enelpol/rag-mini-bioasq
"""

import logging
from io import BytesIO
from pathlib import Path

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


class BioasqDataLoader(RagDataLoader):
    """
    Data loader for the BioASQ biomedical question answering dataset.

    This loader handles loading and processing of BioASQ dataset files,
    including biomedical documents and question-answer pairs with ground
    truth document references.

    The BioASQ dataset typically includes:
    - Biomedical research articles (PubMed abstracts)
    - Questions about biomedical topics
    - Ground truth answers (exact, yes/no, factoid, list)
    - References to relevant documents

    Example:
        >>> loader = BioasqDataLoader(
        ...     dataset_name=DatasetName.BIOASQ,
        ...     split="train",
        ...     sampling_params=DataSamplingParams(question_limit=100)
        ... )
        >>> corpus = loader.get_corpus()
        >>> benchmark = loader.get_benchmark()

    Note:
        This is a placeholder implementation. To use this loader with actual
        BioASQ data, you need to:
        1. Download the BioASQ dataset from http://bioasq.org/
        2. Implement the data loading logic in _get_documents() and
           _get_benchmark_entries()
        3. Handle the specific BioASQ JSON format
        4. Process PubMed article IDs and abstracts
    """

    def __init__(
        self,
        split: DatasetSplit | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        """
        Initialize the BioASQ data loader.

        Args:
            split: Dataset split to load ('train', 'test', or None for all).
            sampling_params: Parameters controlling question and document sampling.
        """
        logger.info(f"Initializing BioasqDataLoader with split='{split}'")

        super().__init__(
            DatasetName.BIOASQ, split, sampling_params, cache_dir=cache_dir
        )

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

        # The dataset should have a split (typically 'test')
        # Get the first available split
        split_name = "test"
        corpus_data = dataset[split_name]

        logger.info(f"Found {len(corpus_data)} documents in split '{split_name}'")

        # Create DocumentObject for each row
        documents: list[DocumentObject] = []
        for row in corpus_data:
            # Extract id and passage from the row
            # Convert id to string in case it's an integer
            doc_id = str(row["id"])
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
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        """
        Load question-answer pairs from the BioASQ dataset.

        This method loads BioASQ questions with their ground truth answers
        and document references from the HuggingFace dataset
        "enelpol/rag-mini-bioasq" subset "question-answer-passages".

        Args:
            split: Dataset split to load ('train', 'test', or None for all).

        Returns:
            List of RagBenchmarkEntry instances with questions and answers.

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.
        """
        hf_path = "enelpol/rag-mini-bioasq"
        subset = "question-answer-passages"

        logger.info(
            f"Loading benchmark entries from HuggingFace dataset: {hf_path}/{subset}, "
            f"split='{split}'"
        )

        # Load the question-answer-passages dataset from HuggingFace
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
            question_id = str(row["id"])
            question = row["question"]
            answer = row["answer"]
            relevant_passage_ids = row["relevant_passage_ids"]

            # Convert relevant_passage_ids to GroundTruthContextId objects
            ground_truth_context_ids = [
                GroundTruthContextId(document_id=str(doc_id))
                for doc_id in relevant_passage_ids
            ]

            # Create RagBenchmarkEntry
            entry = RagBenchmarkEntry(
                question_id=question_id,
                question=question,
                ground_truth_answers=[answer],  # Wrap single answer in list
                ground_truths_context_ids=ground_truth_context_ids,
                is_answerable=True,  # All BioASQ questions are answerable
                additional_information={
                    "source": "bioasq",
                    "question_type": "factoid",  # Default type for BioASQ
                },
            )
            entries.append(entry)

        logger.info(f"Loaded {len(entries)} benchmark entries from BioASQ dataset")
        return entries
