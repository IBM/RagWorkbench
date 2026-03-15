"""
Mini Wikipedia dataset loader implementation.

This module provides a data loader for the Mini Wikipedia RAG dataset.

References:
    - HuggingFace: https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia
"""

import logging
from io import BytesIO
from pathlib import Path

from datasets import load_dataset  # type: ignore[import-not-found]

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)
from ragworkbench.datasets_loader.data_models.document_object import DocumentObject
from ragworkbench.datasets_loader.data_models.rag_benchmark import RagBenchmarkEntry
from ragworkbench.datasets_loader.dataset_names import DatasetName
from ragworkbench.datasets_loader.datasets_utils import get_benchmark_split

logger = logging.getLogger(__name__)


class MiniWikiDataLoader(RagDataLoader):
    """
    Data loader for the Mini Wikipedia RAG dataset.

    This loader handles loading and processing of the Mini Wikipedia dataset
    from HuggingFace, including documents/passages for RAG evaluation.

    The Mini Wikipedia dataset includes:
    - Wikipedia passages/articles
    - Document IDs for retrieval

    Example:
        >>> loader = MiniWikiDataLoader(
        ...     split="train",
        ...     sampling_params=DataSamplingParams(question_limit=100)
        ... )
        >>> corpus = loader.get_corpus()
        >>> benchmark = loader.get_benchmark()

    Note:
        This loader reads documents from HuggingFace dataset "rag-datasets/rag-mini-wikipedia".
        The text-corpus configuration contains the document corpus.
    """

    def __init__(
        self,
        split: DatasetSplit | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        """
        Initialize the Mini Wikipedia data loader.

        Args:
            split: Dataset split to load ('train', 'test', or None for all).
            sampling_params: Parameters controlling question and document sampling.

        Note:
            Documents are loaded from HuggingFace dataset "rag-datasets/rag-mini-wikipedia"
            using the "text-corpus" configuration.
        """
        logger.info(f"Initializing MiniWikiDataLoader with split='{split}'")

        super().__init__(
            DatasetName.MINI_WIKI, split, sampling_params, cache_dir=cache_dir
        )

    def _get_documents(self) -> list[DocumentObject]:
        """
        Load documents from the Mini Wikipedia dataset.

        This method loads documents from the rag-datasets/rag-mini-wikipedia
        HuggingFace dataset using the text-corpus configuration.

        The method:
        1. Loads the dataset from HuggingFace with name="text-corpus"
        2. Extracts documents from the 'train' split
        3. Creates DocumentObject instances

        Returns:
            List of DocumentObject instances, where each document has:
            - name: The passage ID (from 'id' column)
            - content: The passage text (from 'passage' column) as bytes
            - mime_type: Set to 'text/plain' for text documents

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.

        Note:
            The dataset structure has two columns:
            - id: passage identifier (used as document name)
            - passage: passage content
        """
        hf_path = "rag-datasets/rag-mini-wikipedia"
        config_name = "text-corpus"

        logger.info(
            f"Loading documents from HuggingFace dataset: {hf_path}, config: {config_name}"
        )

        # Load the dataset from HuggingFace with the text-corpus configuration
        dataset = load_dataset(hf_path, name=config_name)

        passages_data = dataset["passages"]
        logger.info(f"Processing {len(passages_data)} passages from dataset")

        # Create DocumentObject for each passage
        documents: list[DocumentObject] = []
        for row in passages_data:
            # Extract fields from the row
            doc_id = str(row["id"])
            passage_text = row["passage"]

            # Create DocumentObject
            doc = DocumentObject(
                name=doc_id,
                stream=BytesIO(passage_text.encode("utf-8")),
                mime_type="text/plain",
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from Mini Wikipedia corpus")
        return documents

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        """
        Load question-answer pairs from the Mini Wikipedia dataset.

        This method loads question-answer pairs from the rag-datasets/rag-mini-wikipedia
        HuggingFace dataset using the question-answer configuration.

        Args:
            split: Dataset split to load ('train', 'test', or None for all).
                   Split is handled by get_benchmark_split utility function.

        Returns:
            List of RagBenchmarkEntry instances with questions and answers.

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.

        Note:
            The dataset structure has three columns:
            - id: question identifier
            - question: the question text
            - answer: the answer text

            This dataset does not include ground truth context IDs, so the
            ground_truth_context_ids list will be empty for all entries.

            The split parameter is handled by get_benchmark_split() which creates
            reproducible train/test splits from all available data.
        """
        hf_path = "rag-datasets/rag-mini-wikipedia"
        config_name = "question-answer"

        logger.info(
            f"Loading benchmark entries from HuggingFace dataset: {hf_path}, "
            f"config: {config_name}"
        )

        # Load the dataset from HuggingFace with the question-answer configuration
        dataset = load_dataset(hf_path, name=config_name)

        # Load all available data from the dataset
        # We'll use get_benchmark_split to handle train/test splitting
        available_splits = list(dataset.keys())
        logger.info(f"Available splits in dataset: {available_splits}")

        if not available_splits:
            logger.warning("No splits found in dataset, returning empty list")
            return []

        # Concatenate all available splits to get all data
        all_data = []
        for split_name in available_splits:
            split_data = dataset[split_name]
            all_data.extend(split_data)
            logger.info(f"Loaded {len(split_data)} entries from '{split_name}' split")

        # Create RagBenchmarkEntry for each row
        all_entries: list[RagBenchmarkEntry] = []

        for row in all_data:
            # Extract fields from the row
            question_id = str(row["id"])
            question = row["question"]
            answer = row["answer"]

            # Create ground truth answers list
            # Assuming answer is a single string
            ground_truth_answers = [answer] if isinstance(answer, str) else answer

            # Create RagBenchmarkEntry
            # Note: No ground truth context IDs available in this dataset
            entry = RagBenchmarkEntry(
                question_id=question_id,
                question=question,
                ground_truth_answers=ground_truth_answers,
                ground_truths_context_ids=[],  # No context IDs in this dataset
                is_answerable=True,  # All questions are answerable
            )
            all_entries.append(entry)

        logger.info(f"Loaded {len(all_entries)} total benchmark entries")

        # Use get_benchmark_split to handle train/test splitting
        entries = get_benchmark_split(all_entries, split)
        logger.info(f"After split filtering (split='{split}'): {len(entries)} entries")

        return entries
