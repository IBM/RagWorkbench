"""
CLAP-NQ dataset loader implementation.

This module provides a data loader for the CLAP-NQ (Contrastive Language-Audio Pre-training
for Natural Questions) dataset from PrimeQA.

References:
    - GitHub: https://github.com/primeqa/clapnq
    - HuggingFace: https://huggingface.co/datasets/PrimeQA/clapnq
"""

import csv
import logging
from io import BytesIO, StringIO
from pathlib import Path

import requests
from datasets import load_dataset  # type: ignore[import-not-found]

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


class ClapNqDataLoader(RagDataLoader):
    """
    Data loader for the CLAP-NQ dataset.

    This loader handles loading and processing of CLAP-NQ dataset from GitHub TSV files,
    including documents and question-answer pairs with ground truth context.

    The CLAP-NQ dataset includes:
    - Documents/passages from Natural Questions
    - Questions with ground truth answers
    - Ground truth context references

    Example:
        >>> loader = ClapNqDataLoader(
        ...     dataset_name=DatasetName.CLAP_NQ,
        ...     split="train",
        ...     sampling_params=DataSamplingParams(question_limit=100)
        ... )
        >>> corpus = loader.get_corpus()
        >>> benchmark = loader.get_benchmark()

    Note:
        This loader reads benchmark data from GitHub raw TSV files.
        The 'train' split loads from question_train_answerable.tsv
        The 'test' split loads from question_dev_answerable.tsv
        Documents are loaded from HuggingFace dataset "PrimeQA/clapnq_passages".
    """

    def __init__(
        self,
        split: DatasetSplit | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        """
        Initialize the CLAP-NQ data loader.

        Args:
            split: Dataset split to load ('train', 'test', or None for all).
                   'train' loads from question_train_answerable.tsv
                   'test' loads from question_dev_answerable.tsv
            sampling_params: Parameters controlling question and document sampling.

        Note:
            Benchmark data is loaded from GitHub TSV files.
            Documents are loaded from HuggingFace dataset "PrimeQA/clapnq_passages".
            When split is None, both train and test splits are concatenated.
        """
        logger.info(f"Initializing ClapNqDataLoader with split='{split}'")

        super().__init__(
            DatasetName.CLAP_NQ, split, sampling_params, cache_dir=cache_dir
        )

    def _get_documents(self) -> list[DocumentObject]:
        """
        Load documents from the CLAP-NQ passages dataset.

        This method loads documents from the PrimeQA/clapnq_passages HuggingFace dataset.

        The method:
        1. Loads the clapnq_passages dataset from HuggingFace
        2. Extracts documents from the 'train' split
        3. Creates DocumentObject instances with metadata

        Returns:
            List of DocumentObject instances, where each document has:
            - name: The passage ID (from 'id' column)
            - content: The passage text (from 'text' column) as bytes
            - mime_type: Set to 'text/plain' for text documents
            - metadata: Contains 'title' field from the dataset

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.

        Note:
            The dataset structure has three columns:
            - id: passage identifier (used as document name)
            - text: passage content
            - title: passage title (stored in metadata)
        """
        hf_path = "PrimeQA/clapnq_passages"

        logger.info(f"Loading documents from HuggingFace dataset: {hf_path}")

        # Load the dataset from HuggingFace
        dataset = load_dataset(hf_path)

        # Get the train split (which contains all passages)
        if "train" not in dataset:
            raise Exception(f"'train' split not found in dataset {hf_path}")

        passages_data = dataset["train"]
        logger.info(f"Processing {len(passages_data)} passages from dataset")

        # Create DocumentObject for each passage
        documents: list[DocumentObject] = []
        for row in passages_data:
            # Extract fields from the row
            doc_id = str(row["id"])
            text = row["text"]
            title = row.get("title", "")

            # Create DocumentObject with metadata
            doc = DocumentObject(
                name=doc_id,
                stream=BytesIO(text.encode("utf-8")),
                mime_type="text/plain",
                metadata={"title": title},
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from CLAP-NQ passages corpus")
        return documents

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        """
        Load question-answer pairs from the CLAP-NQ dataset.

        This method loads CLAP-NQ questions with their ground truth answers
        and context documents from GitHub raw TSV files.

        Args:
            split: Dataset split to load ('train', 'test', or None for all).
                   'train' loads from question_train_answerable.tsv
                   'test' loads from question_dev_answerable.tsv

        Returns:
            List of RagBenchmarkEntry instances with questions and answers.

        Raises:
            Exception: If the TSV files cannot be loaded from GitHub.

        Note:
            The split mapping is:
            - 'train' -> question_train_answerable.tsv
            - 'test' -> question_dev_answerable.tsv
            - None -> Both files concatenated

            TSV file structure (tab-separated):
            - id: question ID (string)
            - question: the question text (string)
            - doc-id-list: comma-separated list of document IDs (string)
            - answers: pipe-separated list of answer strings (string)
        """
        # GitHub raw URLs for the TSV files
        train_url = "https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/train/question_train_answerable.tsv"
        test_url = "https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/dev/question_dev_answerable.tsv"

        logger.info(f"Loading benchmark entries from GitHub TSV files, split='{split}'")

        entries: list[RagBenchmarkEntry] = []

        # Determine which URLs to load based on split
        urls_to_load = []
        if split == "train":
            urls_to_load = [("train", train_url)]
        elif split == "test":
            urls_to_load = [("test", test_url)]
        else:
            urls_to_load = [("train", train_url), ("test", test_url)]

        # Load and parse each TSV file
        for split_name, url in urls_to_load:
            logger.info(f"Loading {split_name} split from {url}")

            try:
                # Download the TSV file
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                # Parse TSV content
                tsv_content = response.text
                tsv_reader = csv.DictReader(StringIO(tsv_content), delimiter="\t")

                split_entries = 0
                for row in tsv_reader:
                    # Extract fields from the TSV row
                    question_id = str(row["id"])
                    question = row["question"]

                    # Parse doc-id-list (comma-separated document IDs)
                    doc_id_list_str = row.get("doc-id-list", "")
                    doc_ids = [
                        doc_id.strip()
                        for doc_id in doc_id_list_str.split(",")
                        if doc_id.strip()
                    ]

                    # Create ground truth context IDs from document IDs
                    ground_truth_context_ids = [
                        GroundTruthContextId(document_id=doc_id) for doc_id in doc_ids
                    ]

                    # Parse answers (pipe-separated answer strings)
                    answers_str = row.get("answers", "")
                    ground_truth_answers = [answers_str]

                    # Create RagBenchmarkEntry
                    entry = RagBenchmarkEntry(
                        question_id=question_id,
                        question=question,
                        ground_truth_answers=ground_truth_answers,
                        ground_truths_context_ids=ground_truth_context_ids,
                        is_answerable=True,  # All CLAP-NQ questions are answerable
                    )
                    entries.append(entry)
                    split_entries += 1

                logger.info(f"Loaded {split_entries} entries from {split_name} split")

            except requests.RequestException as e:
                logger.error(f"Failed to load {split_name} split from {url}: {e}")
                raise Exception(
                    f"Failed to download CLAP-NQ {split_name} data from GitHub: {e}"
                ) from e
            except (KeyError, csv.Error) as e:
                logger.error(f"Failed to parse TSV data from {split_name} split: {e}")
                raise Exception(
                    f"Failed to parse CLAP-NQ {split_name} TSV data: {e}"
                ) from e

        logger.info(
            f"Loaded {len(entries)} total benchmark entries from CLAP-NQ dataset"
        )
        return entries
