"""
RealMM dataset loader implementation.

This module provides a data loader for the RealMM multimodal RAG datasets.

References:
    - HuggingFace: https://huggingface.co/datasets/ibm-research/REAL-MM-RAG_FinSlides
    - HuggingFace: https://huggingface.co/datasets/ibm-research/REAL-MM-RAG_FinReport
    - HuggingFace: https://huggingface.co/datasets/ibm-research/REAL-MM-RAG_TechReport
    - HuggingFace: https://huggingface.co/datasets/ibm-research/REAL-MM-RAG_TechSlides
"""

import logging
import mimetypes
from io import BytesIO
from pathlib import Path

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
from ragworkbench.datasets_loader.datasets_utils import get_benchmark_split

logger = logging.getLogger(__name__)

# Mapping from DatasetName to HuggingFace dataset path
DATASET_NAME_TO_HF_PATH = {
    DatasetName.REAL_MM_FIN_SLIDES: "ibm-research/REAL-MM-RAG_FinSlides",
    DatasetName.REAL_MM_FIN_REPORT: "ibm-research/REAL-MM-RAG_FinReport",
    DatasetName.REAL_MM_TECH_REPORT: "ibm-research/REAL-MM-RAG_TechReport",
    DatasetName.REAL_MM_TECH_SLIDES: "ibm-research/REAL-MM-RAG_TechSlides",
}


class RealMMRagDataLoader(RagDataLoader):
    """
    Data loader for the RealMM multimodal RAG datasets.

    This loader handles loading and processing of the RealMM datasets from
    HuggingFace, including multimodal documents and question-answer pairs
    with ground truth document references.

    Supported datasets:
    - REAL-MM-RAG_FinSlides: Financial slide documents
    - REAL-MM-RAG_FinReport: Financial report documents
    - REAL-MM-RAG_TechReport: Technical report documents
    - REAL-MM-RAG_TechSlides: Technical slide documents

    The RealMM datasets include:
    - Multimodal documents (text, images, tables, charts)
    - Questions requiring multimodal understanding
    - Ground truth answers
    - References to relevant documents and specific content locations

    Example:
        >>> loader = RealMMRagDataLoader(
        ...     dataset_name=DatasetName.REAL_MM_FIN_SLIDES,
        ...     split="train",
        ...     sampling_params=DataSamplingParams(question_limit=100)
        ... )
        >>> corpus = loader.get_corpus()
        >>> benchmark = loader.get_benchmark()
    """

    def __init__(
        self,
        dataset_name: DatasetName,
        split: DatasetSplit | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        """
        Initialize the RealMM data loader.

        Args:
            dataset_name: Which RealMM dataset to load.
                         Must be one of: REAL_MM_FIN_SLIDES, REAL_MM_FIN_REPORT,
                         REAL_MM_TECH_REPORT, REAL_MM_TECH_SLIDES.
            split: Dataset split to load ('train', 'test', or None for all).
            sampling_params: Parameters controlling question and document sampling.
                           Defaults to no sampling (all data included).

        Raises:
            ValueError: If dataset_name is not a supported RealMM dataset.
            Exception: If the dataset cannot be loaded from HuggingFace.

        Note:
            The initialization process:
            1. Validates dataset_name and determines HuggingFace path
            2. Loads all documents via _get_documents()
            3. Loads all benchmark entries via _get_benchmark_entries()
            4. Applies sampling based on sampling_params
            5. Creates RagBenchmark and RagCorpus instances
            6. Logs the final dataset size
        """
        # Validate dataset_name and get HuggingFace path
        if dataset_name not in DATASET_NAME_TO_HF_PATH:
            raise ValueError(
                f"Unsupported dataset_name: {dataset_name}. "
                f"Must be one of: {list(DATASET_NAME_TO_HF_PATH.keys())}"
            )

        self.hf_dataset_path = DATASET_NAME_TO_HF_PATH[dataset_name]

        logger.info(
            f"Initializing RealMMRagDataLoader with dataset='{dataset_name}', "
            f"split='{split}' from HuggingFace dataset: {self.hf_dataset_path}"
        )

        super().__init__(dataset_name, split, sampling_params, cache_dir=cache_dir)

    def _get_documents(self) -> list[DocumentObject]:
        """
        Load documents from the RealMM dataset corpus.

        This method loads the document corpus from the HuggingFace dataset
        ibm-research/REAL-MM-RAG_FinSlides. Documents include multimodal
        financial slide content such as images.

        The dataset contains the following columns:
        - image_filename: The document ID (filename)
        - image: The actual image data (PIL Image object)

        Returns:
            List of DocumentObject instances, where each document has:
            - name: The image filename (document ID)
            - stream: The image content as a BytesIO stream
            - mime_type: The MIME type determined from the filename extension

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.
        """
        logger.info(
            f"Loading documents from HuggingFace dataset: {self.hf_dataset_path}"
        )

        # Load the dataset from HuggingFace (only 'test' split is available)
        dataset = load_dataset(self.hf_dataset_path)
        data = dataset["test"]
        logger.info(f"Loaded {len(data)} entries from test split")

        # Track unique documents to avoid duplicates
        seen_filenames = set()
        documents: list[DocumentObject] = []

        for row in data:
            image_filename = row["image_filename"]

            # Skip if we've already processed this document
            if image_filename in seen_filenames:
                continue
            seen_filenames.add(image_filename)

            # Get the image data (PIL Image object from HuggingFace datasets)
            image = row["image"]

            # Determine MIME type from filename extension
            mime_type, _ = mimetypes.guess_type(image_filename)
            if mime_type is None:
                # Default to PNG if we can't determine the type
                mime_type = "image/png"
                logger.warning(
                    f"Could not determine MIME type for {image_filename}, "
                    f"defaulting to {mime_type}"
                )

            # Convert PIL Image to bytes
            image_bytes = BytesIO()
            # Determine format from filename extension
            image_format = image_filename.split(".")[-1].upper()
            if image_format == "JPG":
                image_format = "JPEG"

            # Save image to BytesIO buffer
            image.save(image_bytes, format=image_format)
            image_bytes.seek(0)  # Reset to beginning of stream

            # Create DocumentObject
            doc = DocumentObject(
                name=image_filename,
                stream=image_bytes,
                mime_type=mime_type,
            )
            documents.append(doc)

        logger.info(
            f"Loaded {len(documents)} unique documents from RealMM corpus "
            f"(from {len(data)} total entries)"
        )
        return documents

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        """
        Load question-answer pairs from the RealMM dataset.

        This method loads question-answer pairs with ground truth answers
        and document references from the HuggingFace dataset
        ibm-research/REAL-MM-RAG_FinSlides.

        The dataset contains the following columns:
        - id: The query/question ID
        - image_filename: The ground truth context document ID (image file)
        - answer: The ground truth answer string

        Note: The HuggingFace dataset only contains a 'test' split. The split
        parameter is handled by using get_benchmark_split() to create train/test
        splits from the available data.

        Args:
            split: Dataset split to load ('train', 'test', or None for all).

        Returns:
            List of RagBenchmarkEntry instances with questions and answers.

        Raises:
            Exception: If the dataset cannot be loaded from HuggingFace.
        """
        logger.info(
            f"Loading benchmark entries from HuggingFace dataset: {self.hf_dataset_path}, "
            f"split='{split}'"
        )

        # Load the dataset from HuggingFace (only 'test' split is available)
        dataset = load_dataset(self.hf_dataset_path)
        data = dataset["test"]
        logger.info(f"Loaded {len(data)} entries from test split")

        # Filter to only include rows where query is not None
        data = data.filter(lambda row: row["query"] is not None)
        logger.info(f"Filtered to {len(data)} entries with non-null queries")

        # Create RagBenchmarkEntry for each row
        entries: list[RagBenchmarkEntry] = []
        for row in data:
            # Extract fields from the row
            question_id = str(row["id"])
            question = row[
                "query"
            ]  # query is guaranteed to be not None after filtering
            answer = row["answer"]
            image_filename = row["image_filename"]

            # Create ground truth context ID from the image filename
            ground_truth_context_ids = [
                GroundTruthContextId(document_id=str(image_filename))
            ]

            # Create RagBenchmarkEntry
            entry = RagBenchmarkEntry(
                question_id=question_id,
                question=question,
                ground_truth_answers=[answer],  # Wrap single answer in list
                ground_truths_context_ids=ground_truth_context_ids,
                is_answerable=True,  # All RealMM questions are answerable
            )
            entries.append(entry)

        logger.info(f"Loaded {len(entries)} benchmark entries from RealMM dataset")

        # Apply split using get_benchmark_split utility
        return get_benchmark_split(entries, split)
