"""AIT QA data loader implementation for RAG (Retrieval-Augmented Generation) benchmarking.

This module provides a concrete implementation of the RagDataLoader abstract class
for loading the AIT QA benchmark dataset from HuggingFace and GitHub sources. It loads
queries and ground truth contexts from the HuggingFace AITQARetrieval dataset, ground
truth answers from the AITQA GitHub repository, and PDF documents from a local directory.

The data loader is designed to work with the AIT QA dataset, which contains:
    - Queries: Loaded from HuggingFace dataset 'ibm-research/AITQARetrieval'
    - Ground truth contexts (qrels): Loaded from the same HuggingFace dataset
    - Ground truth answers: Loaded from AITQA GitHub repository JSONL file
    - PDF documents: Loaded from local directory (auto-downloaded if missing)

Typical usage example:
    data_loader = AITQaDataLoader(split=DatasetSplit.TEST)
    benchmark_entries = data_loader.get_benchmark_entries()
    documents = data_loader.get_documents()
"""

import json
import logging
import mimetypes
import random
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

from datasets import load_dataset  # type: ignore[import-not-found]

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragworkbench.datasets_loader.ait_qa_data.config import get_ait_qa_documents_dir
from ragworkbench.datasets_loader.data_models import (
    DataSamplingParams,
    DocumentObject,
    GroundTruthContextId,
    RagBenchmarkEntry,
)

logger = logging.getLogger(__name__)

SEED = 43


class AITQaDataLoader(RagDataLoader):
    """Loads AIT QA benchmark data from HuggingFace, GitHub, and local PDF files.

    This class implements the RagDataLoader abstract class to provide a data loader
    for the AIT QA benchmark dataset. It loads queries and ground truth contexts from
    the HuggingFace AITQARetrieval dataset, ground truth answers from the AITQA GitHub
    repository, and PDF documents from a local directory.

    The loader retrieves data from the following sources:
        - Queries: HuggingFace dataset 'ibm-research/AITQARetrieval' (queries subset)
        - Ground truth contexts (qrels): HuggingFace dataset 'ibm-research/AITQARetrieval' (test split)
        - Ground truth answers: AITQA GitHub repository JSONL file
        - PDF documents: Local directory (auto-downloaded if missing)

    Attributes:
        benchmark_entries: List of RagBenchmarkEntry objects containing questions,
            answers, and ground truth context references.
        documents: List of DocumentObject instances representing the loaded PDF files.

    Example:
        >>> loader = AITQaDataLoader(split=DatasetSplit.TEST)
        >>> entries = loader.get_benchmark_entries()
        >>> docs = loader.get_documents()
        >>> print(f"Loaded {len(entries)} questions and {len(docs)} documents")
    """

    def __init__(
        self,
        split: DatasetSplit | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        """Initialize the AITQaDataLoader with lazy loading configuration.

        This method uses lazy initialization - data is only loaded when first accessed
        through the _get_documents() or _get_benchmark_entries() methods.

        Args:
            split: Optional dataset split to use (DatasetSplit.TRAIN or DatasetSplit.TEST).
                Used for splitting the dataset into train/test sets.
            sampling_params: Parameters for data sampling (e.g., limiting number of
                samples, random seed). Defaults to DataSamplingParams() with no sampling.
            cache_dir: Optional directory path for caching processed data. If None,
                no caching is performed.

        Note:
            Data loading is deferred until _get_documents() or _get_benchmark_entries()
            is called by the parent class. This improves initialization performance and
            allows for better error handling during data loading.
        """
        # Initialize lazy loading flags
        self._benchmark_entries: list[RagBenchmarkEntry] | None = None
        self._documents: list[DocumentObject] | None = None

        # Call parent constructor which will trigger lazy loading via abstract methods
        super().__init__(
            split=split,
            dataset_name="AIT_QA_Dataset",
            sampling_params=sampling_params,
            cache_dir=cache_dir,
        )

    def _load_benchmark_entries(self) -> None:
        """Load and parse benchmark entries from HuggingFace dataset (lazy initialization).

        This method is called once when benchmark entries are first accessed.
        It loads queries and ground truth data from HuggingFace and GitHub.
        """
        # Load the queries from the HuggingFace AITQARetrieval dataset
        logger.info("Loading AIT QA queries from HuggingFace dataset...")
        queries_dataset = load_dataset(
            "ibm-research/AITQARetrieval", "queries", split="test_queries"
        )

        # Load the qrels (query-document relevance) from the default subset
        logger.info(
            "Loading AIT QA qrels (ground truth contexts) from HuggingFace dataset..."
        )
        qrels_dataset = load_dataset("ibm-research/AITQARetrieval", split="test")

        # Build a mapping from question_id to list of document_ids (ground truth contexts)
        # The qrels dataset has multiple rows per question, each with a relevant document
        # Note: Document IDs need to be normalized from format "United-2018_27.md" to "United-2018.pdf"
        qid_to_dids: dict[str, set[str]] = {}
        for item in qrels_dataset:
            qid = item["qid"]
            did = item["did"]

            # Normalize the document ID: remove everything after '_' and change extension to .pdf
            # Example: "United-2018_27.md" -> "United-2018.pdf"
            if "_" in did:
                normalized_did = did.split("_")[0] + ".pdf"
            else:
                # If no underscore, just replace .md with .pdf
                normalized_did = did.replace(".md", ".pdf")

            if qid not in qid_to_dids:
                qid_to_dids[qid] = set()
            # Add to set (automatically handles duplicates after normalization)
            qid_to_dids[qid].add(normalized_did)

        logger.info(f"Loaded ground truth contexts for {len(qid_to_dids)} questions")

        # Load ground truth answers from the AITQA GitHub repository
        logger.info("Loading ground truth answers from AITQA GitHub repository...")
        aitqa_jsonl_url = "https://raw.githubusercontent.com/IBM/AITQA/master/raw_data/aitqa_questions.jsonl"
        qid_to_answers: dict[str, list[str]] = {}

        with urlopen(aitqa_jsonl_url) as response:
            for line in response:
                line_str = line.decode("utf-8").strip()
                if line_str:
                    data = json.loads(line_str)
                    qid = data["id"]
                    qid_to_answers[qid] = data["answers"]

        logger.info(f"Loaded ground truth answers for {len(qid_to_answers)} questions")

        # Create RagBenchmarkEntry objects from the queries dataset
        # The queries_dataset contains _id and text columns which map to question_id and question
        self._benchmark_entries = []

        for item in queries_dataset:
            question_id = item["_id"]
            question = item["text"]

            # Get the ground truth context IDs for this question
            document_ids = qid_to_dids.get(question_id, set())
            ground_truths_context_ids = [
                GroundTruthContextId(document_id=doc_id) for doc_id in document_ids
            ]

            # Get the ground truth answers for this question
            ground_truth_answers = qid_to_answers.get(question_id, None)

            # Create a RagBenchmarkEntry with the loaded data
            entry = RagBenchmarkEntry(
                question_id=question_id,
                question=question,
                ground_truth_answers=ground_truth_answers,
                ground_truths_context_ids=ground_truths_context_ids,
                is_answerable=True,  # Assume all queries are answerable
            )
            self._benchmark_entries.append(entry)

        logger.info(
            f"Loaded {len(self._benchmark_entries)} benchmark entries from HuggingFace"
        )

    def _load_documents(self) -> None:
        """Load document PDFs from the documents directory (lazy initialization).

        This method is called once when documents are first accessed.
        It loads all PDF files from the configured documents directory.
        """
        # Use shared configuration to get the documents directory location
        # This ensures consistency with where create_ait_qa_dataset.py downloads files
        ait_qa_pdf_document_folder = get_ait_qa_documents_dir()

        # Import the URL_TO_OUTPUT_FILE dictionary to get the list of expected files
        from ragworkbench.datasets_loader.ait_qa_data.create_ait_qa_dataset import (
            URL_TO_OUTPUT_FILE,
        )

        # Extract the expected filenames from the URL_TO_OUTPUT_FILE dictionary
        expected_files = list(URL_TO_OUTPUT_FILE.values())

        # Check if the directory exists and contains all required files
        needs_download = False
        if not ait_qa_pdf_document_folder.exists():
            logger.info(
                f"Documents directory does not exist: {ait_qa_pdf_document_folder}"
            )
            needs_download = True
        else:
            # Check if all expected files are present
            missing_files = []
            for filename in expected_files:
                file_path = ait_qa_pdf_document_folder / filename
                if not file_path.exists():
                    missing_files.append(filename)

            if missing_files:
                logger.warning(
                    f"Documents directory exists but is missing {len(missing_files)} file(s): "
                    f"{', '.join(missing_files)}"
                )
                needs_download = True

        # Download the dataset if directory doesn't exist or files are missing
        if needs_download:
            from ragworkbench.datasets_loader.ait_qa_data.create_ait_qa_dataset import (
                create_ait_qa_dataset,
            )

            logger.info("Downloading AIT QA dataset...")
            create_ait_qa_dataset()

        self._documents = []

        # Process all PDF files in the documents directory
        for file in ait_qa_pdf_document_folder.glob("*.pdf"):
            # Attempt to determine the MIME type from the file extension
            mime_type, _ = mimetypes.guess_type(file)
            if mime_type is None:
                # Fallback to application/pdf if MIME type detection fails
                # This ensures the document can still be processed
                mime_type = "application/pdf"
                logger.warning(
                    f"Could not determine MIME type for {file}, "
                    f"defaulting to {mime_type}"
                )

            # Load the file content into a BytesIO buffer
            # This allows the PDF to be treated as a file-like object in memory
            buffer = BytesIO(file.read_bytes())

            # Create a DocumentObject with the file metadata and content stream
            doc = DocumentObject(
                name=file.name,
                stream=buffer,
                mime_type=mime_type,
            )
            self._documents.append(doc)

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        """Retrieve the benchmark entries for the specified split.

        This method implements the abstract _get_benchmark_entries method from
        RagDataLoader. It returns the benchmark entries that were loaded from
        HuggingFace and GitHub sources.

        Args:
            split: The dataset split to retrieve (DatasetSplit.TRAIN or DatasetSplit.TEST).
                If None, returns all entries. Otherwise, splits the data with a 70/30 ratio.

        Returns:
            A list of RagBenchmarkEntry objects containing questions, ground truth
            answers, and references to the source documents.
        """
        # Lazy load benchmark entries if not already loaded
        if self._benchmark_entries is None:
            self._load_benchmark_entries()

        if split is None:
            # Return all the entries
            return self._benchmark_entries  # type: ignore[return-value]
        else:
            # Split the data into train/test sets
            items = self._benchmark_entries[:]  # type: ignore[index]
            random.seed(SEED)
            random.shuffle(items)

            n = len(items)
            train_ratio = 0.7
            split_idx = round(n * train_ratio)

            train_entries = items[:split_idx]  # ~70%
            test_entries = items[split_idx:]  # ~30%

            if split == DatasetSplit.TRAIN:
                return train_entries
            elif split == DatasetSplit.TEST:
                return test_entries
            else:
                raise ValueError(f"Invalid split {split}")

    def _get_documents(self) -> list[DocumentObject]:
        """Retrieve all loaded document objects.

        This method implements the abstract _get_documents method from RagDataLoader.
        It returns the PDF documents that were loaded from the local documents directory.
        If the directory doesn't exist, it will be automatically created and populated.

        Returns:
            A list of DocumentObject instances, each containing:
                - name: The filename of the document
                - stream: A BytesIO buffer with the PDF content
                - mime_type: The MIME type of the document (e.g., 'application/pdf')

        Note:
            The documents are loaded lazily on first access. Each document's
            content is stored as a BytesIO stream for efficient access.
        """
        # Lazy load documents if not already loaded
        if self._documents is None:
            self._load_documents()

        return self._documents  # type: ignore[return-value]


# Made with Bob
