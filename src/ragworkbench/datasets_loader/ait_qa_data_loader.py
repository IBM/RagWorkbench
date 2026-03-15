# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AIT QA data loader implementation for RAG (Retrieval-Augmented Generation) benchmarking.

This module provides a concrete implementation of the RagDataLoader abstract class
for loading the AIT QA benchmark dataset from local files. It reads question-answer
pairs from a CSV file and loads corresponding PDF documents from a local directory.

The data loader is designed to work with the AIT QA dataset, which contains:
    - benchmark.csv: Contains questions, answers, and references to ground truth documents
    - documents/: Directory containing PDF files referenced in the benchmark

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
    """Loads AIT QA benchmark data from local CSV and PDF files.

    This class implements the RagDataLoader abstract class to provide a data loader
    for the AIT QA benchmark dataset. It loads question-answer pairs from a CSV file
    and their corresponding PDF documents from a local directory.

    The loader expects the following structure:
        - A CSV file (benchmark.csv) with columns: question_id, question,
          correct_answer, correct_answer_document_ids, is_answerable, golden_contexts
        - A documents directory containing PDF files referenced in the CSV

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
        """Initialize the AITQaDataLoader with benchmark data and documents.

        This method performs the following initialization steps:
        1. Locates the ait_qa_pdf directory using shared configuration
        2. Loads benchmark.csv and parses it into RagBenchmarkEntry objects
        3. Loads all PDF files from the documents directory into DocumentObject instances
        4. Calls the parent RagDataLoader constructor with the dataset configuration

        Args:
            split: Optional dataset split to use (DatasetSplit.TRAIN or DatasetSplit.TEST).
                Used for splitting the dataset into train/test sets.
            sampling_params: Parameters for data sampling (e.g., limiting number of
                samples, random seed). Defaults to DataSamplingParams() with no sampling.
            cache_dir: Optional directory path for caching processed data. If None,
                no caching is performed.

        Raises:
            FileNotFoundError: If benchmark.csv or the documents directory cannot be found.
            pd.errors.EmptyDataError: If benchmark.csv is empty.
            KeyError: If required columns are missing from benchmark.csv.

        Note:
            All PDF files in the documents directory are loaded into memory as BytesIO
            streams. For large datasets, consider the memory implications.

            The documents directory location is determined by the shared configuration
            module (ait_qa_data.config), ensuring consistency with the dataset downloader.
        """
        # Step 1: Load and parse benchmark entries from HuggingFace dataset
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
        self.benchmark_entries: list[RagBenchmarkEntry] = []

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
            self.benchmark_entries.append(entry)

        logger.info(
            f"Loaded {len(self.benchmark_entries)} benchmark entries from HuggingFace"
        )

        # Step 2: Load document PDFs from the documents directory
        # Use shared configuration to get the documents directory location
        # This ensures consistency with where create_ait_qa_dataset.py downloads files
        ait_qa_pdf_document_folder = get_ait_qa_documents_dir()

        # Check if the directory exists
        if not ait_qa_pdf_document_folder.exists():
            raise FileNotFoundError(
                f"Documents directory not found: {ait_qa_pdf_document_folder}. "
                f"Please run create_ait_qa_dataset.py first to download the PDFs."
            )

        self.documents: list[DocumentObject] = []

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
            self.documents.append(doc)

        # Step 3: Initialize the parent RagDataLoader with configuration
        super().__init__(
            split=split,
            dataset_name="AIT_QA_Dataset",
            sampling_params=sampling_params,
            cache_dir=cache_dir,
        )

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        """Retrieve the benchmark entries for the specified split.

        This method implements the abstract _get_benchmark_entries method from
        RagDataLoader. It returns the pre-loaded benchmark entries that were
        parsed from the CSV file during initialization.

        Args:
            split: The dataset split to retrieve (DatasetSplit.TRAIN or DatasetSplit.TEST).
                If None, returns all entries. Otherwise, splits the data with a 70/30 ratio.

        Returns:
            A list of RagBenchmarkEntry objects containing questions, ground truth
            answers, and references to the source documents.
        """
        if split is None:
            # Return all the entries
            return self.benchmark_entries
        else:
            # Split the data into train/test sets
            items = self.benchmark_entries[:]
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
        It returns the pre-loaded PDF documents that were read from the documents
        directory during initialization.

        Returns:
            A list of DocumentObject instances, each containing:
                - name: The filename of the document
                - stream: A BytesIO buffer with the PDF content
                - mime_type: The MIME type of the document (e.g., 'application/pdf')

        Note:
            The documents are loaded into memory during initialization. Each document's
            content is stored as a BytesIO stream for efficient access.
        """
        return self.documents


# Made with Bob
