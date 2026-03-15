import io
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.datasets_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)
from ragworkbench.datasets_loader.data_models.document_object import DocumentObject
from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)
from ragworkbench.datasets_loader.dataset_names import DatasetName

# Constants
OFFICEQA_CSV_URL = (
    "https://raw.githubusercontent.com/databricks/officeqa/main/officeqa.csv"
)
OFFICEQA_DOCUMENTS_ZIP_URL = "https://github.com/databricks/officeqa/raw/main/treasury_bulletins_parsed/transformed/treasury_bulletins_transformed.zip"


class OfficeQADataLoader(RagDataLoader):
    """
    Data loader for the OfficeQA dataset from Databricks.

    The dataset contains questions and answers about US Treasury documents.

    Source: https://github.com/databricks/officeqa
    """

    def __init__(
        self,
        split: DatasetSplit | None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        """
        Initialize the OfficeQA data loader.

        Args:
            split: Dataset split ('train', 'test', or None for all data)
            sampling_params: Sampling parameters for the dataset
        """
        # Load the CSV from GitHub
        self.total_df: pd.DataFrame = pd.read_csv(OFFICEQA_CSV_URL)
        self._train_df = self.total_df.sample(frac=0.7, random_state=42)
        self._test_df = self.total_df.drop(self._train_df.index)

        super().__init__(
            dataset_name=DatasetName.OFFICEQA,
            split=split,
            sampling_params=sampling_params,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _source_file_to_document_id(source_file: str) -> str:
        """
        Convert source file name to a document ID.

        Args:
            source_file: Source file name (e.g., "treasury_bulletin_1941_01.txt")

        Returns:
            Document ID (cleaned version of the file name)
        """
        # Use the source file name as-is, or clean it if needed
        return source_file.strip()

    @staticmethod
    def _parse_source_files(source_files_str: str) -> list[str]:
        """
        Parse source files string which may contain multiple files separated by \\r\\n.

        Args:
            source_files_str: String containing one or more source file names

        Returns:
            List of individual source file names
        """
        # Split by \r\n and filter out empty strings
        files = [f.strip() for f in source_files_str.split("\r\n") if f.strip()]
        return files if files else [source_files_str.strip()]

    def _get_documents(self) -> list[DocumentObject]:
        """
        Extract unique documents from the dataset.

        Downloads the ZIP file containing treasury bulletin documents,
        extracts them in memory, and creates DocumentObject instances
        with the actual document content.

        Returns:
            List of DocumentObject instances
        """
        # Get unique source files from the dataset
        unique_source_files: set[str] = set()
        for _, row in self.total_df.iterrows():
            source_files_str = str(row["source_files"])
            # Parse multiple files if they exist
            source_files = self._parse_source_files(source_files_str)
            unique_source_files.update(source_files)

        # Download the ZIP file
        response = requests.get(OFFICEQA_DOCUMENTS_ZIP_URL, timeout=60)
        response.raise_for_status()

        # Extract documents from ZIP file in memory
        docs: list[DocumentObject] = []
        with ZipFile(io.BytesIO(response.content)) as zip_file:
            # Get all file names in the ZIP
            zip_file_names = zip_file.namelist()

            # Process each unique source file
            for source_file in unique_source_files:
                # Find the matching file in the ZIP
                # The ZIP may have a directory structure, so we look for files ending with the source_file name
                matching_files = [f for f in zip_file_names if f.endswith(source_file)]

                if not matching_files:
                    # If file not found in ZIP, create a placeholder
                    doc_content = f"Document {source_file} not found in ZIP archive."
                    content_bytes = doc_content.encode("utf-8")
                else:
                    # Read the first matching file
                    zip_path = matching_files[0]
                    with zip_file.open(zip_path) as file:
                        content_bytes = file.read()

                # Create DocumentObject
                doc: DocumentObject = DocumentObject(
                    name=self._source_file_to_document_id(source_file),
                    mime_type="text/markdown",
                    stream=io.BytesIO(content_bytes),
                )
                docs.append(doc)

        return docs

    def _get_split_dataframe(self, split: DatasetSplit | None) -> pd.DataFrame:
        """Get the appropriate dataframe for the given split."""
        if split is None:
            return self.total_df

        # At this point, both _train_df and _test_df are guaranteed to be DataFrames
        if split == "train":
            assert self._train_df is not None
            return self._train_df
        else:
            assert self._test_df is not None
            return self._test_df

    def _row_to_benchmark_entry(self, row: pd.Series) -> RagBenchmarkEntry:
        """Convert a dataframe row to a RagBenchmarkEntry."""
        source_files_str = str(row["source_files"])
        # Parse multiple files if they exist
        source_files = self._parse_source_files(source_files_str)

        # Create GroundTruthContextId for each source file
        ground_truth_context_ids = [
            GroundTruthContextId(
                document_id=self._source_file_to_document_id(source_file)
            )
            for source_file in source_files
        ]

        return RagBenchmarkEntry(
            question_id=str(row["uid"]),
            question=str(row["question"]),
            ground_truth_answers=[str(row["answer"])],
            ground_truths_context_ids=ground_truth_context_ids,
            is_answerable=True,
        )

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        """
        Create benchmark entries from the dataset.

        Args:
            split: Dataset split to use ('train', 'test', or None for all)

        Returns:
            List of RagBenchmarkEntry instances
        """
        df = self._get_split_dataframe(split)

        return [self._row_to_benchmark_entry(row) for _, row in df.iterrows()]
