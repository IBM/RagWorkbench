import mimetypes
from pathlib import Path

from pydantic import BaseModel, Field

from ragworkbench.datasets_loader.data_models.document_object import DocumentObject


class RagCorpus(BaseModel):
    """
    A collection of documents forming a RAG (Retrieval-Augmented Generation) corpus.

    This class manages a corpus of documents used in RAG systems, providing utilities
    for accessing documents and exporting them to the filesystem. It ensures at least
    one document exists and provides convenient indexing and iteration patterns.

    Attributes:
        documents: List of DocumentObject instances in the corpus.

    Example:
        >>> corpus = RagCorpus(documents=[...])
        >>> print(f"Corpus contains {len(corpus)} documents")
        >>> first_doc = corpus[0]
        >>> corpus.export_to_folder(Path("./output"))
    """

    documents: list[DocumentObject] = Field(
        frozen=True,
        min_length=1,
        description="List of documents in the corpus. Must contain at least one document.",
    )

    def __len__(self) -> int:
        """
        Return the number of documents in the corpus.

        Returns:
            Integer count of documents.

        Example:
            >>> corpus = RagCorpus(documents=[...])
            >>> len(corpus)
            3
        """
        return len(self.documents)

    def __getitem__(self, idx: int) -> DocumentObject:
        """
        Access a document by index.

        Args:
            idx: Zero-based index of the document to retrieve.

        Returns:
            The DocumentObject at the specified index.

        Raises:
            IndexError: If the index is out of range.

        Example:
            >>> corpus = RagCorpus(documents=[...])
            >>> first_doc = corpus[0]
            >>> last_doc = corpus[-1]
        """
        return self.documents[idx]

    def export_to_folder(self, output_folder: Path) -> None:
        """
        Export all documents in the corpus to a specified folder.

        This method writes each document to disk, creating the output folder if it
        doesn't exist. File names are derived from document names, with appropriate
        extensions added based on MIME types if not already present.

        Args:
            output_folder: Path to the directory where documents will be exported.
                          Created if it doesn't exist.

        Note:
            - Existing files with the same name will be overwritten.
            - Document streams are rewound before reading to ensure complete content.
            - File extensions are automatically added based on MIME type if missing.

        Example:
            >>> corpus = RagCorpus(documents=[...])
            >>> corpus.export_to_folder(Path("./exported_docs"))
            # Creates ./exported_docs/ with all documents written to disk
        """
        output_folder.mkdir(parents=True, exist_ok=True)

        doc: DocumentObject
        for doc in self.documents:
            # Get document stream and rewind to beginning
            data = doc.stream
            data.seek(0)

            # Determine file extension from MIME type
            file_extension: str | None = mimetypes.guess_extension(doc.mime_type)
            file_name: str = doc.name

            # Add extension if not already present
            if file_extension and not file_name.endswith(file_extension):
                file_name += file_extension

            # Write document to file
            output_path: Path = output_folder / file_name
            with open(output_path, "wb") as f:
                f.write(data.read())
