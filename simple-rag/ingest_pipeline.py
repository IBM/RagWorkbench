"""Ingestion pipeline for Simple RAG."""

import hashlib
import logging
from io import BytesIO

import litellm
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from pydantic import Field
from simple_rag.config import SimpleRagIngestParams
from simple_rag.milvus_client import MilvusVectorStore

from ragworkbench.api.ingest import IngestPipeline
from ragworkbench.api.ingest_artifact import IngestArtifact
from ragworkbench.datasets_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models import DocumentObject

logger = logging.getLogger(__name__)


class SimpleRagIngestArtifact(IngestArtifact):
    """Artifact containing ingestion results and connection details."""

    collection_name: str = Field(description="Milvus collection name")
    milvus_host: str = Field(description="Milvus server host")
    milvus_port: int = Field(description="Milvus server port")
    embedding_model: str = Field(description="Embedding model used")
    dimension: int = Field(description="Embedding dimension")


class SimpleRagIngestPipeline(IngestPipeline):
    """Simple RAG ingestion pipeline."""

    def __init__(self, _params: SimpleRagIngestParams) -> None:
        """Initialize the ingest pipeline."""
        super().__init__(_params)
        self._params: SimpleRagIngestParams = _params

    def _generate_collection_name(self) -> str:
        """Generate collection name from parameters hash."""
        # Create hash from key parameters
        params_str = (
            f"{self._params.embedding_model}"
            f"{self._params.chunking_config.tokenizer}"
            f"{self._params.chunking_config.max_tokens}"
            f"{self._params.chunking_config.merge_peers}"
        )
        hash_obj = hashlib.md5(params_str.encode())
        return f"simple_rag_{hash_obj.hexdigest()[:16]}"

    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from the model."""
        # Generate a test embedding to determine dimension
        response = litellm.embedding(
            model=self._params.embedding_model,
            input=["test"],
            api_key=self._params.tracking_api_key,
        )
        return len(response.data[0]["embedding"])

    def _convert_document(self, doc: DocumentObject) -> str:
        """Convert DocumentObject to text using Docling."""
        converter = DocumentConverter()

        # Convert BytesIO to bytes for Docling
        doc_bytes = doc.stream.read()
        doc.stream.seek(0)  # Reset stream position

        # Convert document
        result = converter.convert(BytesIO(doc_bytes))
        return result.document.export_to_markdown()

    def _chunk_text(self, text: str) -> list[str]:
        """Chunk text using Docling HybridChunker."""
        chunker = HybridChunker(
            tokenizer=self._params.chunking_config.tokenizer,
            max_tokens=self._params.chunking_config.max_tokens,
            merge_peers=self._params.chunking_config.merge_peers,
        )

        chunks = list(chunker.chunk(text))
        return [chunk.text for chunk in chunks]

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using LiteLLM."""
        response = litellm.embedding(
            model=self._params.embedding_model,
            input=texts,
            api_key=self._params.tracking_api_key,
        )
        return [item["embedding"] for item in response.data]

    def process(self, data_loader: RagDataLoader) -> list[IngestArtifact]:
        """
        Process documents and ingest into Milvus.

        Args:
            data_loader: Data loader providing documents

        Returns:
            List containing single IngestArtifact
        """
        logger.info("Starting Simple RAG ingestion")

        # Generate collection name
        collection_name = self._generate_collection_name()
        logger.info(f"Using collection name: {collection_name}")

        # Get embedding dimension
        dimension = self._get_embedding_dimension()
        logger.info(f"Embedding dimension: {dimension}")

        # Initialize Milvus
        vector_store = MilvusVectorStore(
            config=self._params.milvus_config,
            collection_name=collection_name,
            dimension=dimension,
        )
        vector_store.create_collection()

        # Get documents from data loader
        corpus = data_loader.get_corpus()
        documents = corpus.documents

        logger.info(f"Processing {len(documents)} documents")

        # Process each document
        all_chunks = []
        all_texts = []

        for doc in documents:
            # Convert document to text
            text = self._convert_document(doc)

            # Chunk text
            chunks = self._chunk_text(text)

            # Create chunk metadata
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"{doc.name}_{idx}"
                all_chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "document_id": doc.name,
                        "chunk_text": chunk_text,
                        "chunk_index": idx,
                    }
                )
                all_texts.append(chunk_text)

        logger.info(f"Generated {len(all_chunks)} chunks")

        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch = all_texts[i : i + batch_size]
            embeddings = self._generate_embeddings(batch)
            all_embeddings.extend(embeddings)
            logger.info(f"Generated embeddings for batch {i // batch_size + 1}")

        # Insert into Milvus
        vector_store.insert_embeddings(all_chunks, all_embeddings)

        # Create artifact
        artifact = SimpleRagIngestArtifact(
            collection_name=collection_name,
            milvus_host=self._params.milvus_config.host,
            milvus_port=self._params.milvus_config.port,
            embedding_model=self._params.embedding_model,
            dimension=dimension,
        )

        logger.info("Ingestion complete")
        return [artifact]
