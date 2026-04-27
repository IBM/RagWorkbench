"""Ingestion pipeline for Simple RAG."""

import hashlib
import logging
from pathlib import Path

import litellm
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from pydantic import Field
from transformers import AutoTokenizer

from ragworkbench.api.ingest import IngestPipeline
from ragworkbench.api.ingest_artifact import IngestArtifact
from ragworkbench.boards.board_model import CacheMode
from ragworkbench.boards.board_registry import ingest_pipeline
from ragworkbench.datasets_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models import DocumentObject
from simple_rag.config import SimpleRagIngestParams
from simple_rag.docling_cache import DoclingCache
from simple_rag.milvus_client import MilvusIngester

# Suppress docling log messages
logging.getLogger("docling").setLevel(logging.WARNING)
logging.getLogger("docling.document_converter").setLevel(logging.WARNING)
logging.getLogger("docling.chunking").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class SimpleRagIngestArtifact(IngestArtifact):
    """Artifact containing ingestion results and connection details for Milvus Lite."""

    collection_name: str = Field(description="Milvus collection name")
    milvus_uri: str = Field(description="Milvus Lite URI")
    embedding_model: str = Field(description="Embedding model used")


@ingest_pipeline(name="simple_rag", params_class=SimpleRagIngestParams)
class SimpleRagIngestPipeline(IngestPipeline):
    """Simple RAG ingestion pipeline."""

    EMBEDDING_TIMEOUT = 30  # seconds

    def __init__(
        self,
        params: SimpleRagIngestParams,
        cache_dir: Path | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ) -> None:
        """Initialize the ingest pipeline."""
        super().__init__(_params=params, cache_dir=cache_dir, cache_mode=cache_mode)
        self._params: SimpleRagIngestParams = params

        # Initialize tokenizer for truncation
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._params.chunking_config.tokenizer
        )
        logger.info(f"Initialized tokenizer: {self._params.chunking_config.tokenizer}")

        # Initialize Docling cache if cache_dir is provided and mode is not OFF
        # Note: DocumentConverter has no configurable parameters that affect conversion,
        # so we don't need a config_dict for cache subdirectory hashing
        self._docling_cache = None
        if cache_dir is not None and cache_mode != CacheMode.OFF:
            self._docling_cache = DoclingCache(
                cache_dir=cache_dir,
                cache_mode=cache_mode,
            )
            logger.info(
                f"Docling cache initialized at {self._docling_cache.cache_path} "
                f"(mode: {cache_mode.value})"
            )

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
        logger.info(
            f"Getting embedding dimension for model: {self._params.embedding_model}"
        )
        logger.info(
            f"Dimension check params: model={self._params.embedding_model}, "
            f"input=['test'], "
            f"api_key={'***' if self._params.tracking_api_key else None}, "
            f"timeout={self.EMBEDDING_TIMEOUT}"
        )
        response = litellm.embedding(
            model=self._params.embedding_model,
            input=["test"],
            api_key=self._params.tracking_api_key,
            timeout=self.EMBEDDING_TIMEOUT,
        )
        dimension = len(response.data[0]["embedding"])
        logger.info(f"Embedding dimension: {dimension}")
        return dimension

    def _convert_document(self, doc: DocumentObject):
        """
        Convert DocumentObject to DoclingDocument using Docling.

        Uses cache if enabled to avoid re-converting the same document.
        """
        # Check cache first if enabled
        if self._docling_cache is not None:
            cached_doc = self._docling_cache.get(doc.name)
            if cached_doc is not None:
                logger.debug(f"Cache hit for document: {doc.name}")
                return cached_doc
            logger.debug(f"Cache miss for document: {doc.name}")

        # Convert document
        converter = DocumentConverter()
        result = converter.convert(doc)
        docling_doc = result.document

        # Store in cache if enabled
        if self._docling_cache is not None:
            self._docling_cache.add(doc.name, docling_doc)
            logger.debug(f"Cached document: {doc.name}")

        return docling_doc

    def _chunk_document(self, docling_doc) -> list[str]:
        """Chunk DoclingDocument using Docling HybridChunker."""
        chunker = HybridChunker(
            tokenizer=self._params.chunking_config.tokenizer,
            max_tokens=self._params.chunking_config.max_tokens,
            merge_peers=self._params.chunking_config.merge_peers,
        )

        chunks = list(chunker.chunk(docling_doc))
        return [chunk.text for chunk in chunks]

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_tokens using the configured tokenizer."""
        max_tokens = self._params.chunking_config.max_tokens
        tokens = self._tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) > max_tokens:
            logger.warning(f"Text has {len(tokens)} tokens, truncating to {max_tokens}")
            truncated_tokens = tokens[:max_tokens]
            return self._tokenizer.decode(truncated_tokens, skip_special_tokens=True)

        return text

    def _generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using LiteLLM."""
        # Truncate all texts to max_tokens
        truncated_texts = [self._truncate_text(text) for text in texts]

        logger.info(
            f"Generating embeddings for {len(truncated_texts)} texts using {self._params.embedding_model}"
        )
        response = litellm.embedding(
            model=self._params.embedding_model,
            input=truncated_texts,
            api_key=self._params.tracking_api_key,
            timeout=self.EMBEDDING_TIMEOUT,
        )
        embeddings = [item["embedding"] for item in response.data]
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

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

        # Initialize Milvus ingester
        ingester = MilvusIngester(
            config=self._params.milvus_config,
            collection_name=collection_name,
            dimension=dimension,
        )

        # Get documents from data loader
        corpus = data_loader.get_corpus()
        documents = corpus.documents

        logger.info(f"Processing {len(documents)} documents")

        # Check which documents are already indexed
        indexed_documents = ingester.get_indexed_documents()
        if indexed_documents:
            logger.info(
                f"Found {len(indexed_documents)} documents already indexed in collection '{collection_name}'"
            )

        # Filter out already-indexed documents
        documents_to_process = [
            doc for doc in documents if doc.name not in indexed_documents
        ]

        if not documents_to_process:
            logger.info("All documents are already indexed. Skipping ingestion.")
            # Create artifact with existing collection
            artifact = SimpleRagIngestArtifact(
                collection_name=collection_name,
                milvus_uri=self._params.milvus_config.uri,
                embedding_model=self._params.embedding_model,
            )
            return [artifact]

        logger.info(
            f"Processing {len(documents_to_process)} new documents "
            f"(skipping {len(indexed_documents)} already indexed)"
        )

        # Process each document
        all_chunks = []
        all_texts = []

        for doc in documents_to_process:
            # Convert document to DoclingDocument
            docling_doc = self._convert_document(doc)

            # Chunk document
            chunks = self._chunk_document(docling_doc)

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
        ingester.insert_embeddings(all_chunks, all_embeddings)

        # Create artifact
        artifact = SimpleRagIngestArtifact(
            collection_name=collection_name,
            milvus_uri=self._params.milvus_config.uri,
            embedding_model=self._params.embedding_model,
        )

        # Log cache statistics if cache is enabled
        if self._docling_cache is not None:
            cache_stats = self._docling_cache.get_cache_stats()
            logger.info(
                f"Docling cache stats - Hits: {cache_stats['cache_hit']}, "
                f"Misses: {cache_stats['cache_miss']}, "
                f"Total entries: {cache_stats['total_entries']}"
            )

        logger.info("Ingestion complete")
        return [artifact]
