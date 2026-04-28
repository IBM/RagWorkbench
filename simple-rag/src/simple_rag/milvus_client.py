"""Milvus vector store client for Simple RAG Pipeline."""

import logging
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from simple_rag.config import MilvusConfig

logger = logging.getLogger(__name__)


class MilvusIngester:
    """Handles ingestion operations for Milvus Lite vector database."""

    def __init__(
        self, config: MilvusConfig, collection_name: str, dimension: int
    ) -> None:
        """
        Initialize Milvus ingester.

        Args:
            config: Milvus Lite connection configuration
            collection_name: Name of the collection
            dimension: Embedding dimension
        """
        self.config = config
        self.collection_name = collection_name
        self.dimension = dimension
        self.collection: Collection | None = None

        # Connect to Milvus Lite
        connections.connect(alias="default", uri=self.config.uri)
        logger.info(f"Connected to Milvus Lite at {self.config.uri}")

        # Create collection
        self._create_collection()

    def _create_collection(self) -> None:
        """Create or reuse existing Milvus collection."""
        # Check if collection already exists
        if utility.has_collection(self.collection_name):
            logger.info(f"Reusing existing collection '{self.collection_name}'")
            self.collection = Collection(name=self.collection_name)
            return

        # Define schema for new collection
        fields = [
            FieldSchema(
                name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=255
            ),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension
            ),
        ]

        schema = CollectionSchema(
            fields=fields, description="Simple RAG document chunks"
        )

        # Create collection
        self.collection = Collection(name=self.collection_name, schema=schema)

        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        logger.info(f"Created collection '{self.collection_name}' with COSINE metric")

    def get_indexed_documents(self) -> set[str]:
        """
        Get set of document IDs that are already indexed in the collection.

        Returns:
            Set of document IDs present in the collection
        """
        if self.collection is None:
            self.collection = Collection(name=self.collection_name)

        # Load collection to query it
        self.collection.load()

        # Query for all unique document IDs
        # Use a simple query to get all document_ids
        try:
            results = self.collection.query(
                expr="chunk_index >= 0",  # Match all documents
                output_fields=["document_id"],
            )

            # Extract unique document IDs
            document_ids = {result["document_id"] for result in results}  # type: ignore[misc]
            logger.info(f"Found {len(document_ids)} documents already indexed")

            return document_ids
        except Exception as e:
            logger.warning(f"Failed to query indexed documents: {e}")
            # Return empty set if query fails (e.g., empty collection)
            return set()

    def insert_embeddings(
        self, chunks: list[dict[str, Any]], embeddings: list[list[float]]
    ) -> list[str]:
        """
        Insert chunks and embeddings into Milvus.

        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: List of embedding vectors

        Returns:
            List of inserted chunk IDs
        """
        if self.collection is None:
            self.collection = Collection(name=self.collection_name)

        # Prepare data for insertion
        data = [
            [chunk["chunk_id"] for chunk in chunks],
            [chunk["document_id"] for chunk in chunks],
            [chunk["chunk_text"] for chunk in chunks],
            [chunk["chunk_index"] for chunk in chunks],
            embeddings,
        ]

        # Insert data
        self.collection.insert(data)
        self.collection.flush()
        logger.info(f"Inserted {len(chunks)} chunks into '{self.collection_name}'")

        return [chunk["chunk_id"] for chunk in chunks]


class MilvusRetriever:
    """Handles retrieval operations for Milvus Lite vector database."""

    def __init__(self, config: MilvusConfig, collection_name: str) -> None:
        """
        Initialize Milvus retriever.

        Args:
            config: Milvus Lite connection configuration
            collection_name: Name of the collection
        """
        self.config = config
        self.collection_name = collection_name
        self.collection: Collection | None = None

        # Connect to Milvus Lite
        connections.connect(alias="default", uri=self.config.uri)
        logger.info(f"Connected to Milvus Lite at {self.config.uri}")

    def search(self, query_embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of search results with chunk data
        """
        if self.collection is None:
            self.collection = Collection(name=self.collection_name)

        # Load collection to memory
        self.collection.load()

        # Search
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["document_id", "chunk_text", "chunk_index"],
        )

        # Format results
        formatted_results = []
        for hits in results:  # type: ignore[misc]
            for hit in hits:
                formatted_results.append(
                    {
                        "chunk_id": hit.id,
                        "document_id": hit.entity.get("document_id"),
                        "chunk_text": hit.entity.get("chunk_text"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "distance": hit.distance,
                    }
                )

        return formatted_results
