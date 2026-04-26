"""Tests for Milvus client (mocked)."""

from unittest.mock import MagicMock, patch

import pytest
from simple_rag.config import MilvusConfig
from simple_rag.milvus_client import MilvusVectorStore


@pytest.fixture
def milvus_config():
    """Fixture for Milvus configuration."""
    return MilvusConfig(host="localhost", port=19530)


@pytest.fixture
def mock_collection():
    """Fixture for mocked Milvus collection."""
    collection = MagicMock()
    collection.num_entities = 100
    return collection


class TestMilvusVectorStore:
    """Tests for MilvusVectorStore."""

    @patch("simple_rag.milvus_client.connections")
    def test_initialization(self, mock_connections, milvus_config):
        """Test vector store initialization."""
        store = MilvusVectorStore(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        assert store.collection_name == "test_collection"
        assert store.dimension == 384
        mock_connections.connect.assert_called_once_with(
            alias="default", host="localhost", port=19530
        )

    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_create_collection(
        self, mock_connections, mock_collection_class, milvus_config
    ):
        """Test collection creation."""
        store = MilvusVectorStore(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        store.create_collection()

        # Verify collection was created
        mock_collection_class.assert_called_once()
        mock_collection.create_index.assert_called_once()

    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_insert_embeddings(
        self, mock_connections, mock_collection_class, milvus_config
    ):
        """Test embedding insertion."""
        store = MilvusVectorStore(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection
        store.collection = mock_collection

        chunks = [
            {
                "chunk_id": "doc1_0",
                "document_id": "doc1",
                "chunk_text": "test text",
                "chunk_index": 0,
            }
        ]
        embeddings = [[0.1] * 384]

        chunk_ids = store.insert_embeddings(chunks, embeddings)

        assert chunk_ids == ["doc1_0"]
        mock_collection.insert.assert_called_once()
        mock_collection.flush.assert_called_once()

    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_search(self, mock_connections, mock_collection_class, milvus_config):
        """Test vector search."""
        store = MilvusVectorStore(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        # Mock search results
        mock_hit = MagicMock()
        mock_hit.id = "doc1_0"
        mock_hit.distance = 0.95
        mock_hit.entity.get.side_effect = lambda key: {
            "document_id": "doc1",
            "chunk_text": "test text",
            "chunk_index": 0,
        }.get(key)

        mock_hits = [mock_hit]
        mock_collection = MagicMock()
        mock_collection.search.return_value = [mock_hits]
        mock_collection_class.return_value = mock_collection
        store.collection = mock_collection

        query_embedding = [0.1] * 384
        results = store.search(query_embedding, top_k=5)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "doc1_0"
        assert results[0]["document_id"] == "doc1"
        assert results[0]["distance"] == 0.95
        mock_collection.load.assert_called_once()
        mock_collection.search.assert_called_once()

    @patch("simple_rag.milvus_client.connections")
    def test_get_collection_stats(
        self, mock_connections, milvus_config, mock_collection
    ):
        """Test getting collection statistics."""
        store = MilvusVectorStore(
            config=milvus_config, collection_name="test_collection", dimension=384
        )
        store.collection = mock_collection

        stats = store.get_collection_stats()

        assert stats["name"] == "test_collection"
        assert stats["num_entities"] == 100
        assert stats["dimension"] == 384
