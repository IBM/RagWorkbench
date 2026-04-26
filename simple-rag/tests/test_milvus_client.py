"""Tests for Milvus client (mocked)."""

from unittest.mock import MagicMock, patch

import pytest
from simple_rag.config import MilvusConfig
from simple_rag.milvus_client import MilvusIngester, MilvusRetriever


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


class TestMilvusIngester:
    """Tests for MilvusIngester."""

    @patch("simple_rag.milvus_client.connections")
    def test_initialization(self, mock_connections, milvus_config):
        """Test ingester initialization."""
        ingester = MilvusIngester(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        assert ingester.collection_name == "test_collection"
        assert ingester.dimension == 384
        mock_connections.connect.assert_called_once_with(
            alias="default", host="localhost", port=19530
        )

    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_create_collection(
        self, mock_connections, mock_collection_class, milvus_config
    ):
        """Test collection creation."""
        ingester = MilvusIngester(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        ingester.create_collection()

        # Verify collection was created
        mock_collection_class.assert_called_once()
        mock_collection.create_index.assert_called_once()

    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_insert_embeddings(
        self, mock_connections, mock_collection_class, milvus_config
    ):
        """Test embedding insertion."""
        ingester = MilvusIngester(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection
        ingester.collection = mock_collection

        chunks = [
            {
                "chunk_id": "doc1_0",
                "document_id": "doc1",
                "chunk_text": "test text",
                "chunk_index": 0,
            }
        ]
        embeddings = [[0.1] * 384]

        chunk_ids = ingester.insert_embeddings(chunks, embeddings)

        assert chunk_ids == ["doc1_0"]
        mock_collection.insert.assert_called_once()
        mock_collection.flush.assert_called_once()


class TestMilvusRetriever:
    """Tests for MilvusRetriever."""

    @patch("simple_rag.milvus_client.connections")
    def test_initialization(self, mock_connections, milvus_config):
        """Test retriever initialization."""
        retriever = MilvusRetriever(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        assert retriever.collection_name == "test_collection"
        mock_connections.connect.assert_called_once_with(
            alias="default", host="localhost", port=19530
        )

    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_search(self, mock_connections, mock_collection_class, milvus_config):
        """Test vector search."""
        retriever = MilvusRetriever(
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
        retriever.collection = mock_collection

        query_embedding = [0.1] * 384
        results = retriever.search(query_embedding, top_k=5)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "doc1_0"
        assert results[0]["document_id"] == "doc1"
        assert results[0]["distance"] == 0.95
        mock_collection.load.assert_called_once()
        mock_collection.search.assert_called_once()
