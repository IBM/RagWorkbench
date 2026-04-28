"""Tests for Milvus client (mocked)."""

from unittest.mock import MagicMock, patch

import pytest
from simple_rag.config import MilvusConfig
from simple_rag.milvus_client import MilvusIngester, MilvusRetriever


@pytest.fixture
def milvus_config():
    """Fixture for Milvus configuration."""
    return MilvusConfig(uri="./test_milvus.db")


@pytest.fixture
def mock_collection():
    """Fixture for mocked Milvus collection."""
    collection = MagicMock()
    collection.num_entities = 100
    return collection


class TestMilvusIngester:
    """Tests for MilvusIngester."""

    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.utility")
    @patch("simple_rag.milvus_client.connections")
    def test_initialization(
        self, mock_connections, mock_utility, mock_collection_class, milvus_config
    ):
        """Test ingester initialization."""
        mock_utility.has_collection.return_value = False
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        ingester = MilvusIngester(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        assert ingester.collection_name == "test_collection"
        assert ingester.dimension == 384
        mock_connections.connect.assert_called_once_with(
            alias="default", uri=milvus_config.uri
        )

    @patch("simple_rag.milvus_client.utility")
    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_create_collection_new(
        self, mock_connections, mock_collection_class, mock_utility, milvus_config
    ):
        """Test creating a new collection."""
        mock_utility.has_collection.return_value = False
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        _ = MilvusIngester(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        # Verify collection was created
        mock_utility.has_collection.assert_called_with("test_collection")
        mock_collection_class.assert_called()
        mock_collection.create_index.assert_called_once()

    @patch("simple_rag.milvus_client.utility")
    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_reuse_existing_collection(
        self, mock_connections, mock_collection_class, mock_utility, milvus_config
    ):
        """Test reusing an existing collection."""
        mock_utility.has_collection.return_value = True
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        _ = MilvusIngester(
            config=milvus_config, collection_name="test_collection", dimension=384
        )

        # Verify collection was reused, not created
        mock_utility.has_collection.assert_called_with("test_collection")
        mock_collection_class.assert_called_with(name="test_collection")
        mock_collection.create_index.assert_not_called()

    @patch("simple_rag.milvus_client.utility")
    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_insert_embeddings(
        self, mock_connections, mock_collection_class, mock_utility, milvus_config
    ):
        """Test embedding insertion."""
        mock_utility.has_collection.return_value = False
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        ingester = MilvusIngester(
            config=milvus_config, collection_name="test_collection", dimension=384
        )
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

    @patch("simple_rag.milvus_client.utility")
    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_get_indexed_documents(
        self, mock_connections, mock_collection_class, mock_utility, milvus_config
    ):
        """Test getting indexed documents."""
        mock_utility.has_collection.return_value = False
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        # Mock query results
        mock_collection.query.return_value = [
            {"document_id": "doc1"},
            {"document_id": "doc2"},
            {"document_id": "doc1"},  # Duplicate
        ]

        ingester = MilvusIngester(
            config=milvus_config, collection_name="test_collection", dimension=384
        )
        ingester.collection = mock_collection

        indexed_docs = ingester.get_indexed_documents()

        assert indexed_docs == {"doc1", "doc2"}
        mock_collection.load.assert_called_once()
        mock_collection.query.assert_called_once()

    @patch("simple_rag.milvus_client.utility")
    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_get_indexed_documents_empty(
        self, mock_connections, mock_collection_class, mock_utility, milvus_config
    ):
        """Test getting indexed documents from empty collection."""
        mock_utility.has_collection.return_value = False
        mock_collection = MagicMock()
        mock_collection_class.return_value = mock_collection

        # Mock query failure (empty collection)
        mock_collection.query.side_effect = Exception("Collection is empty")

        ingester = MilvusIngester(
            config=milvus_config, collection_name="test_collection", dimension=384
        )
        ingester.collection = mock_collection

        indexed_docs = ingester.get_indexed_documents()

        assert indexed_docs == set()
        mock_collection.load.assert_called_once()


class TestMilvusRetriever:
    """Tests for MilvusRetriever."""

    @patch("simple_rag.milvus_client.connections")
    def test_initialization(self, mock_connections, milvus_config):
        """Test retriever initialization."""
        retriever = MilvusRetriever(
            config=milvus_config, collection_name="test_collection"
        )

        assert retriever.collection_name == "test_collection"
        mock_connections.connect.assert_called_once_with(
            alias="default", uri=milvus_config.uri
        )

    @patch("simple_rag.milvus_client.Collection")
    @patch("simple_rag.milvus_client.connections")
    def test_search(self, mock_connections, mock_collection_class, milvus_config):
        """Test vector search."""
        retriever = MilvusRetriever(
            config=milvus_config, collection_name="test_collection"
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
