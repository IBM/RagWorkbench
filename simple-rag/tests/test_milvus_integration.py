"""Integration tests for Milvus client with real Milvus Lite database."""

import tempfile
from pathlib import Path

import pytest
from pymilvus import connections
from simple_rag.config import MilvusConfig
from simple_rag.milvus_client import MilvusIngester, MilvusRetriever


@pytest.fixture
def temp_milvus_db():
    """Fixture that provides a temporary Milvus Lite database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_milvus.db"
        config = MilvusConfig(uri=str(db_path))
        yield config, db_path
        # Cleanup: disconnect after each test
        try:
            connections.disconnect(alias="default")
        except Exception:
            pass  # Ignore if already disconnected


class TestMilvusIngesterIntegration:
    """Integration tests for MilvusIngester with real database."""

    def test_create_and_reuse_collection(self, temp_milvus_db):
        """Test creating a collection and then reusing it."""
        config, db_path = temp_milvus_db
        collection_name = "test_collection"
        dimension = 128

        # First ingester - should create collection
        ingester1 = MilvusIngester(config, collection_name, dimension)
        assert ingester1.collection is not None

        # Insert some test data
        chunks = [
            {
                "chunk_id": "doc1_0",
                "document_id": "doc1",
                "chunk_text": "First test chunk",
                "chunk_index": 0,
            },
            {
                "chunk_id": "doc1_1",
                "document_id": "doc1",
                "chunk_text": "Second test chunk",
                "chunk_index": 1,
            },
        ]
        embeddings = [[0.1] * dimension, [0.2] * dimension]
        chunk_ids = ingester1.insert_embeddings(chunks, embeddings)
        assert len(chunk_ids) == 2

        # Second ingester - should reuse existing collection
        ingester2 = MilvusIngester(config, collection_name, dimension)
        assert ingester2.collection is not None

        # Verify we can query the existing data
        indexed_docs = ingester2.get_indexed_documents()
        assert "doc1" in indexed_docs

    def test_get_indexed_documents(self, temp_milvus_db):
        """Test getting indexed documents from a real database."""
        config, db_path = temp_milvus_db
        collection_name = "test_indexed_docs"
        dimension = 128

        # Create ingester and insert data
        ingester = MilvusIngester(config, collection_name, dimension)

        # Insert documents from multiple sources
        chunks = [
            {
                "chunk_id": "doc1_0",
                "document_id": "doc1",
                "chunk_text": "Document 1 chunk",
                "chunk_index": 0,
            },
            {
                "chunk_id": "doc2_0",
                "document_id": "doc2",
                "chunk_text": "Document 2 chunk",
                "chunk_index": 0,
            },
            {
                "chunk_id": "doc2_1",
                "document_id": "doc2",
                "chunk_text": "Document 2 second chunk",
                "chunk_index": 1,
            },
        ]
        embeddings = [[0.1] * dimension, [0.2] * dimension, [0.3] * dimension]
        ingester.insert_embeddings(chunks, embeddings)

        # Get indexed documents
        indexed_docs = ingester.get_indexed_documents()

        # Verify both documents are found
        assert len(indexed_docs) == 2
        assert "doc1" in indexed_docs
        assert "doc2" in indexed_docs

    def test_incremental_indexing(self, temp_milvus_db):
        """Test incremental indexing - adding new documents to existing collection."""
        config, db_path = temp_milvus_db
        collection_name = "test_incremental"
        dimension = 128

        # First batch of documents
        ingester1 = MilvusIngester(config, collection_name, dimension)
        chunks1 = [
            {
                "chunk_id": "doc1_0",
                "document_id": "doc1",
                "chunk_text": "First document",
                "chunk_index": 0,
            }
        ]
        embeddings1 = [[0.1] * dimension]
        ingester1.insert_embeddings(chunks1, embeddings1)

        # Check indexed documents
        indexed_docs = ingester1.get_indexed_documents()
        assert len(indexed_docs) == 1
        assert "doc1" in indexed_docs

        # Second batch - reuse collection and add more documents
        ingester2 = MilvusIngester(config, collection_name, dimension)
        chunks2 = [
            {
                "chunk_id": "doc2_0",
                "document_id": "doc2",
                "chunk_text": "Second document",
                "chunk_index": 0,
            },
            {
                "chunk_id": "doc3_0",
                "document_id": "doc3",
                "chunk_text": "Third document",
                "chunk_index": 0,
            },
        ]
        embeddings2 = [[0.2] * dimension, [0.3] * dimension]
        ingester2.insert_embeddings(chunks2, embeddings2)

        # Verify all documents are now indexed
        indexed_docs = ingester2.get_indexed_documents()
        assert len(indexed_docs) == 3
        assert "doc1" in indexed_docs
        assert "doc2" in indexed_docs
        assert "doc3" in indexed_docs

    def test_empty_collection_get_indexed_documents(self, temp_milvus_db):
        """Test getting indexed documents from an empty collection."""
        config, db_path = temp_milvus_db
        collection_name = "test_empty"
        dimension = 128

        # Create empty collection
        ingester = MilvusIngester(config, collection_name, dimension)

        # Should return empty set for empty collection
        indexed_docs = ingester.get_indexed_documents()
        assert len(indexed_docs) == 0


class TestMilvusRetrieverIntegration:
    """Integration tests for MilvusRetriever with real database."""

    def test_search_with_real_data(self, temp_milvus_db):
        """Test searching with real Milvus Lite database."""
        config, db_path = temp_milvus_db
        collection_name = "test_search"
        dimension = 128

        # Insert test data
        ingester = MilvusIngester(config, collection_name, dimension)
        chunks = [
            {
                "chunk_id": "doc1_0",
                "document_id": "doc1",
                "chunk_text": "First test chunk",
                "chunk_index": 0,
            },
            {
                "chunk_id": "doc1_1",
                "document_id": "doc1",
                "chunk_text": "Second test chunk",
                "chunk_index": 1,
            },
        ]
        embeddings = [[0.1] * dimension, [0.9] * dimension]
        ingester.insert_embeddings(chunks, embeddings)

        # Search with retriever
        retriever = MilvusRetriever(config, collection_name)
        query_embedding = [0.15] * dimension  # Closer to first embedding
        results = retriever.search(query_embedding, top_k=2)

        # Verify results
        assert len(results) == 2
        assert results[0]["document_id"] == "doc1"
        assert "chunk_text" in results[0]
        assert "distance" in results[0]

    def test_search_top_k(self, temp_milvus_db):
        """Test that top_k parameter works correctly."""
        config, db_path = temp_milvus_db
        collection_name = "test_top_k"
        dimension = 128

        # Insert multiple chunks
        ingester = MilvusIngester(config, collection_name, dimension)
        chunks = [
            {
                "chunk_id": f"doc1_{i}",
                "document_id": "doc1",
                "chunk_text": f"Chunk {i}",
                "chunk_index": i,
            }
            for i in range(5)
        ]
        embeddings = [[float(i) / 10] * dimension for i in range(5)]
        ingester.insert_embeddings(chunks, embeddings)

        # Search with different top_k values
        retriever = MilvusRetriever(config, collection_name)
        query_embedding = [0.25] * dimension

        results_k1 = retriever.search(query_embedding, top_k=1)
        assert len(results_k1) == 1

        results_k3 = retriever.search(query_embedding, top_k=3)
        assert len(results_k3) == 3

        results_k5 = retriever.search(query_embedding, top_k=5)
        assert len(results_k5) == 5


class TestMilvusEndToEnd:
    """End-to-end integration tests."""

    def test_full_workflow(self, temp_milvus_db):
        """Test complete workflow: create, insert, reuse, query."""
        config, db_path = temp_milvus_db
        collection_name = "test_workflow"
        dimension = 128

        # Step 1: Create collection and insert initial data
        ingester1 = MilvusIngester(config, collection_name, dimension)
        chunks1 = [
            {
                "chunk_id": "doc1_0",
                "document_id": "doc1",
                "chunk_text": "Initial document",
                "chunk_index": 0,
            }
        ]
        embeddings1 = [[0.5] * dimension]
        ingester1.insert_embeddings(chunks1, embeddings1)

        # Step 2: Verify document is indexed
        indexed_docs = ingester1.get_indexed_documents()
        assert "doc1" in indexed_docs

        # Step 3: Reuse collection and add more data
        ingester2 = MilvusIngester(config, collection_name, dimension)
        chunks2 = [
            {
                "chunk_id": "doc2_0",
                "document_id": "doc2",
                "chunk_text": "Additional document",
                "chunk_index": 0,
            }
        ]
        embeddings2 = [[0.7] * dimension]
        ingester2.insert_embeddings(chunks2, embeddings2)

        # Step 4: Verify both documents are indexed
        indexed_docs = ingester2.get_indexed_documents()
        assert len(indexed_docs) == 2
        assert "doc1" in indexed_docs
        assert "doc2" in indexed_docs

        # Step 5: Search and verify results
        retriever = MilvusRetriever(config, collection_name)
        query_embedding = [0.6] * dimension
        results = retriever.search(query_embedding, top_k=2)

        assert len(results) == 2
        doc_ids = {result["document_id"] for result in results}
        assert doc_ids == {"doc1", "doc2"}


# Made with Bob
