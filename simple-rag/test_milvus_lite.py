#!/usr/bin/env python3
"""Quick test script to verify Milvus Lite compatibility."""

import tempfile
from pathlib import Path

from simple_rag.config import MilvusConfig
from simple_rag.milvus_client import MilvusIngester, MilvusRetriever


def test_milvus_lite_file_based():
    """Test basic Milvus Lite operations with file-based storage."""
    # Create a temporary database file
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_milvus.db"

        # Configure Milvus Lite with file-based storage
        config = MilvusConfig(uri=str(db_path))
        collection_name = "test_collection"
        dimension = 384

        print(f"Testing Milvus Lite with database: {db_path}")

        # Test ingestion
        print("\n1. Testing MilvusIngester...")
        ingester = MilvusIngester(config, collection_name, dimension)
        ingester.create_collection()
        print("   ✓ Collection created successfully")

        # Insert test data
        chunks = [
            {
                "chunk_id": "test_1",
                "document_id": "doc_1",
                "chunk_text": "This is a test chunk",
                "chunk_index": 0,
            },
            {
                "chunk_id": "test_2",
                "document_id": "doc_1",
                "chunk_text": "This is another test chunk",
                "chunk_index": 1,
            },
        ]
        embeddings = [[0.1] * dimension, [0.2] * dimension]

        chunk_ids = ingester.insert_embeddings(chunks, embeddings)
        print(f"   ✓ Inserted {len(chunk_ids)} chunks")

        # Test retrieval
        print("\n2. Testing MilvusRetriever...")
        retriever = MilvusRetriever(config, collection_name, dimension)

        query_embedding = [0.15] * dimension
        results = retriever.search(query_embedding, top_k=2)
        print(f"   ✓ Retrieved {len(results)} results")

        # Verify results
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert results[0]["chunk_id"] in ["test_1", "test_2"]
        print("   ✓ Results validated")

        print("\n✅ All tests passed! Milvus Lite is working correctly.")
        return True


def test_milvus_lite_in_memory():
    """Test Milvus Lite with in-memory database."""
    print("\n" + "=" * 60)
    print("Testing Milvus Lite with in-memory database")
    print("=" * 60)

    # Configure Milvus Lite with in-memory storage
    config = MilvusConfig(uri=":memory:")
    collection_name = "memory_test_collection"
    dimension = 128

    print("Testing with :memory: URI...")

    # Test ingestion
    print("\n1. Testing MilvusIngester...")
    ingester = MilvusIngester(config, collection_name, dimension)
    ingester.create_collection()
    print("   ✓ Collection created successfully")

    # Insert test data
    chunks = [
        {
            "chunk_id": "mem_test_1",
            "document_id": "mem_doc_1",
            "chunk_text": "Memory test chunk",
            "chunk_index": 0,
        }
    ]
    embeddings = [[0.5] * dimension]

    chunk_ids = ingester.insert_embeddings(chunks, embeddings)
    print(f"   ✓ Inserted {len(chunk_ids)} chunks")

    # Test retrieval
    print("\n2. Testing MilvusRetriever...")
    retriever = MilvusRetriever(config, collection_name, dimension)

    query_embedding = [0.5] * dimension
    results = retriever.search(query_embedding, top_k=1)
    print(f"   ✓ Retrieved {len(results)} results")

    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    print("   ✓ Results validated")

    print("\n✅ In-memory test passed!")
    return True


if __name__ == "__main__":
    try:
        # Test with file-based storage
        test_milvus_lite_file_based()

        # Test with in-memory storage
        test_milvus_lite_in_memory()

        print("\n" + "=" * 60)
        print("🎉 All Milvus Lite tests passed successfully!")
        print("   Simple RAG now works with Milvus Lite!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
