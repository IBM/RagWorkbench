"""Tests for ingestion pipeline."""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from simple_rag.config import ChunkingConfig, MilvusConfig, SimpleRagIngestParams
from simple_rag.ingest_pipeline import SimpleRagIngestArtifact, SimpleRagIngestPipeline

from ragworkbench.datasets_loader.data_models import DocumentObject, RagCorpus


@pytest.fixture
def ingest_params():
    """Fixture for ingest parameters."""
    return SimpleRagIngestParams(
        milvus_config=MilvusConfig(),
        chunking_config=ChunkingConfig(max_tokens=256),
        embedding_model="text-embedding-3-small",
    )


@pytest.fixture
def mock_data_loader():
    """Fixture for mocked data loader."""
    loader = MagicMock()

    # Create mock document
    doc = DocumentObject(
        name="test_doc",
        stream=BytesIO(b"test content"),
        mime_type="text/plain",
    )

    corpus = RagCorpus(documents=[doc])
    loader.get_corpus.return_value = corpus

    return loader


class TestSimpleRagIngestPipeline:
    """Tests for SimpleRagIngestPipeline."""

    def test_initialization(self, ingest_params):
        """Test pipeline initialization."""
        pipeline = SimpleRagIngestPipeline(ingest_params)
        assert pipeline._params == ingest_params

    def test_generate_collection_name(self, ingest_params):
        """Test collection name generation."""
        pipeline = SimpleRagIngestPipeline(ingest_params)
        name1 = pipeline._generate_collection_name()
        name2 = pipeline._generate_collection_name()

        # Same params should generate same name
        assert name1 == name2
        assert name1.startswith("simple_rag_")
        assert len(name1) == 27  # "simple_rag_" + 16 hex chars

    def test_generate_collection_name_different_params(self):
        """Test that different params generate different names."""
        params1 = SimpleRagIngestParams(embedding_model="model1")
        params2 = SimpleRagIngestParams(embedding_model="model2")

        pipeline1 = SimpleRagIngestPipeline(params1)
        pipeline2 = SimpleRagIngestPipeline(params2)

        name1 = pipeline1._generate_collection_name()
        name2 = pipeline2._generate_collection_name()

        assert name1 != name2

    @patch("simple_rag.ingest_pipeline.litellm.embedding")
    def test_get_embedding_dimension(self, mock_embedding, ingest_params):
        """Test getting embedding dimension."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1] * 384}])

        pipeline = SimpleRagIngestPipeline(ingest_params)
        dimension = pipeline._get_embedding_dimension()

        assert dimension == 384
        mock_embedding.assert_called_once()

    @patch("simple_rag.ingest_pipeline.DocumentConverter")
    def test_convert_document(self, mock_converter_class, ingest_params):
        """Test document conversion."""
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = (
            "# Test Document\n\nTest content"
        )

        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        pipeline = SimpleRagIngestPipeline(ingest_params)
        doc = DocumentObject(
            name="test.pdf",
            stream=BytesIO(b"pdf content"),
            mime_type="application/pdf",
        )

        text = pipeline._convert_document(doc)

        assert text == "# Test Document\n\nTest content"
        mock_converter.convert.assert_called_once()

    @patch("simple_rag.ingest_pipeline.HybridChunker")
    def test_chunk_text(self, mock_chunker_class, ingest_params):
        """Test text chunking."""
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Chunk 1"
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "Chunk 2"

        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk1, mock_chunk2]
        mock_chunker_class.return_value = mock_chunker

        pipeline = SimpleRagIngestPipeline(ingest_params)
        chunks = pipeline._chunk_text("Test text")

        assert chunks == ["Chunk 1", "Chunk 2"]
        mock_chunker.chunk.assert_called_once_with("Test text")

    @patch("simple_rag.ingest_pipeline.litellm.embedding")
    def test_generate_embeddings(self, mock_embedding, ingest_params):
        """Test embedding generation."""
        mock_embedding.return_value = MagicMock(
            data=[
                {"embedding": [0.1] * 384},
                {"embedding": [0.2] * 384},
            ]
        )

        pipeline = SimpleRagIngestPipeline(ingest_params)
        embeddings = pipeline._generate_embeddings(["text1", "text2"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        mock_embedding.assert_called_once()

    @patch("simple_rag.ingest_pipeline.MilvusIngester")
    @patch("simple_rag.ingest_pipeline.litellm.embedding")
    @patch("simple_rag.ingest_pipeline.HybridChunker")
    @patch("simple_rag.ingest_pipeline.DocumentConverter")
    def test_process(
        self,
        mock_converter_class,
        mock_chunker_class,
        mock_embedding,
        mock_ingester_class,
        ingest_params,
        mock_data_loader,
    ):
        """Test full ingestion process."""
        # Setup mocks
        mock_result = MagicMock()
        mock_result.document.export_to_markdown.return_value = "Test content"
        mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter

        mock_chunk = MagicMock()
        mock_chunk.text = "Test chunk"
        mock_chunker = MagicMock()
        mock_chunker.chunk.return_value = [mock_chunk]
        mock_chunker_class.return_value = mock_chunker

        mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1] * 384}])

        mock_ingester = MagicMock()
        mock_ingester_class.return_value = mock_ingester

        # Run process
        pipeline = SimpleRagIngestPipeline(ingest_params)
        artifacts = pipeline.process(mock_data_loader)

        # Verify
        assert len(artifacts) == 1
        artifact = artifacts[0]
        assert isinstance(artifact, SimpleRagIngestArtifact)
        assert artifact.collection_name.startswith("simple_rag_")
        assert artifact.milvus_host == "localhost"
        assert artifact.milvus_port == 19530
        assert artifact.embedding_model == "text-embedding-3-small"
        assert artifact.dimension == 384

        mock_ingester.create_collection.assert_called_once()
        mock_ingester.insert_embeddings.assert_called_once()
