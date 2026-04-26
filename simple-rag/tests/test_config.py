"""Tests for configuration models."""

import pytest
from pydantic import ValidationError
from simple_rag.config import (
    ChunkingConfig,
    MilvusConfig,
    SimpleRagInferenceParams,
    SimpleRagIngestParams,
)


class TestMilvusConfig:
    """Tests for MilvusConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MilvusConfig()
        assert config.host == "localhost"
        assert config.port == 19530

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MilvusConfig(host="milvus-server", port=9091)
        assert config.host == "milvus-server"
        assert config.port == 9091


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChunkingConfig()
        assert config.tokenizer == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.max_tokens == 512
        assert config.merge_peers is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ChunkingConfig(
            tokenizer="custom-tokenizer", max_tokens=256, merge_peers=False
        )
        assert config.tokenizer == "custom-tokenizer"
        assert config.max_tokens == 256
        assert config.merge_peers is False


class TestSimpleRagIngestParams:
    """Tests for SimpleRagIngestParams."""

    def test_default_values(self):
        """Test default parameter values."""
        params = SimpleRagIngestParams()
        assert params.milvus_config.host == "localhost"
        assert params.chunking_config.max_tokens == 512
        assert params.embedding_model == "text-embedding-3-small"

    def test_custom_values(self):
        """Test custom parameter values."""
        params = SimpleRagIngestParams(
            milvus_config=MilvusConfig(host="custom-host"),
            chunking_config=ChunkingConfig(max_tokens=1024),
            embedding_model="custom-model",
        )
        assert params.milvus_config.host == "custom-host"
        assert params.chunking_config.max_tokens == 1024
        assert params.embedding_model == "custom-model"


class TestSimpleRagInferenceParams:
    """Tests for SimpleRagInferenceParams."""

    def test_default_values(self):
        """Test default parameter values."""
        params = SimpleRagInferenceParams()
        assert params.llm_model == "gpt-3.5-turbo"
        assert params.top_k == 5

    def test_custom_values(self):
        """Test custom parameter values."""
        params = SimpleRagInferenceParams(llm_model="gpt-4", top_k=10)
        assert params.llm_model == "gpt-4"
        assert params.top_k == 10

    def test_top_k_validation(self):
        """Test that top_k must be >= 1."""
        with pytest.raises(ValidationError):
            SimpleRagInferenceParams(top_k=0)

        with pytest.raises(ValidationError):
            SimpleRagInferenceParams(top_k=-1)
