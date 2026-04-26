"""Configuration models for Simple RAG Pipeline."""

from pydantic import BaseModel, Field

from ragworkbench.api.inference import InferenceParams
from ragworkbench.api.ingest import IngestParams


class MilvusConfig(BaseModel):
    """Milvus connection configuration."""

    host: str = Field(default="localhost", description="Milvus server host")
    port: int = Field(default=19530, description="Milvus server port")


class ChunkingConfig(BaseModel):
    """Document chunking configuration using Docling HybridChunker."""

    tokenizer: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Tokenizer for chunking",
    )
    max_tokens: int = Field(default=512, description="Maximum tokens per chunk")
    merge_peers: bool = Field(
        default=True, description="Merge peer chunks when possible"
    )


class SimpleRagIngestParams(IngestParams):
    """Parameters for Simple RAG ingestion pipeline."""

    milvus_config: MilvusConfig = Field(
        default_factory=MilvusConfig, description="Milvus connection settings"
    )
    chunking_config: ChunkingConfig = Field(
        default_factory=ChunkingConfig, description="Document chunking settings"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small", description="LiteLLM embedding model"
    )


class SimpleRagInferenceParams(InferenceParams):
    """Parameters for Simple RAG inference pipeline."""

    llm_model: str = Field(
        default="gpt-3.5-turbo", description="LiteLLM model for generation"
    )
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1)
