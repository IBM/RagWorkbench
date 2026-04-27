"""Configuration models for Simple RAG Pipeline."""

from pathlib import Path

from pydantic import BaseModel, Field

from ragworkbench.api.inference import InferenceParams
from ragworkbench.api.ingest import IngestParams
from ragworkbench.boards.board_model import CacheMode


class MilvusConfig(BaseModel):
    """Milvus Lite connection configuration.

    Uses Milvus Lite for embedded vector storage.
    Examples:
    - File-based: "./milvus_demo.db"
    - In-memory: ":memory:"
    """

    uri: str = Field(
        default="./milvus.db",
        description="Milvus Lite URI (e.g., './milvus.db' or ':memory:')",
    )


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


class DoclingCacheConfig(BaseModel):
    """Configuration for Docling document caching."""

    cache_dir: Path = Field(
        default=Path("./cache"),
        description="Directory for Docling document cache",
    )
    cache_mode: CacheMode = Field(
        default=CacheMode.ON,
        description="Cache operation mode (ON/OFF/REFRESH)",
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
    docling_cache_config: DoclingCacheConfig = Field(
        default_factory=DoclingCacheConfig,
        description="Docling document cache configuration",
    )


class SimpleRagInferenceParams(InferenceParams):
    """Parameters for Simple RAG inference pipeline."""

    llm_model: str = Field(
        default="gpt-3.5-turbo", description="LiteLLM model for generation"
    )
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1)
