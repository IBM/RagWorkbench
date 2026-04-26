# Simple RAG Pipeline Design

## Overview

This document outlines the design of a simple RAG (Retrieval-Augmented Generation) pipeline that follows the RagWorkbench framework patterns. The pipeline includes document ingestion with Docling, chunking, Milvus vector storage, retrieval, and LLM-based generation using LiteLLM.

## Architecture

The pipeline follows the standard RagWorkbench architecture with two main phases:

1. **Ingestion Phase**: Process documents → Chunk → Store in Milvus
2. **Inference Phase**: Retrieve relevant chunks → Generate answer with LLM

## Directory Structure

```
RagWorkbench/simple-rag/
├── __init__.py
├── DESIGN.md                          # This file
├── README.md                          # Usage instructions
├── config.py                          # Configuration models
├── ingest_pipeline.py                 # Ingestion implementation
├── inference_pipeline.py              # Inference implementation
├── milvus_client.py                   # Milvus vector store wrapper
└── pyproject.toml                     # Package configuration and dependencies
```

## Component Design

### 1. Configuration (`config.py`)

**Purpose**: Define Pydantic models for pipeline configuration.

**Classes**:
- `MilvusConfig`: Milvus connection settings
  - `host: str = "localhost"`
  - `port: int = 19530`
  - Note: Collection name will be auto-generated from ingestion parameters hash

- `ChunkingConfig`: Document chunking parameters (using Docling HybridChunker)
  - `tokenizer: str = "sentence-transformers/all-MiniLM-L6-v2"` - Tokenizer for chunking
  - `max_tokens: int = 512` - Maximum tokens per chunk
  - `merge_peers: bool = True` - Merge peer chunks when possible

- `SimpleRagIngestParams(IngestParams)`: Extends base IngestParams
  - `milvus_config: MilvusConfig`
  - `chunking_config: ChunkingConfig`
  - `embedding_model: str = "text-embedding-3-small"` - LiteLLM embedding model

- `SimpleRagInferenceParams(InferenceParams)`: Extends base InferenceParams
  - `llm_model: str = "gpt-3.5-turbo"` - LiteLLM model for generation
  - `top_k: int = 5` - Number of chunks to retrieve
  - Note: Milvus config and embedding model come from IngestArtifact

### 2. Milvus Client (`milvus_client.py`)

**Purpose**: Wrapper for Milvus vector database operations.

**Classes**:
- `MilvusVectorStore`: Manages Milvus collection and operations
  - `__init__(config: MilvusConfig, collection_name: str, dimension: int)`
  - `create_collection() -> None`: Initialize collection with schema (dimension determined by embedding model)
  - `insert_embeddings(chunks: list[dict], embeddings: list[list[float]]) -> list[str]`: Store vectors
  - `search(query_embedding: list[float], top_k: int) -> list[dict]`: Retrieve similar chunks (uses COSINE metric)
  - `delete_collection() -> None`: Clean up collection
  - `get_collection_stats() -> dict`: Return collection statistics

**Schema**:
```python
{
    "chunk_id": "VARCHAR(255)",      # Primary key
    "document_id": "VARCHAR(255)",   # Source document
    "chunk_text": "VARCHAR(65535)",  # Actual text content
    "chunk_index": "INT64",          # Position in document
    "embedding": "FLOAT_VECTOR",     # Vector representation
    "metadata": "JSON"               # Additional info
}
```

### 3. Ingest Pipeline (`ingest_pipeline.py`)

**Purpose**: Implement document ingestion following RagWorkbench patterns.

**Classes**:
- `SimpleRagIngestArtifact(IngestArtifact)`: Stores ingestion results
  - `collection_name: str` - Auto-generated from params hash
  - `milvus_host: str` - Milvus connection host
  - `milvus_port: int` - Milvus connection port
  - `embedding_model: str` - Model used for embeddings
  - `dimension: int` - Embedding dimension

- `SimpleRagIngestPipeline(IngestPipeline)`: Main ingestion logic
  - `__init__(params: SimpleRagIngestParams)`
  - `process(data_loader: RagDataLoader) -> list[IngestArtifact]`: Execute ingestion

**Process Flow**:
1. Generate collection name from params hash
2. Get documents from data_loader
3. For each document:
   - Convert to Docling Document using DocumentConverter
   - Chunk using Docling HybridChunker
   - Generate embeddings via LiteLLM embedding API
   - Store in Milvus with metadata
4. Return IngestArtifact with connection details

### 4. Inference Pipeline (`inference_pipeline.py`)

**Purpose**: Implement retrieval and generation following RagWorkbench patterns.

**Classes**:
- `SimpleRagInferencePipeline(InferencePipeline)`: Main inference logic
  - `__init__(params: SimpleRagInferenceParams, cache_dir, cache_mode)`
  - `set_ingest_artifacts(artifacts: list[IngestArtifact]) -> None`: Configure from ingestion
  - `process_no_cache(entry: RagBenchmarkEntry) -> InferenceResult`: Execute RAG

**Process Flow**:
1. Receive question from benchmark entry
2. Generate query embedding
3. Retrieve top-k chunks from Milvus
4. Format prompt with retrieved context
5. Call LLM via LiteLLM
6. Return InferenceResult with answer and context

**Prompt Template**:
```
Context:
{retrieved_chunks}

Question: {question}

Answer the question based on the provided context. If the context doesn't contain enough information, say so.

Answer:
```


## Dependencies

### Core Dependencies (from RagWorkbench)
- `pydantic`: Configuration models
- `docling-core`: Document processing
- `ragworkbench`: Base framework

### Additional Dependencies (pyproject.toml)
```toml
[project]
name = "simple-rag"
version = "0.1.0"
description = "Simple RAG pipeline implementation for RagWorkbench"
requires-python = ">=3.11"
dependencies = [
    "ragworkbench>=0.1.0",
    "pymilvus>=2.3.0",
    "litellm>=1.0.0",
    "docling>=1.0.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Integration with RagWorkbench

### Follows Framework Patterns

1. **Extends Base Classes**:
   - `IngestPipeline` → `SimpleRagIngestPipeline`
   - `InferencePipeline` → `SimpleRagInferencePipeline`
   - `IngestParams` → `SimpleRagIngestParams`
   - `InferenceParams` → `SimpleRagInferenceParams`

2. **Uses Framework Data Models**:
   - `DocumentObject`: For document handling
   - `RagBenchmarkEntry`: For questions
   - `InferenceResult`: For answers
   - `IngestArtifact`: For ingestion metadata

3. **Compatible with Experiment**:
   - Works with `Experiment.run()`
   - Supports caching via `GenerationCache`
   - Integrates with evaluation metrics

4. **Supports Framework Features**:
   - Cost tracking via `tracking_api_key`
   - Caching with `cache_dir` and `cache_mode`
   - Works with any `RagDataLoader`

## Implementation Notes

### Docling Integration
- Use `docling.document_converter.DocumentConverter` for PDF/DOCX processing
- Extract text from `DocumentObject.stream` (BytesIO)
- Handle multiple document formats (PDF, DOCX, TXT)

### Embedding Generation
- Use `litellm.embedding()` for generating embeddings
- Supports multiple embedding providers (OpenAI, Cohere, etc.)
- Batch process for efficiency
- Use COSINE similarity metric in Milvus

### LiteLLM Integration
- Use `litellm.completion()` for model calls
- Support multiple providers (OpenAI, Anthropic, etc.)
- Handle API keys via environment variables
- Track token usage for cost analysis

### Error Handling
- Graceful degradation if Milvus unavailable
- Retry logic for LLM API calls
- Validation of document formats
- Clear error messages for configuration issues

## Testing Strategy

### Unit Tests
- `test_milvus_client.py`: Vector store operations (mocked)
- `test_config.py`: Configuration validation

### Integration Tests
- `test_ingest_pipeline.py`: End-to-end ingestion
- `test_inference_pipeline.py`: End-to-end inference
- `test_simple_rag_experiment.py`: Full experiment run

### Test Data
- Use `ait-qa` dataset from RagWorkbench

## References

- RagWorkbench API: `src/ragworkbench/api/`
- Data Models: `src/ragworkbench/datasets_loader/data_models/`
- Experiment: `src/ragworkbench/experiment.py`
- Milvus Docs: https://milvus.io/docs
- LiteLLM Docs: https://docs.litellm.ai/
- Docling Docs: https://github.com/DS4SD/docling
