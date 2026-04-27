# Simple RAG Pipeline

A simple RAG (Retrieval-Augmented Generation) pipeline implementation for RagWorkbench that demonstrates document ingestion, vector storage with Milvus Lite, and LLM-based generation using LiteLLM.

## Features

- **Document Processing**: Uses Docling for PDF/DOCX conversion
- **Chunking**: Docling HybridChunker for intelligent text segmentation
- **Vector Storage**: Milvus Lite (embedded vector database) with COSINE similarity
- **Embeddings**: LiteLLM embedding API (supports multiple providers)
- **Generation**: LiteLLM completion API (supports multiple LLM providers)
- **Caching**: Integrated with RagWorkbench's GenerationCache

## Installation

```bash
cd RagWorkbench/simple-rag
pip install -e .
```

## Usage

```python
from ragworkbench import Experiment
from ragworkbench.datasets_loader import DataLoaderFactory
from ragworkbench.boards.board_model import ExperimentConfig
from simple_rag import (
    SimpleRagIngestPipeline,
    SimpleRagInferencePipeline,
    SimpleRagIngestParams,
    SimpleRagInferenceParams,
    MilvusConfig,
    ChunkingConfig,
)

# Configure ingestion
ingest_params = SimpleRagIngestParams(
    milvus_config=MilvusConfig(uri="./milvus.db"),  # File-based storage
    # Or use in-memory: MilvusConfig(uri=":memory:")
    chunking_config=ChunkingConfig(max_tokens=512),
    embedding_model="text-embedding-3-small",
)

# Configure inference
inference_params = SimpleRagInferenceParams(
    llm_model="gpt-3.5-turbo",
    top_k=5,
)

# Load dataset
data_loader = DataLoaderFactory.get_data_loader("ait-qa")

# Create pipelines
ingest_pipeline = SimpleRagIngestPipeline(ingest_params)
inference_pipeline = SimpleRagInferencePipeline(inference_params)

# Run experiment
experiment = Experiment(
    experiment_id="simple_rag_demo",
    data_loader=data_loader,
    ingest_pipeline=ingest_pipeline,
    inference_pipeline=inference_pipeline,
    eval_metrics=[],
    experiment_config=ExperimentConfig(),
)

result = experiment.run()
```

## Configuration

### MilvusConfig
- `uri`: Milvus Lite URI (default: "./milvus.db")
  - File-based: `"./milvus.db"` or any path
  - In-memory: `":memory:"`

### ChunkingConfig
- `tokenizer`: Tokenizer for chunking (default: "sentence-transformers/all-MiniLM-L6-v2")
- `max_tokens`: Maximum tokens per chunk (default: 512)
- `merge_peers`: Merge peer chunks when possible (default: True)

### SimpleRagIngestParams
- `milvus_config`: Milvus connection settings
- `chunking_config`: Document chunking settings
- `embedding_model`: LiteLLM embedding model (default: "text-embedding-3-small")

### SimpleRagInferenceParams
- `llm_model`: LiteLLM model for generation (default: "gpt-3.5-turbo")
- `top_k`: Number of chunks to retrieve (default: 5)

## Architecture

1. **Ingestion Phase**:
   - Documents are converted to text using Docling
   - Text is chunked using HybridChunker
   - Embeddings are generated via LiteLLM
   - Chunks and embeddings are stored in Milvus
   - Collection name is auto-generated from parameters hash

2. **Inference Phase**:
   - Query is embedded using the same embedding model
   - Top-k similar chunks are retrieved from Milvus
   - Retrieved contexts are formatted into a prompt
   - LLM generates answer based on contexts

## Benchmarking

### Quick Start with Example Board

Run the included example board to benchmark Simple RAG on AIT-QA:

```bash
cd RagWorkbench/simple-rag
python evaluate.py
```

This will:
- Load 100 questions from AIT-QA dataset
- Test 2 embedding models (text-embedding-3-large, text-embedding-3-small)
- Test 2 LLM models (gpt-4o, gpt-4o-mini)
- Generate comprehensive metrics and reports

See [`boards/simple_rag_example/README.md`](boards/simple_rag_example/README.md) for details.

### Custom Boards

Create your own board configuration in `boards/your_board_name/board.yaml` and run the evaluation script:

```bash
# Edit evaluate.py to point to your board
python evaluate.py
```

## Testing

```bash
cd RagWorkbench
pytest simple-rag/tests/
```

## Requirements

- Python >= 3.11
- Milvus Lite (automatically installed with pymilvus)
- API keys for LiteLLM providers (OpenAI, etc.)

## License

Apache-2.0
