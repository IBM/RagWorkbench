# Simple RAG Tests

This directory contains tests for the Simple RAG pipeline implementation.

## Test Structure

- `test_config.py` - Unit tests for configuration models
- `test_milvus_client.py` - Unit tests for Milvus client
- `test_ingest_pipeline.py` - Unit tests for ingestion pipeline
- `test_inference_pipeline.py` - Unit tests for inference pipeline
- `test_integration_simple_rag.py` - Integration tests with real AIT-QA dataset

## Running Tests

### Unit Tests

Run all unit tests (fast, no external dependencies):

```bash
cd RagWorkbench/simple-rag
uv run pytest tests/ -m "not integration" -v
```

### Integration Tests

Integration tests require:
1. **OpenAI API Key** - Set `OPENAI_API_KEY` environment variable
2. **Milvus Lite** - Automatically available (no server setup needed)

Run integration tests:

```bash
cd RagWorkbench/simple-rag
export OPENAI_API_KEY="your-api-key"
uv run pytest tests/test_integration_simple_rag.py -v
```

### Run All Tests

```bash
cd RagWorkbench/simple-rag
uv run pytest tests/ -v
```

## Integration Test Details

### `test_simple_rag_integration_with_ait_qa`

This test validates the complete RAG workflow:
1. Loads 2 questions from the AIT-QA dataset
2. Runs ingest pipeline to process documents and create embeddings
3. Runs inference pipeline to answer questions using an Experiment
4. Validates results structure and evaluation metrics

### `test_simple_rag_ingest_and_inference_separately`

This test validates the pipelines independently:
1. Runs ingest pipeline and validates artifacts
2. Runs inference pipeline using the artifacts
3. Validates inference results for a single question

## Test Configuration

Integration tests use minimal data for fast execution:
- **Questions**: 2 samples from AIT-QA dataset
- **Chunk size**: 256 tokens (smaller than default)
- **Top-k retrieval**: 3 chunks (fewer than default)
- **Caching**: Disabled for integration tests

## Skipping Integration Tests

Integration tests are marked with `@pytest.mark.integration` and will be skipped if:
- `OPENAI_API_KEY` is not set

To skip integration tests explicitly:

```bash
uv run pytest tests/ -m "not integration" -v
