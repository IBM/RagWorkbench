# Simple RAG Example Board

This board demonstrates benchmarking the Simple RAG pipeline with the AIT-QA dataset.

## Overview

The board configuration evaluates Simple RAG with:
- **Dataset**: AIT-QA (100 questions)
- **Embedding Models**:
  - OpenAI text-embedding-3-large
  - OpenAI text-embedding-3-small
- **LLM Models**:
  - GPT-4o
  - GPT-4o-mini
- **Chunking**: 512 tokens with peer merging
- **Retrieval**: Top-5 chunks

## Metrics

The board evaluates:
- **Answer Correctness**: LLMaaJ with Llama (Rits)
- **Retrieval Quality**: Match@K, MRR, MAP
- **Usage & Cost**: Token counts and costs per question
- **Tool Usage**: Number of tool calls

## Running the Evaluation

### Prerequisites

1. Install simple-rag:
```bash
cd RagWorkbench/simple-rag
pip install -e .
```

2. Set up environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export RAGBENCH_DATA_DIR="path/to/data"  # Optional, defaults to ./ragworkbench_data
```

### Run Evaluation

```bash
cd RagWorkbench/simple-rag
python evaluate.py
```

The script will:
1. Load the board configuration from `boards/simple_rag_example/board.yaml`
2. Download the AIT-QA dataset (if not cached)
3. Run ingestion for each embedding model configuration
4. Run inference for each LLM model configuration
5. Compute all metrics
6. Generate reports in `boards/simple_rag_example/output/`

## Output

Results are saved to:
- `boards/simple_rag_example/output/combined_results_exp_*.json` - Raw results
- `boards/simple_rag_example/output/report_*.html` - HTML report with charts

## Customization

Edit `board.yaml` to:
- Change the number of questions: `datasets[0].sampling.question_limit`
- Add/remove embedding models: `configurations[0].ingest.params.embedding_model`
- Add/remove LLM models: `configurations[0].inference.params.llm_model`
- Adjust chunking: `configurations[0].ingest.params.chunking_config`
- Modify retrieval: `configurations[0].inference.params.top_k`
- Add/remove metrics: `metrics` section
- Customize report screens: `report.screens` section

## Cache

The board uses caching to speed up repeated runs:
- **Docling Cache**: Cached document conversions in `cache/docling/`
- **Generation Cache**: Cached LLM responses in `cache/generation/`
- **Data Loader Cache**: Cached dataset downloads in `$RAGBENCH_DATA_DIR`

To refresh caches, set `experiment.cache: "refresh"` in `board.yaml`.
