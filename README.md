# RAGWorkbench

A comprehensive benchmarking framework for Retrieval-Augmented Generation (RAG) systems.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Overview

RAGWorkbench is a powerful Python framework designed to evaluate and benchmark RAG systems across multiple datasets and metrics. It provides a unified interface for loading diverse RAG benchmarks, running inference pipelines, and computing comprehensive evaluation metrics.

### Key Features

- 🎯 **Multiple Benchmark Datasets**: Support for 18+ RAG benchmark datasets including AIT-QA, BioASQ, HotpotQA, NarrativeQA, QASPER, and more
- 📊 **Comprehensive Metrics**: Built-in evaluation metrics for context correctness (Recall@K, MRR, MAP) and answer correctness (BERT Score, Sentence-BERT, LLM-as-a-Judge)
- 🔄 **Flexible Pipeline**: Modular architecture supporting custom ingest and inference pipelines
- 💾 **Smart Caching**: File-system based caching for data loading, generation, and evaluation results
- 🌐 **Interactive Explorer**: Web-based dataset exploration tool with advanced filtering capabilities
- 🧪 **Experiment Management**: End-to-end experiment orchestration from data loading to evaluation

## Installation

### Requirements

- Python 3.11 or higher

### Basic Installation

```bash
pip install .
```

### Development Installation

```bash
git clone https://github.com/IBM/RagWorkbench.git
cd RagWorkbench
pip install -e ".[dev]"
```

### Environment Configuration

Some evaluation metrics require environment variables to be configured. See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed instructions on setting up credentials for:
- watsonx.ai LLM-as-a-Judge metrics
- Azure OpenAI metrics

### Optional Dependencies

```bash
# For documentation
pip install .[docs]

# For examples
pip install .[examples]

# Install all optional dependencies
pip install .[all]
```

## Quick Start

### Basic Usage

```python
from ragbench import DataLoaderFactory, DatasetName, Experiment
from ragbench.api.inference import InferencePipeline, InferenceParams
from ragbench.api.ingest import IngestPipeline
from ragbench.eval import MetricDefinition

# Load a dataset
data_loader = DataLoaderFactory.create_data_loader(DatasetName.HOTPOT_QA)

# Define your custom pipelines
class MyIngestPipeline(IngestPipeline):
    def process(self, data_loader):
        # Your ingestion logic
        pass

class MyInferencePipeline(InferencePipeline):
    def __init__(self, params: InferenceParams, cache_dir=None):
        super().__init__(params, cache_dir)

    def set_ingest_artifacts(self, ingest_artifacts):
        # Set up your retrieval system
        pass

    def process_no_cache(self, benchmark_entry):
        # Your inference logic
        pass

# Define evaluation metrics
metrics = [
    MetricDefinition.from_yaml_key("unitxt.context_correctness.retrieval_at_k"),
    MetricDefinition.from_yaml_key("unitxt.answer_correctness.bert_score_recall"),
]

# Create and run experiment
experiment = Experiment(
    name="my_rag_experiment",
    data_loader=data_loader,
    ingest_pipeline=MyIngestPipeline(),
    inference_pipeline=MyInferencePipeline(InferenceParams()),
    eval_metrics=metrics,
    cache_dir="./cache"
)

results, evaluation = experiment.run()
```

## Supported Datasets

RAGWorkbench supports 18+ benchmark datasets across various domains:

| Dataset | Domain | Retrieval Hops | Modalities |
|---------|--------|----------------|------------|
| AIT-QA | Financial | Single | TEXT, TABLE |
| BioASQ | Biomedical | Single | TEXT |
| CLAP-NQ | Wikipedia | Single | TEXT |
| DA-Code | Code | Single | TEXT |
| DABStep | Code | Multi | TEXT |
| HotpotQA | Wikipedia | Multi | TEXT |
| KramaBench | Wikipedia | Single | TEXT |
| Mini-Wiki | Wikipedia | Single | TEXT |
| MLDR | Multilingual | Single | TEXT |
| NarrativeQA | Literature | Single | TEXT |
| OfficeQA | Technical Docs | Single | TEXT |
| QASPER | Scientific Papers | Single | TEXT |
| SecQue | Policies | Single | TEXT |
| WatsonX DocsQA | Technical Docs | Single | TEXT |
| RealMM (4 variants) | Financial/Technical | Single | TEXT, TABLE, IMAGE |

### Loading Datasets

```python
from ragbench import DataLoaderFactory, DatasetName

# Load a specific dataset
loader = DataLoaderFactory.create_data_loader(DatasetName.BIOASQ)

# Get the benchmark and corpus
benchmark = loader.get_benchmark()
corpus = loader.get_corpus()

# Access benchmark entries
for entry in benchmark.get_benchmark_entries():
    print(f"Question: {entry.question}")
    print(f"Ground truth answers: {entry.ground_truth_answers}")
```

## Evaluation Metrics

RAGWorkbench provides comprehensive evaluation metrics through integration with Unitxt:

### Context Correctness Metrics

- **Retrieval@K**: Measures retrieval accuracy at different cutoffs (K=1, 3, 5, 10, 20, 40)
- **MRR (Mean Reciprocal Rank)**: Evaluates the rank of the first relevant document
- **MAP (Mean Average Precision)**: Measures precision across all relevant documents

### Answer Correctness Metrics

- **BERT Score Recall**: Semantic similarity using BERT embeddings
- **Sentence-BERT**: Sentence-level semantic similarity
- **LLM-as-a-Judge**: Uses LLMs (Llama, GPT-4) to evaluate answer quality

### Using Metrics

```python
from ragbench.eval import MetricDefinition

# Load metrics from YAML definitions
metric = MetricDefinition.from_yaml_key("unitxt.context_correctness.retrieval_at_k")

# Or create custom metrics
custom_metric = MetricDefinition(
    metric_id="custom.metric",
    metric_params={"param": "value"},
    metric_fields=["field1", "field2"],
    vendor="unitxt"
)
```

## Caching System

RagWorkbench includes a sophisticated caching system to speed up experiments:

```python
from pathlib import Path

# Enable caching for all components
cache_dir = Path("./cache")

experiment = Experiment(
    name="cached_experiment",
    data_loader=data_loader,
    ingest_pipeline=ingest_pipeline,
    inference_pipeline=MyInferencePipeline(params, cache_dir=cache_dir),
    eval_metrics=metrics,
    cache_dir=cache_dir  # Enables evaluator caching
)
```

The caching system supports:
- **Data Loader Cache**: Caches loaded datasets
- **Generation Cache**: Caches inference results
- **Evaluator Cache**: Caches evaluation results

## Cost Tracking

RAGWorkbench supports optional cost tracking for experiments using LiteLLM proxy. This feature allows you to monitor API usage and costs during experiment runs.

### Enabling Cost Tracking

```python
from ragbench import Experiment

experiment = Experiment(
    name="cost_tracked_experiment",
    data_loader=data_loader,
    ingest_pipeline=ingest_pipeline,
    inference_pipeline=inference_pipeline,
    eval_metrics=metrics,
    track_costs=True,  # Enable cost tracking
    litellm_proxy_url="http://localhost:4000"  # LiteLLM proxy URL
)

# Run experiment - returns results, evaluation, and cost data
results, evaluation, cost_data = experiment.run()

# Access cost information
print(f"Total cost: ${cost_data['total_cost']:.4f}")
print(f"Total tokens: {cost_data['total_tokens']}")
print(f"Models used: {cost_data['models_used']}")
```

### How It Works

1. **API Key Generation**: When cost tracking is enabled, a unique tracking API key is generated for each experiment run
2. **Key Injection**: The tracking key is passed to the OpenRAG service through the ingest pipeline settings
3. **Usage Tracking**: All LLM API calls during the experiment use this tracking key
4. **Cost Retrieval**: After the experiment completes, usage data is retrieved from the LiteLLM proxy

### Configuration

Set the LiteLLM proxy URL in your environment:

```bash
# .env file
LITELLM_PROXY_URL=http://localhost:4000
```

Or pass it directly to the Experiment constructor as shown above.

### Requirements

- LiteLLM proxy server must be running and accessible
- The proxy should be configured to track usage by API key
- See [LiteLLM documentation](https://docs.litellm.ai/) for proxy setup

## Dataset Explorer

RagWorkbench includes an interactive web-based dataset explorer:

```bash
python -m ragbench.dataset_exploration.dataset_explorer
```

Then open your browser to `http://localhost:8080`

### Explorer Features

- 📋 Browse all available datasets in a sortable table
- 🔍 Search datasets by name or description
- 🎨 Filter by domain, retrieval hops, modalities, and more
- 📊 View detailed dataset statistics and metadata
- 📋 Copy dataset names with one click


### Core Components

- **DataLoader**: Loads and manages benchmark datasets
- **IngestPipeline**: Processes and indexes documents
- **InferencePipeline**: Runs retrieval and generation
- **Evaluator**: Computes evaluation metrics
- **Experiment**: Orchestrates the complete workflow

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest tests/datasets_loader/unit

# Run only integration tests
pytest -m integration
```

### Code Quality

```bash
# Format code
black src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Contributing

We welcome contributions! Please see our contributing guidelines for more details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Releases

RAGWorkbench follows [Semantic Versioning](https://semver.org/) and uses automated releases via GitHub Actions.

### Installation

Install the latest stable release from PyPI:

```bash
pip install ragworkbench
```

Install a specific version:

```bash
pip install ragworkbench==0.1.0
```

### Release Process

For maintainers preparing a new release:

1. **Prepare the release**:
   ```bash
   ./scripts/prepare_release.sh 0.2.0
   ```

2. **Create and push the tag**:
   ```bash
   git tag -a v0.2.0 -m "Release version 0.2.0"
   git push origin v0.2.0
   ```

3. **Monitor the automated workflow** at [GitHub Actions](https://github.com/IBM/RagWorkbench/actions)

The release workflow will automatically:
- Build the package
- Publish to PyPI
- Create a GitHub release with release notes

For detailed release instructions, see [RELEASE.md](RELEASE.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- **Matan Orbach** - [matano@il.ibm.com](mailto:matano@il.ibm.com)
- **Assaf Toledo** - [assaf.toledo@ibm.com](mailto:assaf.toledo@ibm.com)
- **Benjamin Sznajder** - [benjams@il.ibm.com](mailto:benjams@il.ibm.com)
- **Odellia Boni** - [odelliab@il.ibm.com](mailto:odelliab@il.ibm.com)

## Acknowledgments

- Built with [Unitxt](https://github.com/IBM/unitxt) for evaluation metrics
- Uses [NiceGUI](https://nicegui.io/) for the dataset explorer
- Integrates with [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

## Support

For questions, issues, or feature requests, please:
- Open an issue on [GitHub Issues](https://github.com/IBM/RagWorkbench/issues)

---
