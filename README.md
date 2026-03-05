# RAGBench

A comprehensive benchmarking framework for Retrieval-Augmented Generation (RAG) systems.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

RAGBench is a powerful framework designed to evaluate and benchmark RAG systems across multiple datasets and metrics. It provides a unified interface for loading datasets, running experiments, and evaluating retrieval and generation performance.

## Features

- 🎯 **Multiple Dataset Support**: Built-in support for 17+ RAG benchmark datasets
- 📊 **Comprehensive Metrics**: Evaluate with various metrics including BERT Score, exact match, F1, and more
- 🔄 **Flexible Pipeline**: Modular design with separate ingest, inference, and evaluation stages
- 💾 **Smart Caching**: Built-in caching for data loading, generation, and evaluation results
- 🎨 **Interactive Exploration**: Dataset exploration UI powered by NiceGUI
- 📈 **Experiment Tracking**: Organize and track multiple experiments with ease

## Supported Datasets

RAGBench supports the following benchmark datasets:

- **AIT QA** - AI and technology question answering
- **BioASQ** - Biomedical question answering
- **CLAP-NQ** - Natural questions with context
- **DA-Code** - Code-related QA
- **DABStep** - Step-by-step reasoning
- **HotpotQA** - Multi-hop question answering
- **KramaBench** - Knowledge-intensive QA
- **Mini Wiki** - Wikipedia-based RAG
- **MLDR** - Multilingual long document retrieval
- **NarrativeQA** - Reading comprehension
- **OfficeQA** - Enterprise documentation QA
- **QASPER** - Question answering on scientific papers
- **SecQue** - Security-related questions
- **WatsonX DocsQA** - Enterprise documentation
- **Real-MM** - Multimodal datasets (financial/technical reports and slides)

## Installation

### Basic Installation

```bash
pip install ragbench
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ragbench.git
cd ragbench

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Optional Dependencies

```bash
# For documentation
pip install ragbench[docs]

# For examples with Docling
pip install ragbench[examples]

# Install all optional dependencies
pip install ragbench[all]
```

## Quick Start

### Basic Usage

```python
from ragbench import Experiment, DataLoaderFactory, DatasetName
from ragbench.api import IngestPipeline, InferencePipeline
from ragbench.eval import MetricDefinition

# Load a dataset
data_loader = DataLoaderFactory.create_loader(
    dataset_name=DatasetName.MINI_WIKI,
    split="test"
)

# Define your pipelines
ingest_pipeline = IngestPipeline(...)  # Your ingest implementation
inference_pipeline = InferencePipeline(...)  # Your inference implementation

# Define evaluation metrics
metrics = [
    MetricDefinition(metric_id="exact_match"),
    MetricDefinition(metric_id="f1"),
]

# Create and run experiment
experiment = Experiment(
    name="my_rag_experiment",
    data_loader=data_loader,
    ingest_pipeline=ingest_pipeline,
    inference_pipeline=inference_pipeline,
    eval_metrics=metrics,
)

# Run the complete pipeline
inference_results, evaluation_results = experiment.run()
```

### Dataset Exploration

```python
from ragbench.dataset_exploration import DatasetExplorer

# Launch interactive dataset explorer
explorer = DatasetExplorer()
explorer.run()
```

## Dataset Setup

### AIT QA Dataset

The AIT QA (Airline Industry Table QA) dataset requires special setup as it uses local PDF files. Follow these steps to prepare the dataset:

#### 1. Set Environment Variable

The AIT QA dataset requires the `RAGBENCH_DATA_DIR` environment variable to be set. This specifies where the dataset files will be stored.

**Option A: Using a .env file (Recommended)**

Create a `.env` file in your project root directory:

```bash
# .env file
RAGBENCH_DATA_DIR=/path/to/your/data/directory
```

The configuration will automatically load this file when needed.

**Option B: System Environment Variable**

```bash
# Set the environment variable (Linux/macOS)
export RAGBENCH_DATA_DIR=/path/to/your/data/directory

# Set the environment variable (Windows)
set RAGBENCH_DATA_DIR=C:\path\to\your\data\directory
```

To make this permanent, add it to your shell configuration file:

```bash
# For bash (~/.bashrc or ~/.bash_profile)
echo 'export RAGBENCH_DATA_DIR=/path/to/your/data/directory' >> ~/.bashrc
source ~/.bashrc

# For zsh (~/.zshrc)
echo 'export RAGBENCH_DATA_DIR=/path/to/your/data/directory' >> ~/.zshrc
source ~/.zshrc
```

#### 2. Download Dataset Files

Run the dataset creation script to download the required PDF files:

```bash
python -m ragbench.datasets_loader.ait_qa_data.create_ait_qa_dataset
```

This script will:
- Create the directory structure: `$RAGBENCH_DATA_DIR/ait_qa_pdf/documents/`
- Download 15 annual report PDFs from airlines (Alaska, American Airlines, Delta, Southwest, United) for fiscal years 2017-2019
- Skip files that already exist (safe to re-run)

The download may take several minutes depending on your internet connection.

#### 3. Use the Dataset

Once setup is complete, you can use the AIT QA dataset like any other dataset:

```python
from ragbench import DataLoaderFactory, DatasetName

# Load AIT QA dataset
loader = DataLoaderFactory.create_loader(
    dataset_name=DatasetName.AIT_QA,
    split="test"
)

# Access the data
corpus = loader.get_corpus()
benchmark = loader.get_benchmark()
```

**Note:** If you try to use the AIT QA dataset without setting `RAGBENCH_DATA_DIR`, you will receive an `EnvironmentError` with instructions on how to set it.

## Architecture

RAGBench follows a modular architecture with three main stages:

1. **Ingest Stage**: Process and index documents from the dataset
2. **Inference Stage**: Retrieve relevant documents and generate answers
3. **Evaluation Stage**: Compute metrics comparing generated answers to ground truth

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Ingest    │────▶│  Inference   │────▶│ Evaluation  │
│  Pipeline   │     │   Pipeline   │     │   Metrics   │
└─────────────┘     └──────────────┘     └─────────────┘
      │                    │                     │
      ▼                    ▼                     ▼
  Documents           Predictions            Scores
```

## Project Structure

```
ragbench/
├── src/ragbench/
│   ├── api/              # Core API interfaces
│   ├── caching/          # Caching implementations
│   ├── datasets_loader/  # Dataset loaders
│   ├── eval/             # Evaluation metrics
│   ├── boards/           # Result visualization
│   ├── dataset_exploration/  # Interactive dataset explorer
│   └── experiment.py     # Main experiment orchestration
├── tests/                # Test suite
└── pyproject.toml        # Project configuration
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests (skip integration tests)
pytest -m "not integration"
```

### Code Quality

```bash
# Format code
black src tests
isort src tests

# Lint code
ruff check src tests

# Type checking
mypy src
```

### Pre-commit Hooks

The project uses pre-commit hooks for code quality:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors

- **Matan Orbach** - [matano@il.ibm.com](mailto:matano@il.ibm.com)
- **Assaf Toledo** - [assaf.toledo@ibm.com](mailto:assaf.toledo@ibm.com)
- **Benjamin Sznajder** - [benjams@il.ibm.com](mailto:benjams@il.ibm.com)

## Citation

If you use RAGBench in your research, please cite:

```bibtex
@software{ragbench2024,
  title = {RAGBench: A Comprehensive Benchmarking Framework for RAG Systems},
  author = {Orbach, Matan and Toledo, Assaf and Sznajder, Benjamin},
  year = {2024},
  url = {https://github.com/yourusername/ragbench}
}
```

## Acknowledgments

- Built with [Unitxt](https://github.com/IBM/unitxt) for metric evaluation
- Uses [Docling](https://github.com/DS4SD/docling) for document processing
- Powered by [HuggingFace Datasets](https://huggingface.co/docs/datasets/)

## Support

For questions, issues, or feature requests, please:
- Open an issue on [GitHub Issues](https://github.com/yourusername/ragbench/issues)
- Check the [documentation](https://github.com/yourusername/ragbench#readme)
