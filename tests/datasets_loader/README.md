# Dataset Tests Documentation

## Overview

This directory contains comprehensive pytest test suites for the `src/datasets` module. The tests are organized into a modular structure for better maintainability and scalability.

## Directory Structure

```
tests/datasets/
├── __init__.py
├── README.md                           # This file
├── conftest.py                         # Root fixture imports
│
├── fixtures/                           # Organized fixture modules
│   ├── __init__.py
│   ├── benchmark_fixtures.py          # Benchmark-related fixtures
│   ├── corpus_fixtures.py             # Corpus/document fixtures
│   └── sampling_fixtures.py           # Sampling parameter fixtures
│
├── helpers/                            # Test utilities and helpers
│   ├── __init__.py
│   └── mock_data_loader.py            # MockRagDataLoader for testing
│
├── unit/                               # Unit tests organized by component
│   ├── __init__.py
│   │
│   ├── data_models/                   # Data model tests
│   │   ├── __init__.py
│   │   ├── test_ground_truth_context_id.py    # 5 tests
│   │   ├── test_rag_benchmark_entry.py        # 5 tests
│   │   └── test_rag_benchmark.py              # 13 tests
│   │
│   ├── loaders/                       # Data loader tests
│   │   ├── __init__.py
│   │   ├── test_abstract_data_loader_initialization.py  # 6 tests
│   │   ├── test_abstract_data_loader_sampling.py        # 8 tests
│   │   ├── test_abstract_data_loader_methods.py         # 8 tests
│   │   └── test_bioasq_data_loader.py                   # 18 tests
│   │
│   └── utils/                         # Utility function tests
│       ├── __init__.py
│       └── test_datasets_utils.py     # 13 tests
│
└── integration/                        # Integration tests (future)
    └── __init__.py
```

## Test Organization

### Fixtures (`fixtures/`)

Fixtures are organized by concern into separate modules:

- **`benchmark_fixtures.py`**: Fixtures for creating benchmark entries, ground truth context IDs, and RagBenchmark instances
- **`corpus_fixtures.py`**: Fixtures for creating documents, document sets, and RagCorpus instances
- **`sampling_fixtures.py`**: Fixtures for creating DataSamplingParams with various configurations

All fixtures are automatically imported in the root `conftest.py` for pytest discovery.

### Helpers (`helpers/`)

Reusable test utilities:

- **`mock_data_loader.py`**: Contains `MockRagDataLoader`, a concrete implementation of `RagDataLoader` for testing without requiring actual dataset files

### Unit Tests (`unit/`)

Unit tests are organized by component type:

#### Data Models (`unit/data_models/`)

Tests for Pydantic data models:

- **`test_ground_truth_context_id.py`** (5 tests)
  - Valid creation with all/minimal fields
  - Immutability verification
  - Page number validation
  - Document ID validation

- **`test_rag_benchmark_entry.py`** (5 tests)
  - Valid creation with all/minimal fields
  - Immutability verification
  - Unanswerable question handling
  - Multiple ground truth contexts

- **`test_rag_benchmark.py`** (13 tests)
  - Minimum entry requirement validation
  - Question/ID retrieval with filtering
  - Benchmark entry filtering
  - Document ID extraction
  - Length method
  - Mixed answerable/unanswerable handling

#### Loaders (`unit/loaders/`)

Tests for data loader implementations:

- **`test_abstract_data_loader_initialization.py`** (6 tests)
  - Initialization with all parameters
  - Default sampling behavior
  - Split handling (train/test/None)
  - Corpus and benchmark creation

- **`test_abstract_data_loader_sampling.py`** (8 tests)
  - Question sampling with/without limits
  - Document sampling with factor-based selection
  - Ground truth document preservation
  - Seed-based reproducibility
  - Combined question + document sampling

- **`test_abstract_data_loader_methods.py`** (8 tests)
  - get_benchmark() and get_corpus() methods
  - Method consistency
  - End-to-end integration workflow
  - Edge cases (minimal dataset, no sampling, factor=0)

- **`test_bioasq_data_loader.py`** (18 tests)
  - Initialization with various parameters
  - Document loading and format validation
  - Benchmark entry loading
  - Split handling
  - Sampling functionality
  - Integration scenarios

#### Utils (`unit/utils/`)

Tests for utility functions:

- **`test_datasets_utils.py`** (13 tests)
  - Train/test split validation
  - No overlap between splits
  - Complete coverage (union equals whole)
  - Reproducibility with seeds
  - Correct split ratios
  - Edge cases (single entry, small datasets)

## Test Coverage Summary

| Component | Test Files | Test Methods | Key Features Tested |
|-----------|------------|--------------|---------------------|
| Data Models | 3 | 23 | Validation, immutability, filtering, document ID extraction |
| Loaders | 4 | 40 | Sampling logic, initialization, reproducibility, integration |
| Utils | 1 | 13 | Split validation, reproducibility, ratio handling |
| **Total** | **8** | **76** | **Comprehensive coverage of all components** |

## Running the Tests

### Prerequisites

Ensure you have Python 3.13+ and the dev dependencies installed:

```bash
pip install -e ".[dev]"
```

### Run All Dataset Tests

```bash
pytest tests/datasets_loader/ -v
```

### Run Tests by Category

```bash
# Data model tests only
pytest tests/datasets_loader/unit/data_models/ -v

# Loader tests only
pytest tests/datasets_loader/unit/loaders/ -v

# Utility tests only
pytest tests/datasets_loader/unit/utils/ -v
```

### Run Specific Test File

```bash
pytest tests/datasets_loader/unit/data_models/test_rag_benchmark.py -v
pytest tests/datasets_loader/unit/loaders/test_abstract_data_loader_sampling.py -v
pytest tests/datasets_loader/unit/loaders/test_bioasq_data_loader.py -v
```

### Run Specific Test Class

```bash
pytest tests/datasets_loader/unit/data_models/test_rag_benchmark.py::TestRagBenchmark -v
pytest tests/datasets_loader/unit/loaders/test_abstract_data_loader_sampling.py::TestRagDataLoaderSampling -v
```

### Run Specific Test Method

```bash
tests/datasets_loader/test_abstract_data_loader.py::TestRagDataLoaderEdgeCases::test_document_factor_zero PASSED

========================= 41 passed in X.XXs =========================
```

## Extending the Tests

To add more tests:

1. **Add fixtures** to `conftest.py` for new test data
2. **Create new test classes** following the naming convention `Test<ClassName>`
3. **Add test methods** with descriptive names starting with `test_`
4. **Use existing fixtures** to reduce code duplication
5. **Document** what each test verifies

## Notes

- Tests use `src.datasets` imports (adjust if package structure changes)
- Temporary directories are automatically cleaned up by pytest
- All tests are designed to be fast (<5 seconds total execution time)
- Tests verify both functionality and Pydantic validation behavior
