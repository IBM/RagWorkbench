# Dataset Tests Documentation

## Overview

This directory contains comprehensive pytest test suites for the `src/datasets` module, focusing on the most critical components:

- **RagBenchmark** data models (GroundTruthContextId, RagBenchmarkEntry, RagBenchmark)
- **RagDataLoader** abstract base class with sampling logic

## Test Files Created

### 1. `conftest.py`
Shared pytest fixtures providing reusable test data:
- `sample_document_objects` - 5 sample DocumentObject instances
- `sample_ground_truth_context_ids` - 5 GroundTruthContextId instances
- `sample_benchmark_entries` - 6 RagBenchmarkEntry instances (4 answerable, 2 unanswerable)
- `sample_rag_benchmark` - Complete RagBenchmark instance
- `sample_rag_corpus` - Complete RagCorpus instance
- `large_document_set` - 20 documents for sampling tests
- `large_benchmark_entries` - 15 entries for sampling tests
- `temp_export_dir` - Temporary directory for file operations

### 2. `test_rag_benchmark.py`
Comprehensive tests for RAG benchmark data models (17 test methods):

#### TestGroundTruthContextId (5 tests)
- Valid creation with all/minimal fields
- Immutability verification (frozen fields)
- Page number validation (≥1)
- Document ID validation (required, non-empty)

#### TestRagBenchmarkEntry (5 tests)
- Valid creation with all/minimal fields
- Immutability verification
- Unanswerable question handling
- Multiple ground truth contexts

#### TestRagBenchmark (7 tests)
- Minimum entry requirement validation
- Question/ID retrieval with filtering
- Benchmark entry filtering (answerable/all)
- Document ID extraction (static method)
- Length method
- Mixed answerable/unanswerable handling
- Empty ground truth contexts

### 3. `test_abstract_data_loader.py`
Comprehensive tests for RagDataLoader (24 test methods):

#### MockRagDataLoader
- Concrete implementation for testing abstract base class
- Configurable test data generation

#### TestRagDataLoaderSampling (8 tests)
- Question sampling with/without limits
- Question limit exceeding available questions
- Document sampling with factor-based selection
- Ground truth document preservation
- Seed-based reproducibility
- Different seeds produce different results
- Combined question + document sampling

#### TestRagDataLoaderInitialization (6 tests)
- Initialization with all parameters
- Default sampling (no sampling)
- Split handling (train/test/None)
- Corpus and benchmark creation

#### TestRagDataLoaderMethods (5 tests)
- get_benchmark() returns correct instance
- get_corpus() returns correct instance
- Method consistency (same instance returned)
- End-to-end integration workflow

#### TestRagDataLoaderEdgeCases (3 tests)
- Minimal dataset (1 doc, 1 question)
- No document sampling (factor=None)
- Document factor=0 (only ground truth docs)

## Test Coverage Summary

| Module | Test Classes | Test Methods | Key Features Tested |
|--------|--------------|--------------|---------------------|
| rag_benchmark.py | 3 | 17 | Validation, immutability, filtering, document ID extraction |
| abstract_data_loader.py | 4 | 24 | Sampling logic, initialization, reproducibility, integration |
| **Total** | **7** | **41** | **Comprehensive coverage of critical components** |

## Running the Tests

### Prerequisites
Ensure you have Python 3.13+ and the dev dependencies installed:

```bash
pip install -e ".[dev]"
```

### Run All Dataset Tests
```bash
pytest tests/datasets/ -v
```

### Run Specific Test File
```bash
pytest tests/datasets/test_rag_benchmark.py -v
pytest tests/datasets/test_abstract_data_loader.py -v
```

### Run Specific Test Class
```bash
pytest tests/datasets/test_rag_benchmark.py::TestRagBenchmark -v
pytest tests/datasets/test_abstract_data_loader.py::TestRagDataLoaderSampling -v
```

### Run Specific Test Method
```bash
pytest tests/datasets/test_rag_benchmark.py::TestRagBenchmark::test_get_questions_answerable_only -v
```

### Run with Coverage Report
```bash
pytest tests/datasets/ --cov=src/datasets --cov-report=html --cov-report=term-missing
```

### Run Tests in Parallel (faster)
```bash
pytest tests/datasets/ -n auto
```

## Test Design Principles

### 1. **Comprehensive Coverage**
- Tests cover happy paths, edge cases, and error conditions
- Validation logic thoroughly tested
- Both positive and negative test cases included

### 2. **Isolation & Independence**
- Each test is independent and can run in any order
- Fixtures provide clean test data for each test
- No shared mutable state between tests

### 3. **Reproducibility**
- Seed-based random sampling ensures deterministic results
- Tests verify reproducibility explicitly

### 4. **Clear Documentation**
- Each test has a descriptive docstring
- Test names clearly indicate what is being tested
- Assertions include meaningful messages

### 5. **Realistic Scenarios**
- MockRagDataLoader simulates real data loader behavior
- Test data mimics actual use cases
- Integration tests verify end-to-end workflows

## Key Testing Patterns Used

### Fixtures for Reusability
```python
@pytest.fixture
def sample_benchmark_entries():
    # Provides reusable test data
    return [...]
```

### Parametrized Tests (can be added)
```python
@pytest.mark.parametrize("page,expected", [(1, True), (0, False), (-1, False)])
def test_page_validation(page, expected):
    # Test multiple scenarios efficiently
    pass
```

### Mock Implementation Pattern
```python
class MockRagDataLoader(RagDataLoader):
    # Concrete implementation for testing abstract class
    def _get_documents(self):
        return [...]
```

### Immutability Testing
```python
with pytest.raises(ValidationError):
    frozen_object.field = "new_value"
```

## Expected Test Results

When all tests pass, you should see:
```
tests/datasets/test_rag_benchmark.py::TestGroundTruthContextId::test_valid_creation_with_all_fields PASSED
tests/datasets/test_rag_benchmark.py::TestGroundTruthContextId::test_valid_creation_minimal_fields PASSED
...
tests/datasets/test_abstract_data_loader.py::TestRagDataLoaderEdgeCases::test_document_factor_zero PASSED

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
