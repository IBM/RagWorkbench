# Integration Tests

This directory contains integration tests that interact with external services and real data.

## Overview

Integration tests load real datasets from HuggingFace and verify data integrity. All tests use the shared `IntegrationTestHelpers` module to reduce code duplication and ensure consistent validation across all loaders.

## Test Files

### Existing Loaders (Refactored)
- **`test_clap_nq_integration.py`** - CLAP-NQ dataset tests
- **`test_miniwiki_integration.py`** - Mini Wikipedia dataset tests
- **`test_watsonx_integration.py`** - WatsonX DocsQA dataset tests

### New Loaders (Added)
- **`test_bioasq_integration.py`** - BioASQ biomedical QA dataset tests
- **`test_hotpotqa_integration.py`** - HotpotQA multi-hop reasoning dataset tests
- **`test_kramabench_integration.py`** - KramaBench dataset tests

## Helper Module

**`tests/datasets_loader/helpers/integration_test_helpers.py`**

Provides reusable validation methods:
- `assert_ground_truth_documents_exist()` - Validates corpus contains all benchmark references
- `assert_document_ids_unique()` - Ensures no duplicate document IDs
- `assert_question_ids_unique()` - Ensures no duplicate question IDs
- `assert_documents_have_content()` - Validates non-empty document content
- `assert_documents_have_metadata()` - Validates required metadata fields
- `assert_entries_have_answers()` - Validates non-empty answers
- `assert_entries_are_answerable()` - Validates answerable flag
- `assert_entries_have_ground_truth_contexts()` - Validates ground truth context IDs
- `assert_corpus_not_empty()` / `assert_benchmark_not_empty()` - Basic data checks

## Running Integration Tests

Integration tests are marked with `@pytest.mark.integration` and can be run separately:

```bash
# Run only integration tests
pytest tests/datasets_loader/integration/ -v -m integration

# Run all tests except integration tests (faster for development)
pytest tests/datasets_loader/ -v -m "not integration"

# Run all tests including integration tests
pytest tests/datasets_loader/ -v

# Run specific loader integration tests
pytest tests/datasets_loader/integration/test_bioasq_integration.py -v
pytest tests/datasets_loader/integration/test_hotpotqa_integration.py -v
```

## Requirements

Integration tests require:
- Internet connection (to download data from HuggingFace)
- The `datasets` package installed
- May take longer to run than unit tests

## Common Test Patterns

All integration tests follow these patterns:

### 1. Ground Truth Validation
```python
@pytest.mark.parametrize("split", ["train", "test"])
def test_ground_truth_documents_exist_in_corpus(self, split):
    loader = DataLoader(split=split)
    corpus = loader.get_corpus()
    benchmark = loader.get_benchmark()
    helpers.assert_ground_truth_documents_exist(corpus, benchmark, split)
```

### 2. Data Integrity Checks
```python
def test_document_ids_are_unique(self):
    loader = DataLoader(split="train")
    corpus = loader.get_corpus()
    helpers.assert_document_ids_unique(corpus)
```

### 3. Content Validation
```python
def test_documents_have_content(self):
    loader = DataLoader(split="train")
    corpus = loader.get_corpus()
    helpers.assert_documents_have_content(corpus, sample_size=20)
```

## Test Coverage Summary

| Loader | Tests | Ground Truth | Uniqueness | Content | Metadata | Answers |
|--------|-------|--------------|------------|---------|----------|---------|
| CLAP-NQ | 1 | ✅ (2 splits) | - | - | - | - |
| MiniWiki | 5 | - | ✅ | ✅ | - | ✅ |
| WatsonX | 4 | ✅ (3 splits) | - | ✅ | ✅ | ✅ |
| BioASQ | 5 | ✅ (2 splits) | ✅ | ✅ | - | ✅ |
| HotpotQA | 6 | ✅ (2 splits) | ✅ | ✅ | - | ✅ |
| KramaBench | 5 | ✅ | ✅ | ✅ | - | ✅ |

## Benefits of Refactoring

1. **DRY Principle** - Validation logic defined once, reused everywhere
2. **Consistency** - All loaders tested with identical standards
3. **Maintainability** - Fix bugs once, benefit everywhere
4. **Extensibility** - Adding new loader tests is trivial (70-75% less code)
5. **Readability** - Self-documenting helper method names
