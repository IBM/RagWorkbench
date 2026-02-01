# Integration Tests

This directory contains integration tests that interact with external services and real data.

## BioASQ Integration Tests

The `test_bioasq_integration.py` file contains integration tests for the BioASQ data loader that:
- Load real data from HuggingFace dataset `enelpol/rag-mini-bioasq`
- Verify data integrity and consistency
- **Validate that all ground-truth documents referenced in benchmark entries exist in the corpus**

### Running Integration Tests

Integration tests are marked with `@pytest.mark.integration` and can be run separately:

```bash
# Run only integration tests
pytest tests/datasets_loader/integration/ -v -m integration

# Run all tests except integration tests (faster for development)
pytest tests/datasets_loader/ -v -m "not integration"

# Run all tests including integration tests
pytest tests/datasets_loader/ -v
```

### Requirements

Integration tests require:
- Internet connection (to download data from HuggingFace)
- The `datasets` package installed
- May take longer to run than unit tests

### Test Coverage

The integration tests verify:
1. **Ground-truth document validation** - All documents referenced in benchmark entries exist in corpus
2. **Split handling** - Tests for train, test, and combined splits
3. **Sampling preservation** - Ground-truth documents are preserved even with sampling
4. **Data consistency** - Overall integrity between corpus and benchmark data
