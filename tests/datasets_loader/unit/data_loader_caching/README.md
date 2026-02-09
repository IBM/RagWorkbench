# DataLoaderCache Tests

This directory contains comprehensive unit tests for the `DataLoaderCache` class.

## Test Coverage

The test suite (`test_data_loader_cache.py`) provides comprehensive coverage of the `DataLoaderCache` class with 60+ test cases organized into 9 sections:

### 1. Initialization and Setup (6 tests)
- Cache directory creation
- Nested directory structure based on config hash
- YAML configuration file creation and preservation
- Path handling (string and Path objects)
- Counter initialization

### 2. Adding Items to Cache (3 tests)
- Adding RagCorpus and RagBenchmark objects
- Cache dictionary updates
- Deep copy creation to prevent mutation

### 3. Retrieving Items from Cache (6 tests)
- Retrieving cached items
- Handling empty cache
- Deep copy on retrieval
- Cache hit/miss tracking

### 4. RagCorpus Serialization and Deserialization (6 tests)
- JSON serialization of RagCorpus with binary streams
- Base64 encoding of document streams
- Stream position preservation
- Binary content handling
- Round-trip serialization/deserialization

### 5. RagBenchmark Serialization (4 tests)
- JSON serialization of RagBenchmark
- Reading corpus and benchmark from files
- Error handling for unexpected files

### 6. Cache Persistence and Loading (4 tests)
- Cache persistence across instances
- Loading existing cache files on initialization
- Class-level cache sharing

### 7. Hash Functions (9 tests)
- MD5 hash generation from bytes and strings
- Dictionary hashing with order independence
- Nested structure handling
- Non-serializable object handling

### 8. Cache File Path Management (3 tests)
- Parameter hash generation
- Cache file path formatting
- File path retrieval

### 9. Edge Cases and Error Handling (19 tests)
- Empty and complex configurations
- Unicode handling in configs and filenames
- Multiple caches with different configs
- Large corpus handling
- Binary document content
- Internal method testing (_add_with_key, _get_with_key)
- Class-level cache sharing

## Key Features Tested

1. **Serialization**: Proper serialization of RagCorpus (with binary streams) and RagBenchmark to JSON
2. **Deserialization**: Correct reconstruction of objects from cached JSON files
3. **Cache Management**: Hit/miss tracking, deep copying, persistence
4. **File System Operations**: Directory creation, file writing/reading, path handling
5. **Hash Functions**: Consistent and deterministic hash generation for cache keys
6. **Edge Cases**: Unicode, binary data, large datasets, empty configs

## Running the Tests

```bash
# Run all cache tests
pytest tests/datasets_loader/unit/data_loader_caching/test_data_loader_cache.py -v

# Run specific test
pytest tests/datasets_loader/unit/data_loader_caching/test_data_loader_cache.py::TestDataLoaderCache::test_initialization_creates_cache_directory -v

# Run with coverage
pytest tests/datasets_loader/unit/data_loader_caching/test_data_loader_cache.py --cov=ragbench.datasets_loader.data_loader_caching.data_loader_cache
```

## Dependencies

The tests use:
- `pytest` for test framework
- `pytest-fixtures` from `tests/datasets_loader/fixtures/` for sample data
- Standard library modules: `json`, `base64`, `pathlib`, `io`

## Notes

- Tests use `tmp_path` fixture for isolated file system operations
- All tests are independent and can run in any order
- Tests verify both functionality and data integrity
- Type safety is maintained throughout with proper assertions
