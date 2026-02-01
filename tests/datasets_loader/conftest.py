"""
Root conftest for datasets_loader tests.

This module imports and exposes all fixtures from the fixtures subpackage,
making them available to all test modules via pytest's fixture discovery.
"""

# Import all fixtures to make them available to pytest
from tests.datasets_loader.fixtures.benchmark_fixtures import (
    large_benchmark_entries,
    sample_benchmark_entries,
    sample_ground_truth_context_ids,
    sample_rag_benchmark,
)
from tests.datasets_loader.fixtures.corpus_fixtures import (
    large_document_set,
    sample_document_objects,
    sample_rag_corpus,
    temp_export_dir,
)
from tests.datasets_loader.fixtures.sampling_fixtures import (
    sample_data_sampling_params,
    sample_data_sampling_params_with_limits,
)

# Expose fixtures for pytest discovery
__all__ = [
    # Benchmark fixtures
    "sample_ground_truth_context_ids",
    "sample_benchmark_entries",
    "sample_rag_benchmark",
    "large_benchmark_entries",
    # Corpus fixtures
    "sample_document_objects",
    "sample_rag_corpus",
    "large_document_set",
    "temp_export_dir",
    # Sampling fixtures
    "sample_data_sampling_params",
    "sample_data_sampling_params_with_limits",
]
