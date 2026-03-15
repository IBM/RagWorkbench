"""
Sampling parameter pytest fixtures.

This module provides fixtures for creating DataSamplingParams instances
with various configurations for testing sampling behavior.
"""

import pytest

from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)


@pytest.fixture
def sample_data_sampling_params() -> DataSamplingParams:
    """
    Create a sample DataSamplingParams instance with default values.

    Returns:
        DataSamplingParams instance with default configuration.
    """
    return DataSamplingParams()


@pytest.fixture
def sample_data_sampling_params_with_limits() -> DataSamplingParams:
    """
    Create a DataSamplingParams instance with sampling limits.

    Returns:
        DataSamplingParams with question_limit=3, document_factor=2, seed=42.
    """
    return DataSamplingParams(question_limit=3, document_factor=2, seed=42)
