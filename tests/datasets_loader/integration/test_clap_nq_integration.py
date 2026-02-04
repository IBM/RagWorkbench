"""
Integration tests for CLAP-NQ data loader.

This module contains integration tests that load real CLAP-NQ data from HuggingFace
and verify data integrity, particularly ensuring that all ground-truth documents
referenced in benchmark entries exist in the corpus.
"""

import pytest

from ragbench.datasets_loader.clap_nq_data_loader import ClapNqDataLoader
from tests.datasets_loader.helpers.integration_test_helpers import (
    IntegrationTestHelpers as helpers,
)


@pytest.mark.integration
class TestClapNqIntegration:
    """Integration tests for CLAP-NQ data loader with real data."""

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_ground_truth_documents_exist_in_corpus(self, split):
        """
        Test that all ground-truth doc_ids in benchmark exist in corpus.

        This test verifies data integrity by ensuring that every document
        referenced in the benchmark's ground truth context IDs actually
        exists in the loaded corpus.

        This is critical for RAG evaluation as missing ground-truth documents
        would make it impossible to properly evaluate retrieval performance.
        """
        # Load CLAP-NQ data
        loader = ClapNqDataLoader(split=split)

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify all ground-truth documents exist in corpus
        helpers.assert_ground_truth_documents_exist(corpus, benchmark, split)
