"""Integration test for Simple RAG pipeline with AIT-QA dataset.

This test validates the complete RAG workflow:
- Loading a minimal subset of AIT-QA data
- Running the ingest pipeline to process documents
- Running the inference pipeline to answer questions
- Running an experiment with one RAG configuration
"""

import os
from pathlib import Path

import pytest
from simple_rag.config import (
    ChunkingConfig,
    MilvusConfig,
    SimpleRagInferenceParams,
    SimpleRagIngestParams,
)
from simple_rag.inference_pipeline import SimpleRagInferencePipeline
from simple_rag.ingest_pipeline import SimpleRagIngestPipeline

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.boards.board_model import CacheMode, ExperimentConfig
from ragworkbench.datasets_loader.ait_qa_data_loader import AITQaDataLoader
from ragworkbench.datasets_loader.data_models import DataSamplingParams
from ragworkbench.eval.metric_models import load_metric_definitions
from ragworkbench.experiment import Experiment


@pytest.fixture
def ragworkbench_data_dir():
    """Return the shared RagWorkbench data directory for integration tests."""
    return Path(__file__).resolve().parents[2] / "ragworkbench_data"


@pytest.fixture
def minimal_ait_qa_loader(ragworkbench_data_dir):
    """Create a data loader with minimal AIT-QA data (1 sample)."""
    sampling_params = DataSamplingParams(
        question_limit=1,  # Use only 1 question for fast testing
        document_factor=0,
    )
    os.environ["RAGBENCH_DATA_DIR"] = str(ragworkbench_data_dir)
    return AITQaDataLoader(
        split=DatasetSplit.TRAIN,
        sampling_params=sampling_params,
        cache_dir=ragworkbench_data_dir,
    )


@pytest.fixture
def ingest_params(tmp_path):
    """Create ingest parameters for the test."""
    # Use temporary file for Milvus Lite database
    db_path = tmp_path / "test_milvus.db"
    return SimpleRagIngestParams(
        milvus_config=MilvusConfig(uri=str(db_path)),
        chunking_config=ChunkingConfig(
            max_tokens=256,  # Smaller chunks for faster testing
        ),
        embedding_model="openai/all-minilm",
    )


@pytest.fixture
def inference_params():
    """Create inference parameters for the test."""
    return SimpleRagInferenceParams(
        llm_model="openai/llama3.2-1b",
        top_k=3,  # Retrieve fewer chunks for faster testing
    )


@pytest.fixture
def experiment_config():
    """Create experiment configuration."""
    return ExperimentConfig(
        cache=CacheMode.ON,  # Enable caching to test cache functionality
        usage_tracking=False,  # Disable usage tracking for test
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - required for embeddings and LLM calls",
)
def test_simple_rag_integration_with_ait_qa(
    minimal_ait_qa_loader,
    ingest_params,
    inference_params,
    experiment_config,
    tmp_path,
):
    """Test complete Simple RAG pipeline with minimal AIT-QA data.

    This integration test:
    1. Loads 1 question from AIT-QA dataset
    2. Runs ingest pipeline to process documents and create embeddings with Milvus Lite
    3. Runs inference pipeline to answer questions
    4. Validates the results structure

    Note: This test requires:
    - OPENAI_API_KEY environment variable for embeddings and LLM
    - Milvus Lite is automatically available (no server setup needed)
    """
    # Create pipelines
    ingest_pipeline = SimpleRagIngestPipeline(ingest_params)
    inference_pipeline = SimpleRagInferencePipeline(
        inference_params,
        cache_dir=str(tmp_path / "cache"),
        cache_mode=CacheMode.ON,
    )

    # Load a simple metric for evaluation
    metric_config = load_metric_definitions()
    # Use a simple retrieval metric that doesn't require LLM calls
    eval_metrics = [
        metric_config.get_metric_definition("unitxt.context_correctness.retrieval_at_k")
    ]

    # Create and run experiment
    experiment = Experiment(
        experiment_id="test_simple_rag_ait_qa",
        data_loader=minimal_ait_qa_loader,
        ingest_pipeline=ingest_pipeline,
        inference_pipeline=inference_pipeline,
        eval_metrics=eval_metrics,
        experiment_config=experiment_config,
        cache_dir=tmp_path,
    )

    # Run the experiment
    result = experiment.run()

    # Validate results
    assert result is not None
    assert result.inference_results is not None
    assert len(result.inference_results) == 1, "Should have 1 inference result"

    # Validate each inference result
    for inference_result in result.inference_results:
        assert inference_result.question_id is not None
        assert inference_result.question is not None
        assert inference_result.answer is not None
        assert len(inference_result.answer) > 0, "Answer should not be empty"
        assert inference_result.contexts is not None
        assert len(inference_result.contexts) > 0, "Should have retrieved contexts"
        assert inference_result.context_ids is not None
        assert len(inference_result.context_ids) > 0, "Should have context IDs"

    # Validate evaluation results
    assert result.evaluation_results is not None
    assert len(result.evaluation_results) > 0, "Should have evaluation results"

    # Check that the retrieval metric was computed
    metric_id = "unitxt.context_correctness.retrieval_at_k"
    assert metric_id in result.evaluation_results

    metric_result = result.evaluation_results[metric_id]
    assert "per_question" in metric_result
    assert "statistics" in metric_result
    assert len(metric_result["per_question"]) == 1

    # Verify cache was used (first run should have 1 miss, 0 hits)
    if inference_pipeline.generation_cache is not None:
        cache_stats = inference_pipeline.generation_cache.get_cache_stats()
        assert cache_stats["cache_miss"] == 1, "Should have 1 cache miss on first run"
        assert cache_stats["cache_hit"] == 0, "Should have 0 cache hits on first run"
        assert cache_stats["total_entries"] == 1, "Should have 1 cached entry"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - required for embeddings and LLM calls",
)
def test_simple_rag_ingest_and_inference_separately(
    minimal_ait_qa_loader,
    ingest_params,
    inference_params,
):
    """Test ingest and inference pipelines separately.

    This test validates:
    1. Ingest pipeline processes documents and returns artifacts
    2. Inference pipeline can use the artifacts to answer questions
    """
    # Create ingest pipeline and process documents
    ingest_pipeline = SimpleRagIngestPipeline(ingest_params)
    ingest_artifacts = ingest_pipeline.process(minimal_ait_qa_loader)

    # Validate ingest artifacts
    assert len(ingest_artifacts) == 1, "Should return one artifact"
    artifact = ingest_artifacts[0]
    # Type check and cast to SimpleRagIngestArtifact
    from simple_rag.ingest_pipeline import SimpleRagIngestArtifact

    assert isinstance(artifact, SimpleRagIngestArtifact)
    assert artifact.collection_name.startswith("simple_rag_")
    assert artifact.milvus_uri == ingest_params.milvus_config.uri
    assert artifact.embedding_model == ingest_params.embedding_model

    # Create inference pipeline and set artifacts
    inference_pipeline = SimpleRagInferencePipeline(inference_params)
    inference_pipeline.set_ingest_artifacts(ingest_artifacts)

    # Get benchmark entries
    benchmark = minimal_ait_qa_loader.get_benchmark()
    assert len(benchmark.entries) == 1, "Should have 1 benchmark entry"

    # Process first question
    first_entry = benchmark.entries[0]
    result = inference_pipeline.process_no_cache(first_entry)

    # Validate inference result
    assert result.question_id == first_entry.question_id
    assert result.question == first_entry.question


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - required for embeddings and LLM calls",
)
def test_inference_cache_hit_miss(
    minimal_ait_qa_loader,
    ingest_params,
    inference_params,
    tmp_path,
):
    """Test that inference cache correctly handles cache hits and misses.

    This test validates:
    1. First call results in cache miss and stores result
    2. Second call with same question results in cache hit
    3. Cache statistics are correctly tracked
    """
    # Create and configure pipelines
    ingest_pipeline = SimpleRagIngestPipeline(ingest_params)
    ingest_artifacts = ingest_pipeline.process(minimal_ait_qa_loader)

    cache_dir = tmp_path / "cache"
    inference_pipeline = SimpleRagInferencePipeline(
        inference_params,
        cache_dir=str(cache_dir),
        cache_mode=CacheMode.ON,
    )
    inference_pipeline.set_ingest_artifacts(ingest_artifacts)

    # Get benchmark entry
    benchmark = minimal_ait_qa_loader.get_benchmark()
    assert len(benchmark.benchmark_entries) == 1
    entry = benchmark.benchmark_entries[0]

    # First call - should be cache miss
    result1 = inference_pipeline.process(entry)
    assert result1.answer is not None
    assert len(result1.answer) > 0

    # Verify cache statistics after first call
    assert inference_pipeline.generation_cache is not None
    cache_stats = inference_pipeline.generation_cache.get_cache_stats()
    assert cache_stats["cache_miss"] == 1, "First call should be cache miss"
    assert cache_stats["cache_hit"] == 0, "First call should have no hits"
    assert cache_stats["total_entries"] == 1, "Should have 1 cached entry"

    # Second call with same entry - should be cache hit
    result2 = inference_pipeline.process(entry)
    assert result2.answer == result1.answer, "Cached result should match original"
    assert result2.contexts == result1.contexts, "Cached contexts should match"
    assert result2.context_ids == result1.context_ids, "Cached context IDs should match"

    # Verify cache statistics after second call
    cache_stats = inference_pipeline.generation_cache.get_cache_stats()
    assert cache_stats["cache_miss"] == 1, "Should still have only 1 miss"
    assert cache_stats["cache_hit"] == 1, "Second call should be cache hit"
    assert cache_stats["total_entries"] == 1, "Should still have 1 cached entry"

    # Verify cache file was created
    cache_files = list(cache_dir.glob("generation/**/*.json"))
    assert len(cache_files) == 1, "Should have exactly 1 cache file"

    # Verify cache config file was created
    cache_config_files = list(cache_dir.glob("generation/**/generation_cache.yaml"))
    assert len(cache_config_files) == 1, "Should have cache config file"
