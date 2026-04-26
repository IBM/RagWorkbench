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
    """Create a data loader with minimal AIT-QA data (2 samples)."""
    sampling_params = DataSamplingParams(
        question_limit=1,  # Use only 2 questions for fast testing
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
        llm_model="llama3.2-1b",
        top_k=3,  # Retrieve fewer chunks for faster testing
    )


@pytest.fixture
def experiment_config():
    """Create experiment configuration."""
    return ExperimentConfig(
        cache=CacheMode.OFF,  # Disable caching for integration test
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
    1. Loads 2 questions from AIT-QA dataset
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
        cache_dir=str(tmp_path),
        cache_mode=CacheMode.OFF,
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
    assert len(result.inference_results) == 2, "Should have 2 inference results"

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
    assert len(metric_result["per_question"]) == 2


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
    assert len(benchmark.entries) == 2, "Should have 2 benchmark entries"

    # Process first question
    first_entry = benchmark.entries[0]
    result = inference_pipeline.process_no_cache(first_entry)

    # Validate inference result
    assert result.question_id == first_entry.question_id
    assert result.question == first_entry.question
    assert result.answer is not None
    assert len(result.answer) > 0
    assert result.contexts is not None
    assert result.context_ids is not None
    assert len(result.contexts) > 0
    assert len(result.context_ids) > 0
    assert len(result.contexts) == len(result.context_ids)
