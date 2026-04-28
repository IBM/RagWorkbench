"""Tests for inference pipeline."""

from unittest.mock import MagicMock, patch

import pytest
from simple_rag.config import SimpleRagInferenceParams
from simple_rag.inference_pipeline import SimpleRagInferencePipeline
from simple_rag.ingest_pipeline import SimpleRagIngestArtifact

from ragworkbench.datasets_loader.data_models import RagBenchmarkEntry


@pytest.fixture
def inference_params():
    """Fixture for inference parameters."""
    return SimpleRagInferenceParams(llm_model="gpt-3.5-turbo", top_k=3)


@pytest.fixture
def ingest_artifact():
    """Fixture for ingest artifact."""
    return SimpleRagIngestArtifact(
        collection_name="test_collection",
        milvus_uri="http://localhost:19530",
        embedding_model="text-embedding-3-small",
    )


@pytest.fixture
def benchmark_entry():
    """Fixture for benchmark entry."""
    return RagBenchmarkEntry(
        question_id="q1",
        question="What is the capital of France?",
        ground_truth_answers=["Paris"],
        ground_truths_context_ids=[],
        is_answerable=True,
    )


class TestSimpleRagInferencePipeline:
    """Tests for SimpleRagInferencePipeline."""

    def test_initialization(self, inference_params):
        """Test pipeline initialization."""
        pipeline = SimpleRagInferencePipeline(inference_params)
        assert pipeline._params == inference_params
        assert pipeline.retriever is None
        assert pipeline.embedding_model is None

    @patch("simple_rag.inference_pipeline.MilvusRetriever")
    def test_set_ingest_artifacts(
        self, mock_retriever_class, inference_params, ingest_artifact
    ):
        """Test setting ingest artifacts."""
        mock_retriever = MagicMock()
        mock_retriever_class.return_value = mock_retriever

        pipeline = SimpleRagInferencePipeline(inference_params)
        pipeline.set_ingest_artifacts([ingest_artifact])

        assert pipeline.retriever is not None
        assert pipeline.embedding_model == "text-embedding-3-small"
        mock_retriever_class.assert_called_once()

    def test_set_ingest_artifacts_empty_list(self, inference_params):
        """Test error when no artifacts provided."""
        pipeline = SimpleRagInferencePipeline(inference_params)

        with pytest.raises(ValueError, match="No ingest artifacts provided"):
            pipeline.set_ingest_artifacts([])

    def test_set_ingest_artifacts_wrong_type(self, inference_params):
        """Test error when wrong artifact type provided."""
        pipeline = SimpleRagInferencePipeline(inference_params)
        wrong_artifact = MagicMock()

        with pytest.raises(TypeError, match="Expected SimpleRagIngestArtifact"):
            pipeline.set_ingest_artifacts([wrong_artifact])

    @patch("simple_rag.inference_pipeline.litellm.embedding")
    @patch("simple_rag.inference_pipeline.MilvusRetriever")
    def test_generate_query_embedding(
        self, mock_retriever_class, mock_embedding, inference_params, ingest_artifact
    ):
        """Test query embedding generation."""
        mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1] * 384}])

        pipeline = SimpleRagInferencePipeline(inference_params)
        pipeline.set_ingest_artifacts([ingest_artifact])

        embedding = pipeline._generate_query_embedding("test query")

        assert len(embedding) == 384
        mock_embedding.assert_called_once()

    def test_generate_query_embedding_not_initialized(self, inference_params):
        """Test error when embedding model not set."""
        pipeline = SimpleRagInferencePipeline(inference_params)

        with pytest.raises(RuntimeError, match="Embedding model not set"):
            pipeline._generate_query_embedding("test query")

    def test_format_prompt(self, inference_params):
        """Test prompt formatting."""
        pipeline = SimpleRagInferencePipeline(inference_params)

        contexts = ["Context 1", "Context 2"]
        prompt = pipeline._format_prompt("What is X?", contexts)

        assert "Context 1" in prompt
        assert "Context 2" in prompt
        assert "What is X?" in prompt
        assert "Answer:" in prompt

    @patch("simple_rag.inference_pipeline.litellm.completion")
    @patch("simple_rag.inference_pipeline.litellm.embedding")
    @patch("simple_rag.inference_pipeline.MilvusRetriever")
    def test_process_no_cache(
        self,
        mock_retriever_class,
        mock_embedding,
        mock_completion,
        inference_params,
        ingest_artifact,
        benchmark_entry,
    ):
        """Test full inference process."""
        # Setup mocks
        mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1] * 384}])

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [
            {
                "chunk_id": "doc1_0",
                "document_id": "doc1",
                "chunk_text": "Paris is the capital of France.",
                "chunk_index": 0,
                "distance": 0.95,
            }
        ]
        mock_retriever_class.return_value = mock_retriever

        mock_message = MagicMock()
        mock_message.get.return_value = "Paris"
        mock_choice = MagicMock()
        mock_choice.get.return_value = mock_message
        mock_response = {"choices": [mock_choice]}
        mock_completion.return_value = mock_response

        # Run process
        pipeline = SimpleRagInferencePipeline(inference_params)
        pipeline.set_ingest_artifacts([ingest_artifact])
        result = pipeline.process_no_cache(benchmark_entry)

        # Verify
        assert result.question_id == "q1"
        assert result.question == "What is the capital of France?"
        assert result.answer == "Paris"
        assert len(result.contexts) == 1
        assert "Paris is the capital" in result.contexts[0]
        assert result.context_ids == ["doc1"]

        mock_retriever.search.assert_called_once()
        mock_completion.assert_called_once()

    @patch("simple_rag.inference_pipeline.MilvusRetriever")
    def test_process_no_cache_not_initialized(
        self, mock_retriever_class, inference_params, benchmark_entry
    ):
        """Test error when retriever not initialized."""
        pipeline = SimpleRagInferencePipeline(inference_params)

        with pytest.raises(RuntimeError, match="Retriever not initialized"):
            pipeline.process_no_cache(benchmark_entry)
