"""Inference pipeline for Simple RAG."""

import logging
from typing import Any, cast

import litellm

from ragworkbench.api.inference import InferencePipeline
from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.api.ingest_artifact import IngestArtifact
from ragworkbench.boards.board_model import CacheMode
from ragworkbench.boards.board_registry import inference_pipeline
from ragworkbench.datasets_loader.data_models import RagBenchmarkEntry
from simple_rag.config import MilvusConfig, SimpleRagInferenceParams
from simple_rag.ingest_pipeline import SimpleRagIngestArtifact
from simple_rag.milvus_client import MilvusRetriever

logger = logging.getLogger(__name__)


@inference_pipeline(name="simple_rag", params_class=SimpleRagInferenceParams)
class SimpleRagInferencePipeline(InferencePipeline):
    """Simple RAG inference pipeline."""

    EMBEDDING_TIMEOUT = 30  # seconds

    def __init__(
        self,
        params: SimpleRagInferenceParams,
        cache_dir: str | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ) -> None:
        """Initialize the inference pipeline."""
        super().__init__(params, cache_dir, cache_mode)
        self._params: SimpleRagInferenceParams = params
        self.retriever: MilvusRetriever | None = None
        self.embedding_model: str | None = None
        self.collection_name: str | None = None

    def set_ingest_artifacts(self, ingest_artifacts: list[IngestArtifact]) -> None:
        """
        Configure pipeline from ingestion artifacts.

        Args:
            ingest_artifacts: List of ingest artifacts (expects single artifact)
        """
        if not ingest_artifacts:
            raise ValueError("No ingest artifacts provided")

        artifact = ingest_artifacts[0]
        if not isinstance(artifact, SimpleRagIngestArtifact):
            raise TypeError(
                f"Expected SimpleRagIngestArtifact, got {type(artifact).__name__}"
            )

        # Initialize Milvus retriever from artifact
        milvus_config = MilvusConfig(uri=artifact.milvus_uri)
        self.retriever = MilvusRetriever(
            config=milvus_config,
            collection_name=artifact.collection_name,
        )
        self.embedding_model = artifact.embedding_model
        self.collection_name = artifact.collection_name

        logger.info(
            f"Configured inference pipeline with collection '{artifact.collection_name}'"
        )

    def _get_additional_cache_params(self) -> dict[str, Any] | None:
        """
        Get additional parameters to include in the cache key.

        Returns collection_name to ensure cache isolation between different
        Milvus collections, as they may contain different document sets.

        Returns:
            Dictionary with collection_name if set, None otherwise
        """
        if self.collection_name is not None:
            return {"collection_name": self.collection_name}
        return None

    def _generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for query."""
        if self.embedding_model is None:
            raise RuntimeError(
                "Embedding model not set. Call set_ingest_artifacts first."
            )

        response = litellm.embedding(
            model=self.embedding_model,
            input=[query],
            api_key=self._params.tracking_api_key,
            timeout=self.EMBEDDING_TIMEOUT,
        )
        return response.data[0]["embedding"]

    def _format_prompt(self, question: str, contexts: list[str]) -> str:
        """Format prompt with retrieved contexts."""
        context_str = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])

        return f"""Context:
{context_str}

Question: {question}

Answer the question based on the provided context. If the context doesn't contain enough information, say so.

Answer:"""

    def process_no_cache(self, benchmark_entry: RagBenchmarkEntry) -> InferenceResult:
        """
        Process a benchmark entry without using cache.

        Args:
            benchmark_entry: The benchmark entry to process

        Returns:
            InferenceResult with answer and retrieved contexts
        """
        if self.retriever is None:
            raise RuntimeError(
                "Retriever not initialized. Call set_ingest_artifacts first."
            )

        question = benchmark_entry.question
        logger.info(f"Processing question: {question[:100]}...")

        # Generate query embedding
        query_embedding = self._generate_query_embedding(question)

        # Retrieve relevant chunks
        search_results = self.retriever.search(
            query_embedding=query_embedding, top_k=self._params.top_k
        )

        # Extract contexts and IDs
        contexts = [result["chunk_text"] for result in search_results]
        context_ids = [result["document_id"] for result in search_results]

        # Format prompt
        prompt = self._format_prompt(question, contexts)

        # Call LLM
        response = cast(
            dict[str, Any],
            litellm.completion(
                model=self._params.llm_model,
                messages=[{"role": "user", "content": prompt}],
                api_key=self._params.tracking_api_key,
            ),
        )

        choices = response.get("choices")
        if not choices:
            raise RuntimeError("LLM returned no choices")

        message = cast(dict[str, Any], choices[0].get("message", {}))
        answer = message.get("content")
        if not isinstance(answer, str) or not answer:
            raise RuntimeError("LLM returned an empty response")

        # Create inference result
        result = InferenceResult(
            question_id=benchmark_entry.question_id,
            question=question,
            ground_truth_answers=benchmark_entry.ground_truth_answers,
            ground_truths_context_ids=benchmark_entry.ground_truths_context_ids,
            is_answerable=benchmark_entry.is_answerable,
            additional_information=benchmark_entry.additional_information,
            answer=answer,
            context_ids=context_ids,
            contexts=contexts,
        )

        logger.info(f"Generated answer: {answer[:100]}...")
        return result
