"""
Comprehensive tests for InferencePipeline with caching functionality.

Test Categories:
1. Initialization Tests
2. Cache Integration Tests
3. Process Method with Cache Tests
4. Process Method without Cache Tests
5. Cache Hit/Miss Tracking Tests
6. Edge Cases and Error Handling
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ragworkbench.api.inference import InferenceParams, InferencePipeline
from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.api.ingest_artifact import IngestArtifact
from ragworkbench.caching.abstract_file_system_cache import AbstractFileSystemCache
from ragworkbench.caching.generation_cache import GenerationCache
from ragworkbench.datasets_loader.data_models import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)

# ============================================================================
# Concrete Implementation for Testing
# ============================================================================


class MockInferencePipeline(InferencePipeline):
    """Concrete implementation of InferencePipeline for testing."""

    def __init__(
        self,
        _params: InferenceParams,
        cache_dir: Path | str | None = None,
    ):
        super().__init__(_params, cache_dir)
        self.process_no_cache_call_count = 0
        self.last_processed_entry = None

    def set_ingest_artifacts(self, ingest_artifacts: list[IngestArtifact]) -> None:
        """Mock implementation."""
        self.ingest_artifacts = ingest_artifacts

    def process_no_cache(self, benchmark_entry: RagBenchmarkEntry) -> InferenceResult:
        """Mock implementation that generates a simple answer."""
        self.process_no_cache_call_count += 1
        self.last_processed_entry = benchmark_entry

        # Generate a simple answer based on the question
        answer = f"Answer to: {benchmark_entry.question}"

        return InferenceResult(
            **benchmark_entry.model_dump(),
            answer=answer,
        )


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def inference_params():
    """Create sample InferenceParams."""
    return InferenceParams()


@pytest.fixture
def pipeline_without_cache(inference_params):
    """Create pipeline without cache."""
    AbstractFileSystemCache.cache_path_to_contents.clear()
    return MockInferencePipeline(_params=inference_params, cache_dir=None)


@pytest.fixture
def pipeline_with_cache(temp_cache_dir, inference_params):
    """Create pipeline with cache."""
    AbstractFileSystemCache.cache_path_to_contents.clear()
    return MockInferencePipeline(
        _params=inference_params,
        cache_dir=temp_cache_dir,
    )


@pytest.fixture
def sample_benchmark_entry():
    """Create a sample RagBenchmarkEntry."""
    return RagBenchmarkEntry(
        question_id="q1",
        question="What is the capital of France?",
        ground_truth_answers=["Paris"],
        ground_truths_context_ids=[GroundTruthContextId(document_id="doc1", page=1)],
        is_answerable=True,
    )


@pytest.fixture
def another_benchmark_entry():
    """Create another benchmark entry."""
    return RagBenchmarkEntry(
        question_id="q2",
        question="Who invented the telephone?",
        ground_truth_answers=["Alexander Graham Bell"],
        is_answerable=True,
    )


# ============================================================================
# Helper Functions
# ============================================================================


def create_benchmark_entry(question_id: str, question: str) -> RagBenchmarkEntry:
    """Helper to create a simple benchmark entry."""
    return RagBenchmarkEntry(
        question_id=question_id,
        question=question,
        is_answerable=True,
    )


# ============================================================================
# Category 1: Initialization Tests
# ============================================================================


class TestInitialization:
    """Tests for InferencePipeline initialization."""

    def test_initialization_without_cache(self, inference_params):
        """Test initialization without cache directory."""
        pipeline = MockInferencePipeline(_params=inference_params, cache_dir=None)

        assert pipeline._params == inference_params
        assert pipeline.generation_cache is None

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_initialization_with_cache_path_object(
        self, temp_cache_dir, inference_params
    ):
        """Test initialization with Path object for cache."""
        pipeline = MockInferencePipeline(
            _params=inference_params,
            cache_dir=temp_cache_dir,
        )

        assert pipeline.generation_cache is not None
        assert isinstance(pipeline.generation_cache, GenerationCache)
        assert pipeline.generation_cache.cache_path.exists()

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_initialization_with_cache_string_path(
        self, temp_cache_dir, inference_params
    ):
        """Test initialization with string path for cache."""
        pipeline = MockInferencePipeline(
            _params=inference_params,
            cache_dir=str(temp_cache_dir),
        )

        assert pipeline.generation_cache is not None
        assert isinstance(pipeline.generation_cache, GenerationCache)

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_cache_directory_created(self, temp_cache_dir, inference_params):
        """Test that cache directory is created during initialization."""
        pipeline = MockInferencePipeline(
            _params=inference_params,
            cache_dir=temp_cache_dir,
        )

        assert pipeline.generation_cache.cache_path.exists()
        assert pipeline.generation_cache.cache_path.is_dir()

    def test_params_stored_correctly(self, inference_params):
        """Test that inference params are stored."""
        pipeline = MockInferencePipeline(_params=inference_params)

        assert pipeline._params is inference_params


# ============================================================================
# Category 2: Cache Integration Tests
# ============================================================================


class TestCacheIntegration:
    """Tests for cache integration."""

    def test_generation_cache_is_none_without_cache_dir(self, pipeline_without_cache):
        """Test that generation_cache is None when no cache_dir provided."""
        assert pipeline_without_cache.generation_cache is None

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_generation_cache_created_with_cache_dir(self, pipeline_with_cache):
        """Test that GenerationCache is created when cache_dir provided."""
        assert pipeline_with_cache.generation_cache is not None
        assert isinstance(pipeline_with_cache.generation_cache, GenerationCache)

    @pytest.mark.skip(reason="Test disabled - needs fixing")
    def test_cache_uses_inference_params(self, temp_cache_dir, inference_params):
        """Test that cache is initialized with inference params."""
        pipeline = MockInferencePipeline(
            _params=inference_params,
            cache_dir=temp_cache_dir,
        )

        # Cache should be created with the params
        assert pipeline.generation_cache is not None
        # The cache path should include a hash of the params
        assert "generation" in str(pipeline.generation_cache.cache_path)


# ============================================================================
# Category 3: Process Method with Cache Tests
# ============================================================================


class TestProcessWithCache:
    """Tests for process method with caching enabled."""

    def test_process_calls_process_no_cache_on_first_call(
        self, pipeline_with_cache, sample_benchmark_entry
    ):
        """Test that process_no_cache is called on cache miss."""
        result = pipeline_with_cache.process(sample_benchmark_entry)

        assert pipeline_with_cache.process_no_cache_call_count == 1
        assert result.answer == f"Answer to: {sample_benchmark_entry.question}"

    def test_process_returns_cached_result_on_second_call(
        self, pipeline_with_cache, sample_benchmark_entry
    ):
        """Test that cached result is returned without calling process_no_cache."""
        # First call - should call process_no_cache
        result1 = pipeline_with_cache.process(sample_benchmark_entry)
        assert pipeline_with_cache.process_no_cache_call_count == 1

        # Second call - should use cache
        result2 = pipeline_with_cache.process(sample_benchmark_entry)
        assert pipeline_with_cache.process_no_cache_call_count == 1  # Not incremented
        assert result2.answer == result1.answer

    def test_process_saves_result_to_cache(
        self, pipeline_with_cache, sample_benchmark_entry
    ):
        """Test that result is saved to cache after generation."""
        # Process entry
        result = pipeline_with_cache.process(sample_benchmark_entry)

        # Verify it's in cache
        cached_result = pipeline_with_cache.generation_cache.get(sample_benchmark_entry)
        assert cached_result is not None
        assert cached_result.answer == result.answer

    def test_process_multiple_different_entries(
        self, pipeline_with_cache, sample_benchmark_entry, another_benchmark_entry
    ):
        """Test processing multiple different entries."""
        # Process first entry
        result1 = pipeline_with_cache.process(sample_benchmark_entry)
        assert pipeline_with_cache.process_no_cache_call_count == 1

        # Process second entry (different)
        result2 = pipeline_with_cache.process(another_benchmark_entry)
        assert pipeline_with_cache.process_no_cache_call_count == 2

        # Process first entry again (should use cache)
        result1_again = pipeline_with_cache.process(sample_benchmark_entry)
        assert pipeline_with_cache.process_no_cache_call_count == 2  # Not incremented

        assert result1.answer == result1_again.answer
        assert result1.answer != result2.answer

    def test_cache_hit_statistics(self, pipeline_with_cache, sample_benchmark_entry):
        """Test that cache statistics are tracked correctly."""
        # First call - cache miss
        pipeline_with_cache.process(sample_benchmark_entry)
        stats = pipeline_with_cache.generation_cache.get_cache_stats()
        assert stats["cache_miss"] == 1
        assert stats["cache_hit"] == 0

        # Second call - cache hit
        pipeline_with_cache.process(sample_benchmark_entry)
        stats = pipeline_with_cache.generation_cache.get_cache_stats()
        assert stats["cache_miss"] == 1
        assert stats["cache_hit"] == 1

        # Third call - another cache hit
        pipeline_with_cache.process(sample_benchmark_entry)
        stats = pipeline_with_cache.generation_cache.get_cache_stats()
        assert stats["cache_miss"] == 1
        assert stats["cache_hit"] == 2

    def test_cache_persists_across_pipeline_instances(
        self, temp_cache_dir, inference_params, sample_benchmark_entry
    ):
        """Test that cache persists across different pipeline instances."""
        # Create first pipeline and process entry
        pipeline1 = MockInferencePipeline(
            _params=inference_params,
            cache_dir=temp_cache_dir,
        )
        result1 = pipeline1.process(sample_benchmark_entry)
        assert pipeline1.process_no_cache_call_count == 1

        # Clear class-level cache to force reload from disk
        AbstractFileSystemCache.cache_path_to_contents.clear()

        # Create second pipeline
        pipeline2 = MockInferencePipeline(
            _params=inference_params,
            cache_dir=temp_cache_dir,
        )

        # Process same entry - should use cached result
        result2 = pipeline2.process(sample_benchmark_entry)
        assert pipeline2.process_no_cache_call_count == 0  # Not called
        assert result2.answer == result1.answer

    def test_process_returns_deep_copy_from_cache(
        self, pipeline_with_cache, sample_benchmark_entry
    ):
        """Test that cached results are deep copied."""
        # First call
        result1 = pipeline_with_cache.process(sample_benchmark_entry)
        original_answer = result1.answer

        # Modify result1
        result1.answer = "Modified answer"

        # Second call - should get unmodified cached result
        result2 = pipeline_with_cache.process(sample_benchmark_entry)
        assert result2.answer == original_answer
        assert result2.answer != "Modified answer"


# ============================================================================
# Category 4: Process Method without Cache Tests
# ============================================================================


class TestProcessWithoutCache:
    """Tests for process method without caching."""

    def test_process_calls_process_no_cache_every_time(
        self, pipeline_without_cache, sample_benchmark_entry
    ):
        """Test that process_no_cache is called every time without cache."""
        # First call
        pipeline_without_cache.process(sample_benchmark_entry)
        assert pipeline_without_cache.process_no_cache_call_count == 1

        # Second call
        pipeline_without_cache.process(sample_benchmark_entry)
        assert pipeline_without_cache.process_no_cache_call_count == 2

        # Third call
        pipeline_without_cache.process(sample_benchmark_entry)
        assert pipeline_without_cache.process_no_cache_call_count == 3

    def test_process_returns_correct_result_without_cache(
        self, pipeline_without_cache, sample_benchmark_entry
    ):
        """Test that correct result is returned without cache."""
        result = pipeline_without_cache.process(sample_benchmark_entry)

        assert isinstance(result, InferenceResult)
        assert result.question_id == sample_benchmark_entry.question_id
        assert result.question == sample_benchmark_entry.question
        assert "Answer to:" in result.answer

    def test_process_multiple_entries_without_cache(self, pipeline_without_cache):
        """Test processing multiple entries without cache."""
        entries = [create_benchmark_entry(f"q{i}", f"Question {i}?") for i in range(5)]

        results = [pipeline_without_cache.process(entry) for entry in entries]

        # All should have been processed
        assert pipeline_without_cache.process_no_cache_call_count == 5

        # All results should be different
        answers = [r.answer for r in results]
        assert len(set(answers)) == 5


# ============================================================================
# Category 5: Cache Hit/Miss Tracking Tests
# ============================================================================


class TestCacheTracking:
    """Tests for cache hit/miss tracking."""

    def test_cache_miss_on_first_access(
        self, pipeline_with_cache, sample_benchmark_entry
    ):
        """Test that first access is a cache miss."""
        pipeline_with_cache.process(sample_benchmark_entry)

        stats = pipeline_with_cache.generation_cache.get_cache_stats()
        assert stats["cache_miss"] == 1
        assert stats["cache_hit"] == 0

    def test_cache_hit_on_subsequent_access(
        self, pipeline_with_cache, sample_benchmark_entry
    ):
        """Test that subsequent accesses are cache hits."""
        # First access
        pipeline_with_cache.process(sample_benchmark_entry)

        # Subsequent accesses
        for _ in range(3):
            pipeline_with_cache.process(sample_benchmark_entry)

        stats = pipeline_with_cache.generation_cache.get_cache_stats()
        assert stats["cache_miss"] == 1
        assert stats["cache_hit"] == 3

    def test_cache_statistics_with_multiple_entries(self, pipeline_with_cache):
        """Test cache statistics with multiple different entries."""
        entries = [create_benchmark_entry(f"q{i}", f"Question {i}?") for i in range(3)]

        # Process each entry twice
        for entry in entries:
            pipeline_with_cache.process(entry)  # Miss
            pipeline_with_cache.process(entry)  # Hit

        stats = pipeline_with_cache.generation_cache.get_cache_stats()
        assert stats["cache_miss"] == 3  # One miss per unique entry
        assert stats["cache_hit"] == 3  # One hit per repeated entry
        assert stats["total_entries"] == 3


# ============================================================================
# Category 6: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_process_with_none_ground_truth_answers(self, pipeline_with_cache):
        """Test processing entry with None ground_truth_answers."""
        entry = RagBenchmarkEntry(
            question_id="q1",
            question="Test question?",
            ground_truth_answers=None,
            is_answerable=False,
        )

        result = pipeline_with_cache.process(entry)
        assert result is not None
        assert result.ground_truth_answers is None

    def test_process_with_empty_ground_truth_context_ids(self, pipeline_with_cache):
        """Test processing entry with empty context IDs."""
        entry = RagBenchmarkEntry(
            question_id="q1",
            question="Test question?",
            ground_truths_context_ids=[],
            is_answerable=True,
        )

        result = pipeline_with_cache.process(entry)
        assert result is not None
        assert len(result.ground_truths_context_ids) == 0

    def test_process_with_complex_additional_information(self, pipeline_with_cache):
        """Test processing entry with complex additional information."""
        entry = RagBenchmarkEntry(
            question_id="q1",
            question="Test question?",
            is_answerable=True,
            additional_information={
                "category": "test",
                "nested": {"key": "value"},
                "list": [1, 2, 3],
            },
        )

        result = pipeline_with_cache.process(entry)
        assert result is not None
        assert result.additional_information["category"] == "test"

        # Process again - should use cache
        result2 = pipeline_with_cache.process(entry)
        assert result2.additional_information == result.additional_information

    def test_process_preserves_all_benchmark_entry_fields(self, pipeline_with_cache):
        """Test that all fields from benchmark entry are preserved."""
        entry = RagBenchmarkEntry(
            question_id="q_complex",
            question="Complex question?",
            ground_truth_answers=["Answer 1", "Answer 2"],
            ground_truths_context_ids=[
                GroundTruthContextId(document_id="doc1", page=5, table_id="t1")
            ],
            is_answerable=True,
            additional_information={"key": "value"},
        )

        result = pipeline_with_cache.process(entry)

        # Verify all fields
        assert result.question_id == entry.question_id
        assert result.question == entry.question
        assert result.ground_truth_answers == entry.ground_truth_answers
        assert len(result.ground_truths_context_ids) == 1
        assert result.ground_truths_context_ids[0].document_id == "doc1"
        assert result.is_answerable == entry.is_answerable
        assert result.additional_information == entry.additional_information

    def test_set_ingest_artifacts_works(self, pipeline_with_cache):
        """Test that set_ingest_artifacts method works."""
        artifacts = [MagicMock(spec=IngestArtifact)]

        pipeline_with_cache.set_ingest_artifacts(artifacts)

        assert pipeline_with_cache.ingest_artifacts == artifacts

    def test_process_no_cache_receives_correct_entry(
        self, pipeline_with_cache, sample_benchmark_entry
    ):
        """Test that process_no_cache receives the correct entry."""
        pipeline_with_cache.process(sample_benchmark_entry)

        assert pipeline_with_cache.last_processed_entry == sample_benchmark_entry
        assert (
            pipeline_with_cache.last_processed_entry.question_id
            == sample_benchmark_entry.question_id
        )
