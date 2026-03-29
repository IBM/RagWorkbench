"""
Tests for cache mode functionality (on/off/refresh).
"""

from pathlib import Path

import pytest

from ragworkbench.api.inference import InferenceParams, InferencePipeline
from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.api.ingest_artifact import IngestArtifact
from ragworkbench.boards.board_model import CacheMode
from ragworkbench.caching.abstract_file_system_cache import AbstractFileSystemCache
from ragworkbench.caching.generation_cache import GenerationCache
from ragworkbench.datasets_loader.data_models import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)


class MockInferencePipeline(InferencePipeline):
    """Mock implementation for testing."""

    def __init__(
        self,
        params: InferenceParams,
        cache_dir: Path | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ):
        super().__init__(params, cache_dir, cache_mode)
        self.process_count = 0

    def set_ingest_artifacts(self, ingest_artifacts: list[IngestArtifact]) -> None:
        pass

    def process_no_cache(self, benchmark_entry: RagBenchmarkEntry) -> InferenceResult:
        self.process_count += 1
        return InferenceResult(
            **benchmark_entry.model_dump(),
            answer=f"Answer {self.process_count}",
        )


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def inference_params():
    """Create sample inference parameters."""
    return InferenceParams()


@pytest.fixture
def sample_benchmark_entry():
    """Create sample benchmark entry."""
    return RagBenchmarkEntry(
        question_id="q1",
        question="What is the capital of France?",
        ground_truth_answers=["Paris"],
        ground_truths_context_ids=[GroundTruthContextId(document_id="doc1", page=1)],
        is_answerable=True,
    )


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear class-level cache before each test."""
    AbstractFileSystemCache.cache_path_to_contents.clear()
    yield
    AbstractFileSystemCache.cache_path_to_contents.clear()


class TestCacheModeON:
    """Tests for cache mode ON (default behavior)."""

    def test_cache_on_reads_and_writes(
        self, temp_cache_dir, inference_params, sample_benchmark_entry
    ):
        """Test that cache mode ON reads from and writes to cache."""
        pipeline = MockInferencePipeline(inference_params, temp_cache_dir, CacheMode.ON)

        # First call - should miss cache and process
        result1 = pipeline.process(sample_benchmark_entry)
        assert pipeline.process_count == 1
        assert result1.answer == "Answer 1"

        # Second call - should hit cache
        result2 = pipeline.process(sample_benchmark_entry)
        assert pipeline.process_count == 1  # No additional processing
        assert result2.answer == "Answer 1"  # Same cached result

    def test_cache_on_is_default(
        self, temp_cache_dir, inference_params, sample_benchmark_entry
    ):
        """Test that cache mode ON is the default."""
        pipeline = MockInferencePipeline(inference_params, temp_cache_dir)
        assert pipeline._cache_mode == CacheMode.ON

        # Verify it behaves like ON mode
        pipeline.process(sample_benchmark_entry)
        pipeline.process(sample_benchmark_entry)
        assert pipeline.process_count == 1


class TestCacheModeOFF:
    """Tests for cache mode OFF."""

    def test_cache_off_no_cache_created(
        self, temp_cache_dir, inference_params, sample_benchmark_entry
    ):
        """Test that cache mode OFF does not create cache."""
        pipeline = MockInferencePipeline(
            inference_params, temp_cache_dir, CacheMode.OFF
        )

        # Process entry
        pipeline.process(sample_benchmark_entry)

        # Cache should not be created
        assert pipeline.generation_cache is None

    def test_cache_off_always_processes(
        self, temp_cache_dir, inference_params, sample_benchmark_entry
    ):
        """Test that cache mode OFF always processes without caching."""
        pipeline = MockInferencePipeline(
            inference_params, temp_cache_dir, CacheMode.OFF
        )

        # First call
        result1 = pipeline.process(sample_benchmark_entry)
        assert pipeline.process_count == 1
        assert result1.answer == "Answer 1"

        # Second call - should process again (no cache)
        result2 = pipeline.process(sample_benchmark_entry)
        assert pipeline.process_count == 2
        assert result2.answer == "Answer 2"

        # Third call - should process again
        result3 = pipeline.process(sample_benchmark_entry)
        assert pipeline.process_count == 3
        assert result3.answer == "Answer 3"


class TestCacheModeREFRESH:
    """Tests for cache mode REFRESH."""

    def test_cache_refresh_uses_in_memory_cache(
        self, temp_cache_dir, inference_params, sample_benchmark_entry
    ):
        """Test that cache mode REFRESH uses in-memory cache but skips disk loading."""
        pipeline = MockInferencePipeline(
            inference_params, temp_cache_dir, CacheMode.REFRESH
        )

        # First call - should process and write to in-memory cache
        result1 = pipeline.process(sample_benchmark_entry)
        assert pipeline.process_count == 1
        assert result1.answer == "Answer 1"

        # Second call - should read from in-memory cache (not process again)
        result2 = pipeline.process(sample_benchmark_entry)
        assert pipeline.process_count == 1  # No additional processing
        assert result2.answer == "Answer 1"  # Same cached result from in-memory cache

    def test_cache_refresh_writes_to_cache(
        self, temp_cache_dir, inference_params, sample_benchmark_entry
    ):
        """Test that cache mode REFRESH writes to cache."""
        pipeline = MockInferencePipeline(
            inference_params, temp_cache_dir, CacheMode.REFRESH
        )

        # Process entry
        pipeline.process(sample_benchmark_entry)

        # Cache should be created and contain the entry
        assert pipeline.generation_cache is not None
        cache_stats = pipeline.generation_cache.get_cache_stats()
        assert cache_stats["total_entries"] == 1

    def test_cache_refresh_refreshes_existing(
        self, temp_cache_dir, inference_params, sample_benchmark_entry
    ):
        """Test that cache mode REFRESH refreshes existing cache entries."""
        # First, create cache with ON mode
        pipeline_on = MockInferencePipeline(
            inference_params, temp_cache_dir, CacheMode.ON
        )
        result1 = pipeline_on.process(sample_benchmark_entry)
        assert result1.answer == "Answer 1"

        # Clear class-level cache to force reload
        AbstractFileSystemCache.cache_path_to_contents.clear()

        # Now use REFRESH mode
        pipeline_refresh = MockInferencePipeline(
            inference_params, temp_cache_dir, CacheMode.REFRESH
        )
        result2 = pipeline_refresh.process(sample_benchmark_entry)
        assert (
            result2.answer == "Answer 1"
        )  # New processing (process_count starts at 0)

        # Verify the cache was refreshed
        assert pipeline_refresh.generation_cache is not None
        cache_stats = pipeline_refresh.generation_cache.get_cache_stats()
        assert cache_stats["total_entries"] == 1


class TestGenerationCacheWithModes:
    """Tests for GenerationCache with different modes."""

    def test_generation_cache_rejects_off_mode(self, temp_cache_dir, inference_params):
        """Test that GenerationCache raises error when created with OFF mode."""
        with pytest.raises(
            ValueError, match="should not be created when cache_mode is OFF"
        ):
            GenerationCache(
                cache_dir=temp_cache_dir,
                inference_params=inference_params,
                cache_mode=CacheMode.OFF,
            )

    def test_generation_cache_on_mode_returns_cached(
        self, temp_cache_dir, inference_params, sample_benchmark_entry
    ):
        """Test that GenerationCache with ON mode returns cached values."""
        cache = GenerationCache(
            cache_dir=temp_cache_dir,
            inference_params=inference_params,
            cache_mode=CacheMode.ON,
        )

        result = InferenceResult(
            question_id="q1",
            question="Test question",
            ground_truth_answers=["Test"],
            ground_truths_context_ids=[
                GroundTruthContextId(document_id="doc1", page=1)
            ],
            is_answerable=True,
            answer="Cached answer",
        )

        # Add to cache
        cache.add(sample_benchmark_entry, result)

        # Get from cache
        cached_result = cache.get(sample_benchmark_entry)
        assert cached_result is not None
        assert cached_result.answer == "Cached answer"

    def test_generation_cache_refresh_mode_uses_in_memory_cache(
        self, temp_cache_dir, inference_params, sample_benchmark_entry
    ):
        """Test that GenerationCache with REFRESH mode uses in-memory cache but skips disk loading."""
        cache = GenerationCache(
            cache_dir=temp_cache_dir,
            inference_params=inference_params,
            cache_mode=CacheMode.REFRESH,
        )

        result = InferenceResult(
            question_id="q1",
            question="Test question",
            ground_truth_answers=["Test"],
            ground_truths_context_ids=[
                GroundTruthContextId(document_id="doc1", page=1)
            ],
            is_answerable=True,
            answer="Cached answer",
        )

        # Add to in-memory cache
        cache.add(sample_benchmark_entry, result)

        # Get from in-memory cache - should return the cached value in REFRESH mode
        cached_result = cache.get(sample_benchmark_entry)
        assert cached_result is not None
        assert cached_result.answer == "Cached answer"
