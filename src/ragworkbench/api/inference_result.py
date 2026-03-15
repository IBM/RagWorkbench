from typing import Any

from ragworkbench.datasets_loader.data_models import RagBenchmarkEntry

# Type alias for trajectory data structure
# Each trajectory entry represents a search query and its results
Trajectory = list[dict[str, Any]]


class InferenceResult(RagBenchmarkEntry):
    answer: str
    context_ids: list[str] | None = None
    contexts: list[str] | None = None
    trajectory: Trajectory | None = None
