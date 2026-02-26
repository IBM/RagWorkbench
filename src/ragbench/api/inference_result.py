from ragbench.datasets_loader.data_models import RagBenchmarkEntry


class InferenceResult(RagBenchmarkEntry):
    answer: str
    context_ids: list[str] | None = None
    contexts: list[str] | None = None
