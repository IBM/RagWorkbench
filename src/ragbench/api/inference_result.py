from ragbench.datasets_loader.data_models import RagBenchmarkEntry


class InferenceResult(RagBenchmarkEntry):
    answer: str
    # TODO - Add
    # "context_ids",
    # "contexts",
