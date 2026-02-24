from ragbench.api.inference import InferencePipeline
from ragbench.api.inference_result import InferenceResult
from ragbench.api.ingest import IngestPipeline
from ragbench.datasets_loader import RagDataLoader
from ragbench.datasets_loader.data_models import RagBenchmark
from ragbench.eval import MetricDefinition


class Experiment:
    def __init__(
        self,
        name: str,
        data_loader: RagDataLoader,
        ingest_pipeline: IngestPipeline,
        inference_pipeline: InferencePipeline,
        eval_metrics: list[MetricDefinition],
    ):
        self.name = name
        self.data_loader = data_loader
        self.ingest_pipeline = ingest_pipeline
        self.inference_pipeline = inference_pipeline

        self.metric_definitions: list[MetricDefinition] = eval_metrics

    def run(self):

        # prepare the data
        rag_benchmark: RagBenchmark = self.data_loader.get_benchmark()

        # run ingest
        ingest_artifacts = self.ingest_pipeline.process(data_loader=self.data_loader)

        # set the ingest artifacts for the inference pipeline
        self.inference_pipeline.set_ingest_artifacts(ingest_artifacts=ingest_artifacts)

        # ml-flow wrapper for the inference part
        results: list[InferenceResult] = []
        for benchmark_entry in rag_benchmark.get_benchmark_entries():
            # run the inference
            result: InferenceResult = self.inference_pipeline.process(
                benchmark_entry=benchmark_entry
            )
            # collect the result
            results.append(result)

        # Now run the evaluation via the evaluator code!
        # TODO!
