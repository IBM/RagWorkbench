from typing import Any, TypeVar

from ragbench.api.inference import InferenceParams, InferencePipeline
from ragbench.api.ingest import IngestParams, IngestPipeline

INGEST_T1 = TypeVar("INGEST_T1", bound=IngestPipeline)
INGEST_T2 = TypeVar("INGEST_T2", bound=IngestParams)

INFERENCE_T1 = TypeVar("INFERENCE_T1", bound=InferencePipeline)
INFERENCE_T2 = TypeVar("INFERENCE_T2", bound=InferenceParams)


class BoardRegistry:

    def __init__(self):
        self.ingest_pipelines: dict[
            str, tuple[type[IngestPipeline], type[IngestParams]]
        ] = {}
        self.inference_pipelines: dict[
            str, tuple[type[InferencePipeline], type[InferenceParams]]
        ] = {}

    def register_ingest(
        self,
        name: str,
        ingest_class: type[INGEST_T1],
        params_class: type[INGEST_T2],
    ):
        self.ingest_pipelines[name] = (ingest_class, params_class)

    def register_inference(
        self,
        name: str,
        inference_class: type[INFERENCE_T1],
        params_class: type[INFERENCE_T2],
    ):
        self.inference_pipelines[name] = (inference_class, params_class)

    def create_ingest_pipeline(
        self,
        name: str,
        params: dict[str, Any],
    ):
        ingest_class, params_class = self.ingest_pipelines[name]
        return ingest_class(params_class(**params))

    def create_inference_pipeline(
        self,
        name: str,
        params: dict[str, Any],
    ):
        inference_class, inference_params = self.inference_pipelines[name]
        return inference_class(inference_params(**params))
