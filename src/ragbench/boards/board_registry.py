from typing import Any, Generic, TypeVar

from ragbench.api.inference import InferenceParams, InferencePipeline
from ragbench.api.ingest import IngestParams, IngestPipeline

U1 = TypeVar("U1", bound=IngestParams)
U2 = TypeVar("U2", bound=InferenceParams)


class BoardRegistry(Generic[U1, U2]):

    def __init__(self):
        self.ingest_pipelines: dict[str, tuple[type[IngestPipeline[U1]], type[U1]]] = {}
        self.inference_pipelines: dict[
            str, tuple[type[InferencePipeline[U2]], type[U2]]
        ] = {}

    def register_ingest(
        self,
        name: str,
        ingest_class: type[IngestPipeline[U1]],
        params_class: type[U1],
    ):
        self.ingest_pipelines[name] = (ingest_class, params_class)

    def register_inference(
        self,
        name: str,
        inference_class: type[InferencePipeline[U2]],
        params_class: type[U2],
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
        ingest_class, params_class = self.inference_pipelines[name]
        return ingest_class(params_class(**params))
