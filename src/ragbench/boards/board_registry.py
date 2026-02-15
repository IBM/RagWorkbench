from typing import Any, TypeVar

from ragbench.api.inference import InferenceParams, InferencePipeline
from ragbench.api.ingest import IngestParams, IngestPipeline

INGEST_T1 = TypeVar("INGEST_T1", bound=IngestPipeline)
INGEST_T2 = TypeVar("INGEST_T2", bound=IngestParams)

INFERENCE_T1 = TypeVar("INFERENCE_T1", bound=InferencePipeline)
INFERENCE_T2 = TypeVar("INFERENCE_T2", bound=InferenceParams)


class BoardRegistry:
    _ingest_pipelines: dict[str, tuple[type[IngestPipeline], type[IngestParams]]] = {}
    _inference_pipelines: dict[
        str, tuple[type[InferencePipeline], type[InferenceParams]]
    ] = {}

    @classmethod
    def register_ingest(
        cls,
        name: str,
        ingest_class: type[INGEST_T1],
        params_class: type[INGEST_T2],
    ):
        cls._ingest_pipelines[name] = (ingest_class, params_class)

    @classmethod
    def register_inference(
        cls,
        name: str,
        inference_class: type[INFERENCE_T1],
        params_class: type[INFERENCE_T2],
    ):
        cls._inference_pipelines[name] = (inference_class, params_class)

    @classmethod
    def create_ingest_pipeline(
        cls,
        name: str,
        params: dict[str, Any],
    ):
        ingest_class, params_class = cls._ingest_pipelines[name]
        params_instance = params_class.model_validate(params)
        ingest_instance = ingest_class()
        ingest_instance.set_params(params_instance)
        return ingest_instance

    @classmethod
    def create_inference_pipeline(
        cls,
        name: str,
        params: dict[str, Any],
    ):
        inference_class, params_class = cls._inference_pipelines[name]
        params_instance = params_class.model_validate(params)
        inference_instance = inference_class()
        inference_instance.set_params(params_instance)
        return inference_instance
