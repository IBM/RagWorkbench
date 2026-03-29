from pathlib import Path
from typing import Any, TypeVar

from ragworkbench.api.inference import InferenceParams, InferencePipeline
from ragworkbench.api.ingest import IngestParams, IngestPipeline
from ragworkbench.boards.board_model import CacheMode

INGEST_T1 = TypeVar("INGEST_T1", bound=IngestPipeline)
INGEST_T2 = TypeVar("INGEST_T2", bound=IngestParams)

INFERENCE_T1 = TypeVar("INFERENCE_T1", bound=InferencePipeline)
INFERENCE_T2 = TypeVar("INFERENCE_T2", bound=InferenceParams)


def ingest_pipeline(name: str, params_class: type[IngestParams]):
    """
    Decorator to register an ingest pipeline with the BoardRegistry.

    Usage:
        @ingest_pipeline(name="my_pipeline", params_class=MyIngestParams)
        class MyIngest(IngestPipeline):
            ...
    """

    def decorator(cls: type[IngestPipeline]) -> type[IngestPipeline]:
        BoardRegistry.register_ingest(name, cls, params_class)
        return cls

    return decorator


def inference_pipeline(name: str, params_class: type[InferenceParams]):
    """
    Decorator to register an inference pipeline with the BoardRegistry.

    Usage:
        @inference_pipeline(name="my_pipeline", params_class=MyInferenceParams)
        class MyInference(InferencePipeline):
            ...
    """

    def decorator(cls: type[InferencePipeline]) -> type[InferencePipeline]:
        BoardRegistry.register_inference(name, cls, params_class)
        return cls

    return decorator


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
        if name not in cls._ingest_pipelines:
            raise ValueError(
                f"Ingest pipeline '{name}' not found. "
                f"Available pipelines: {list(cls._ingest_pipelines.keys())}. "
                f"Make sure the pipeline module is imported and decorated with @ingest_pipeline."
            )

        ingest_class, params_class = cls._ingest_pipelines[name]
        return ingest_class(params_class.model_validate(params))

    @classmethod
    def create_inference_pipeline(
        cls,
        name: str,
        params: dict[str, Any],
        cache_dir: Path | None = None,
        cache_mode: CacheMode = CacheMode.ON,
    ):
        if name not in cls._inference_pipelines:
            raise ValueError(
                f"Inference pipeline '{name}' not found. "
                f"Available pipelines: {list(cls._inference_pipelines.keys())}. "
                f"Make sure the pipeline module is imported and decorated with @inference_pipeline."
            )

        inference_class, inference_params = cls._inference_pipelines[name]
        return inference_class(
            inference_params.model_validate(params),
            cache_dir=cache_dir,
            cache_mode=cache_mode,
        )
