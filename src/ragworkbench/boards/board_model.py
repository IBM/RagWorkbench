from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ragworkbench.api.dataset import Dataset


class CacheMode(StrEnum):
    """Cache operation modes."""

    ON = "on"  # Cache is enabled: read and write
    OFF = "off"  # Cache is disabled: no read or write
    REFRESH = "refresh"  # Refresh mode: skip disk loading, use in-memory cache, write new results


class PipelineConfiguration(BaseModel):
    name: str
    params: dict[str, Any]


class Configuration(BaseModel):
    name: str
    description: str
    ingest: PipelineConfiguration
    inference: PipelineConfiguration


class Chart(BaseModel):
    type: str
    title: str


class Screen(BaseModel):
    title: str
    columns: dict[str, str]
    chart: Chart


class Report(BaseModel):
    screens: list[Screen]


class ExperimentConfig(BaseModel):
    """Configuration for experiment-level settings."""

    usage_tracking: bool = False
    litellm_proxy_url: str = "http://localhost:4000"
    cache: CacheMode = CacheMode.ON


class Board(BaseModel):
    name: str
    description: str
    datasets: list[Dataset]
    configurations: list[Configuration]
    metrics: list[str]
    report: Report
    experiment: ExperimentConfig = ExperimentConfig()


class BoardResults(BaseModel):
    board: Board
    results: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)
