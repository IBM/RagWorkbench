from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ragbench.api.dataset import Dataset


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
    scores: dict[str, str]
    chart: Chart


class Report(BaseModel):
    screens: list[Screen]


class Board(BaseModel):
    name: str
    description: str
    datasets: list[Dataset]
    configurations: list[Configuration]
    metrics: list[str]
    report: Report


class BoardResults(BaseModel):
    board: Board
    results: pd.DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)
