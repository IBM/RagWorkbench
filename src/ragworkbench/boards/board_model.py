# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ragworkbench.api.dataset import Dataset


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
