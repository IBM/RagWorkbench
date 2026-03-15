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

from abc import ABC, abstractmethod

from pydantic import BaseModel

from ragworkbench import RagDataLoader
from ragworkbench.api.ingest_artifact import IngestArtifact


class IngestParams(BaseModel):
    pass


class IngestRuntimeParams(BaseModel):
    pass


class IngestPipeline(ABC):

    @abstractmethod
    def __init__(self, _params: IngestParams) -> None:
        pass

    @abstractmethod
    def process(
        self,
        data_loader: RagDataLoader,
    ) -> list[IngestArtifact]:
        pass
