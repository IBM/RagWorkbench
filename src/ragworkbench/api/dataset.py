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

from enum import StrEnum, auto

from pydantic import BaseModel, field_validator

from ragworkbench.datasets_loader.data_models import DataSamplingParams
from ragworkbench.datasets_loader.dataset_names import DatasetName


class DatasetSplit(StrEnum):
    TRAIN = auto()  # automatically becomes "train"
    TEST = auto()  # automatically becomes "test"


class Dataset(BaseModel):
    name: DatasetName | str
    split: DatasetSplit | None = None
    sampling: DataSamplingParams = DataSamplingParams()

    @field_validator("split", mode="before")
    @classmethod
    def convert_split_to_enum(cls, v):
        """Convert string to DatasetSplit enum if needed."""
        if v is None:
            return v
        if isinstance(v, str):
            # Convert string to DatasetSplit enum
            # This handles both lowercase and uppercase strings
            return DatasetSplit(v.lower())
        return v

    def _format_dataset_name(self):
        if isinstance(self.name, DatasetName):
            dataset_name = self.name.value
        else:
            dataset_name = self.name
        return dataset_name

    def id(self):
        dataset_name = self._format_dataset_name()

        dataset_id = f"name-{dataset_name}"

        if self.split:
            dataset_id += f"_split-{self.split.value}"

        sample_id = self.sampling.as_id()
        if sample_id:
            dataset_id += f"_{sample_id}"

        return dataset_id
