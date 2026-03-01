from typing import Literal

from pydantic import BaseModel

from ragbench import DatasetName
from ragbench.datasets_loader.data_models import DataSamplingParams


class Dataset(BaseModel):
    name: DatasetName | str
    split: Literal["train", "test"] | None = None
    sampling: DataSamplingParams = DataSamplingParams()

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
            dataset_id += f"_split-{self.split}"

        sample_id = self.sampling.as_id()
        if sample_id:
            dataset_id += f"_{sample_id}"

        return dataset_id
