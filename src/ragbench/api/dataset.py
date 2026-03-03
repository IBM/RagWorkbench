from enum import StrEnum, auto

from pydantic import BaseModel

from ragbench import DatasetName
from ragbench.datasets_loader.data_models import DataSamplingParams


class DatasetSplit(StrEnum):
    TRAIN = auto()  # automatically becomes "train"
    TEST = auto()  # automatically becomes "test"


class Dataset(BaseModel):
    name: DatasetName | str
    split: DatasetSplit | None = None
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
            dataset_id += f"_split-{self.split.value}"

        sample_id = self.sampling.as_id()
        if sample_id:
            dataset_id += f"_{sample_id}"

        return dataset_id
