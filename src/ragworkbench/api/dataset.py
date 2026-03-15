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
