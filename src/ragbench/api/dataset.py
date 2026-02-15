from typing import Literal

from pydantic import BaseModel

from ragbench import DatasetName
from ragbench.datasets_loader.data_models import DataSamplingParams


class Dataset(BaseModel):
    name: DatasetName | str
    split: Literal["train", "test"] | None = None
    sampling: DataSamplingParams = DataSamplingParams()
