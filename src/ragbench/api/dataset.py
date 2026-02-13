from typing import Literal

from ragbench import DatasetName
from ragbench.datasets_loader.data_models import DataSamplingParams


class Dataset:
    name: DatasetName | str
    split: Literal["train", "test"] | None = None
    sampling: DataSamplingParams = DataSamplingParams()
