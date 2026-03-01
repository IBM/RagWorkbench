from typing import Literal

from pydantic import BaseModel, computed_field

from ragbench import DatasetName
from ragbench.datasets_loader.data_models import DataSamplingParams


class Dataset(BaseModel):
    name: DatasetName | str
    split: Literal["train", "test"] | None = None
    sampling: DataSamplingParams = DataSamplingParams()

    @computed_field
    def id(self) -> str:
        """
        Generate a unique identifier for this dataset configuration.

        Combines the dataset name, split, and sampling parameters into a
        single string that uniquely identifies this dataset configuration.

        Returns:
            A string identifier, e.g. 'bioasq_test_q-50_seed-43' or 'bioasq_test'.
        """
        parts: list[str] = [str(self.name)]

        if self.split is not None:
            parts.append(self.split)

        sampling_id = self.sampling.as_id()
        if sampling_id:
            parts.append(sampling_id)

        return "_".join(parts)
