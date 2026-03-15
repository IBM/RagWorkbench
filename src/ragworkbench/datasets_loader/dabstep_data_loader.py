import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from datasets import load_dataset  # type: ignore[import-not-found]
from huggingface_hub import HfFileSystem  # type: ignore[import-not-found]

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.datasets_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)
from ragworkbench.datasets_loader.data_models.document_object import DocumentObject
from ragworkbench.datasets_loader.data_models.rag_benchmark import RagBenchmarkEntry
from ragworkbench.datasets_loader.dataset_names import DatasetName
from ragworkbench.datasets_loader.datasets_utils import guess_mime

NUMBER_OF_THREADS = 2


class DabStepDataLoader(RagDataLoader):
    def __init__(
        self,
        split: DatasetSplit | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
        verbose: bool = True,
        progress_every: int = 50,
    ):
        self.verbose = verbose
        self.progress_every = max(1, int(progress_every))

        super().__init__(
            dataset_name=DatasetName.DABSTEP,
            split=split,
            sampling_params=sampling_params,
            cache_dir=cache_dir,
        )

    def _get_documents(self) -> list[DocumentObject]:
        base = "hf://datasets/adyen/DABstep/data/context"
        # Deterministic order
        paths_str: list[str] = sorted(p for p in HfFileSystem().glob(f"{base}/**/*"))

        def load_one(path: str) -> tuple[str, DocumentObject]:
            fs = HfFileSystem()  # thread-safe: one FS per worker
            with fs.open(path, "rb") as f:
                content = f.read()

            # We have to keep all the path from input/ as document name.
            name = path.split("/context/", 1)[1]
            mime_type = guess_mime(name)

            doc_obj = DocumentObject(
                name=name,
                mime_type=mime_type,
                stream=io.BytesIO(content),
                metadata={"path": path},
            )
            return path, doc_obj

        # Submit all jobs
        results: dict[str, DocumentObject] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=NUMBER_OF_THREADS) as executor:
            future_to_path = {
                executor.submit(load_one, path): path for path in paths_str
            }

            for future in as_completed(future_to_path):
                path, doc = future.result()
                results[path] = doc
                completed += 1

        # Preserve deterministic order
        documents = [results[path] for path in paths_str]
        return documents

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        test_df: pd.DataFrame = load_dataset(
            path="adyen/DABstep", name="tasks", split="default"
        ).to_pandas()
        train_df: pd.DataFrame = load_dataset(
            path="adyen/DABstep", name="tasks", split="dev"
        ).to_pandas()
        # Split is None
        df = pd.concat([train_df, test_df])
        if split is not None:
            if split == "train":
                df = train_df
            elif split == "test":
                df = test_df
        benchmark_entries: list[RagBenchmarkEntry] = []
        for _, row in df.iterrows():
            if "answer" in row:
                ground_truth_answers: list[str] = [str(row["answer"])]
                rag_benchmark_entry = RagBenchmarkEntry(
                    question_id=str(row["task_id"]),
                    question=str(row["question"]),
                    ground_truth_answers=ground_truth_answers,
                    ground_truths_context_ids=[],
                    additional_information={
                        "guidelines": row["guidelines"],
                        "level": row["level"],
                    },
                )
                benchmark_entries.append(rag_benchmark_entry)

        return benchmark_entries
