import hashlib
import io
from typing import Literal

import pandas as pd  # type: ignore[import-untyped]
from datasets import load_dataset  # type: ignore[import-not-found]

from ragbench.datasets_loader import RagDataLoader
from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets_loader.data_models.document_object import DocumentObject
from ragbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)
from ragbench.datasets_loader.dataset_names import DatasetName


class NarrativeQaDataLoader(RagDataLoader):
    def __init__(
        self,
        split: Literal["train", "test"] | None,
        data_sampling: DataSamplingParams = DataSamplingParams(),
    ):
        # We read the content of the HF
        hf_dataset = load_dataset(
            "deepmind/narrativeqa"
        )  # ["train"].to_pandas()  # There is only train on HF
        self.hf_dataset_train_df = hf_dataset["train"].to_pandas()
        self.hf_dataset_test_df = hf_dataset["test"].to_pandas()
        # Since we do not have corpus, we map the ground_truth_context to be the corpus
        documents = (
            self.hf_dataset_train_df["document"].tolist()
            + self.hf_dataset_test_df["document"].tolist()
        )
        self.corpus_text_to_context_id: dict[str, GroundTruthContextId] = {
            d["text"]: GroundTruthContextId(document_id=d["id"]) for d in documents
        }

        super().__init__(
            dataset_name=DatasetName.NARRATIVE_QA,
            split=split,
            sampling_params=data_sampling,
        )

    def _get_documents(self) -> list[DocumentObject]:
        return [
            DocumentObject(
                mime_type="text/plain",
                name=ground_truth_doc_id.document_id,
                stream=io.BytesIO(str(text).encode("utf-8")),
            )
            for text, ground_truth_doc_id in self.corpus_text_to_context_id.items()
        ]

    def _get_benchmark_entries(
        self, split: Literal["train", "test"] | None
    ) -> list[RagBenchmarkEntry]:
        if split is not None:

            if split == "train":
                df = self.hf_dataset_train_df
            else:
                df = self.hf_dataset_test_df
        else:
            df = pd.concat([self.hf_dataset_train_df, self.hf_dataset_test_df], axis=1)

        entries = []
        for _, row in df.iterrows():
            hash_object = hashlib.md5()
            hash_object.update(str(row["question"]).encode("utf-8"))
            question_id = hash_object.hexdigest()
            answers = []
            for a in row["answers"]:
                if isinstance(a, float):
                    continue
                elif isinstance(a, dict):
                    answers.append(a["text"])
                else:
                    answers.append(a[0]["text"])
            document_id = (
                row["document"]["id"]
                if isinstance(row["document"], dict)
                else row["document"][0]["id"]
            )
            entry = RagBenchmarkEntry(
                question_id=question_id,
                question=str(row["question"]),
                ground_truth_answers=answers,
                ground_truth_context_ids=[
                    GroundTruthContextId(document_id=str(document_id))
                ],
            )
            entries.append(entry)
        return entries
