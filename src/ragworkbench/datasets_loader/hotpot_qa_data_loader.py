import hashlib
import io
from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import load_dataset  # type: ignore[import-not-found]

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.datasets_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)
from ragworkbench.datasets_loader.data_models.document_object import DocumentObject
from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)
from ragworkbench.datasets_loader.dataset_names import DatasetName


class HotpotQaDataLoader(RagDataLoader):
    def __init__(
        self,
        split: DatasetSplit | None,
        level: Literal["easy", "medium", "hard"] | None = None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        self.total_df: pd.DataFrame = load_dataset(
            path="hotpotqa/hotpot_qa", name="distractor", split="train"
        ).to_pandas()
        self.level = level
        super().__init__(
            dataset_name=DatasetName.HOTPOT_QA,
            split=split,
            sampling_params=sampling_params,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _hash_text(text: str) -> str:
        normalized = HotpotQaDataLoader._normalize_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_text(s: str) -> str:
        s = s.lower().strip()
        # Add these lines temporarily
        s = s.replace("\u00a0", " ")  # nbsp → space
        s = s.replace("\u200b", "")  # zero width space
        s = s.replace("\u200c", "").replace("\u200d", "")
        s = s.replace("\xad", "")  # soft hyphen

        # Unicode normalization (very often solves the é / é issue)
        import unicodedata

        s = unicodedata.normalize(
            "NFC", s
        )  # or "NFKC" if you also want to fold compatibility chars

        return s.replace(" ", "_").replace(".", "_").lower()

    @staticmethod
    def _title_to_document_id(title: str, text: str) -> str:

        return f"{HotpotQaDataLoader._normalize_text(title)}_{HotpotQaDataLoader._hash_text(text)}.txt"

    def _get_documents(self) -> list[DocumentObject]:
        title_to_text: dict[str, str] = {}
        for ctx in self.total_df["context"]:
            for title, sentences in zip(ctx["title"], ctx["sentences"], strict=True):
                text = " ".join(sentences)
                normalized_title = self._title_to_document_id(title=title, text=text)
                if normalized_title in title_to_text:
                    if self._normalize_text(text) != self._normalize_text(
                        title_to_text[normalized_title]
                    ):
                        print(
                            f"{normalized_title}\n`{self._normalize_text(text)}`\n!=\n`{self._normalize_text(title_to_text[normalized_title])}`\n\n"
                        )
                else:
                    title_to_text[normalized_title] = text
        # Now we have removed all duplication
        docs: list[DocumentObject] = []
        for title, text in title_to_text.items():
            doc: DocumentObject = DocumentObject(
                name=title,
                mime_type="text/plain",
                stream=io.BytesIO(str(text).encode("utf-8")),
            )
            docs.append(doc)
        return docs

    @staticmethod
    def _row_to_entry(row) -> RagBenchmarkEntry:
        ground_truth_context_ids_list = []
        context_dict = row["context"]
        for title, sentences in zip(
            context_dict["title"], context_dict["sentences"], strict=True
        ):
            text = " ".join(sentences)
            normalized_title = HotpotQaDataLoader._title_to_document_id(
                title=title, text=text
            )
            ground_truth_context_ids_list.append(
                GroundTruthContextId(document_id=normalized_title)
            )

        return RagBenchmarkEntry(
            question_id=row["id"],
            question=row["question"],
            ground_truth_answers=[row["answer"]],
            ground_truths_context_ids=ground_truth_context_ids_list,
            is_answerable=True,
        )

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        if split is not None:
            df_train = self.total_df.sample(frac=0.7, random_state=42)
            df_test = self.total_df.drop(df_train.index)

            if split == "train":
                df = df_train
            else:
                df = df_test
        else:
            df = self.total_df

        if self.level is not None:
            df = df[df["level"] == self.level].copy()

        return [self._row_to_entry(row) for _, row in df.iterrows()]
