import io
from pathlib import Path

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


class MLDRDataLoader(RagDataLoader):
    def __init__(
        self,
        split: DatasetSplit | None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        # # We read the content of the HF
        # hf_dataset = load_dataset("Shitao/MLDR")  # ["train"].to_pandas()  # There is only train on HF
        # self.hf_dataset_train_df = hf_dataset["train"].to_pandas()
        # self.hf_dataset_test_df = hf_dataset["test"].to_pandas()
        # # Since we do not have corpus, we map the ground_truth_context to be the corpus
        # documents = self.hf_dataset_train_df["full_text"].tolist() + self.hf_dataset_test_df["full_text"].tolist()
        # doc_ids = self.hf_dataset_train_df["id"].tolist() + self.hf_dataset_test_df["id"].tolist()
        # self.corpus_doc_id_to_text: dict[str, dict] = {doc_id: d for doc_id, d in zip(doc_ids, documents)}
        self.corpus_df: pd.DataFrame = load_dataset(
            "Shitao/MLDR", "corpus-en", split="corpus"
        ).to_pandas()
        self.train_df: pd.DataFrame = load_dataset(
            "Shitao/MLDR", "en", split="train"
        ).to_pandas()
        self.test_df: pd.DataFrame = load_dataset(
            "Shitao/MLDR", "en", split="test"
        ).to_pandas()

        super().__init__(
            dataset_name=DatasetName.MLDR,
            split=split,
            sampling_params=sampling_params,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _json_to_markdown(data: dict) -> str:
        md_lines = []

        for section, paras in zip(
            data["section_name"], data["paragraphs"], strict=True
        ):
            # Split section hierarchy (e.g. "Proposed Method ::: Polarity Function")

            if section is not None:
                parts = [p.strip() for p in section.split(":::")]

                for depth, part in enumerate(parts, start=1):
                    heading = "#" * depth + " " + part
                    md_lines.append(heading)

                # Add paragraphs under the last heading
                for para in paras:
                    md_lines.append("")
                    md_lines.append(para.strip())
                    md_lines.append("")

        return "\n".join(md_lines).strip()

    def _get_documents(self) -> list[DocumentObject]:
        doc_id_to_text: dict = self.corpus_df.set_index("docid")["text"].to_dict()

        # Some docid miss in the corpus - so we take their content from the benchmark
        for benchmark_df in [self.train_df, self.test_df]:
            for _, row in benchmark_df.iterrows():
                for p in row["positive_passages"]:
                    docid = p["docid"]
                    text = p["text"]
                    assert (
                        docid not in doc_id_to_text.keys()
                        or text == doc_id_to_text[docid]
                    ), f"We found the content of `{docid}` to be different in corpus and benchmark"
                    doc_id_to_text[docid] = text

        document_objects: list[DocumentObject] = []
        for docid, text in doc_id_to_text.items():
            document_objects.append(
                DocumentObject(
                    mime_type="text/plain",
                    name=docid,
                    stream=io.BytesIO(str(text).encode("utf-8")),
                )
            )
        return document_objects

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        if split is None:
            df: pd.DataFrame = pd.concat([self.train_df, self.test_df], axis=0)
        else:
            if split == "train":
                df = self.train_df
            elif split == "test":
                df = self.test_df
            else:
                raise NotImplementedError(f"Got an unexpected split `{split}`")

        entries = []
        for _, row in df.iterrows():
            query_id = str(row["query_id"])
            gt_docs_ids = [
                GroundTruthContextId(document_id=p["docid"])
                for p in row["positive_passages"]
            ]
            question = str(row["query"])
            entry = RagBenchmarkEntry(
                question_id=query_id,
                question=question,
                ground_truths_context_ids=gt_docs_ids,
            )
            entries.append(entry)
        return entries
