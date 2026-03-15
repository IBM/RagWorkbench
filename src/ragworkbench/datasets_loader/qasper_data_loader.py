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


class QasperQaDataLoader(RagDataLoader):
    def __init__(
        self,
        split: DatasetSplit | None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        # We read the content of the HF
        hf_dataset = load_dataset(
            "allenai/qasper"
        )  # ["train"].to_pandas()  # There is only train on HF
        self.hf_dataset_train_df = hf_dataset["train"].to_pandas()
        self.hf_dataset_test_df = hf_dataset["test"].to_pandas()
        # Since we do not have corpus, we map the ground_truth_context to be the corpus
        documents = (
            self.hf_dataset_train_df["full_text"].tolist()
            + self.hf_dataset_test_df["full_text"].tolist()
        )
        doc_ids = (
            self.hf_dataset_train_df["id"].tolist()
            + self.hf_dataset_test_df["id"].tolist()
        )
        self.corpus_doc_id_to_text: dict[str, dict] = {}
        for doc_id, d in zip(doc_ids, documents, strict=True):
            self.corpus_doc_id_to_text[doc_id] = d

        super().__init__(
            dataset_name=DatasetName.QASPER,
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
        return [
            DocumentObject(
                mime_type="text/markdown",
                name=doc_id,
                stream=io.BytesIO(
                    str(self._json_to_markdown(data=text)).encode("utf-8")
                ),
            )
            for doc_id, text in self.corpus_doc_id_to_text.items()
        ]

    @staticmethod
    def _extract_answer(list_of_answers: list[dict]) -> list[str]:
        answers = []
        for answer_dict in list_of_answers:
            if len(answer_dict.get("free_form_answer", "")) > 0:
                answers.append(str(answer_dict["free_form_answer"]))
            elif len(answer_dict.get("extractive_spans", "")) > 0:
                answers.append(str(answer_dict["extractive_spans"]))
        return answers

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        if split is not None:

            if split == "train":
                df = self.hf_dataset_train_df
            else:
                df = self.hf_dataset_test_df
        else:
            df = pd.concat([self.hf_dataset_train_df, self.hf_dataset_test_df], axis=0)

        entries = []
        for _, row in df.iterrows():
            doc_id = str(row["id"])
            qas_dict = row["qas"]
            questions = qas_dict["question"]
            question_ids = qas_dict["question_id"]
            answers_list = qas_dict["answers"]
            for question_id, question, answers in zip(
                question_ids, questions, answers_list, strict=True
            ):
                # First we check that 'answer' contains content
                answer_dict = answers["answer"]
                answers_list = self._extract_answer(answer_dict)

                if len(answers_list) > 0:
                    entry = RagBenchmarkEntry(
                        question_id=str(question_id),
                        question=str(question),
                        ground_truth_answers=answers_list,
                        ground_truths_context_ids=[
                            GroundTruthContextId(document_id=doc_id)
                        ],
                    )
                    entries.append(entry)
        return entries
