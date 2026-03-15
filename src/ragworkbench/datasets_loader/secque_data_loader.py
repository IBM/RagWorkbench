import hashlib
import io
from pathlib import Path

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


class SecqueDataLoader(RagDataLoader):
    def __init__(
        self,
        split: DatasetSplit | None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        # We read the content of the HF
        self.hf_dataset_df = load_dataset("nogabenyoash/SecQue")[
            "train"
        ].to_pandas()  # There is only train on HF

        # Since we do not have corpus, we map the ground_truth_context to be the corpus
        context_html_without_headers = self.hf_dataset_df[
            "context_markdown_with_headers"
        ].tolist()
        self.corpus_text_to_context_id: dict[str, GroundTruthContextId] = {}
        for s in context_html_without_headers:
            shake = hashlib.shake_128()
            shake.update(s.encode("utf-8"))
            doc_id = shake.hexdigest(10)
            self.corpus_text_to_context_id[s] = GroundTruthContextId(
                document_id=f"{doc_id}.md"
            )

        super().__init__(
            dataset_name=DatasetName.SECQUE,
            split=split,
            sampling_params=sampling_params,
            cache_dir=cache_dir,
        )

    def _get_documents(self) -> list[DocumentObject]:
        return [
            DocumentObject(
                mime_type="text/markdown",
                name=ground_truth_doc_id.document_id,
                stream=io.BytesIO(str(text).encode("utf-8")),
            )
            for text, ground_truth_doc_id in self.corpus_text_to_context_id.items()
        ]

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        if split is not None:
            df_train = self.hf_dataset_df.sample(frac=0.7, random_state=42)
            df_test = self.hf_dataset_df.drop(df_train.index)

            if split == "train":
                df = df_train
            else:
                df = df_test
        else:
            df = self.hf_dataset_df

        entries = []
        for _, row in df.iterrows():
            answers = row["ground_truth_answer"]
            # ensure it's a list of str
            if isinstance(answers, str):
                answers = [answers]
            elif isinstance(answers, (list, tuple)):
                answers = [str(a) for a in answers]
            else:
                answers = [str(answers)]
            ground_truth_context_id = self.corpus_text_to_context_id.get(
                str(row["context_markdown_with_headers"])
            )
            if ground_truth_context_id is not None:
                entry = RagBenchmarkEntry(
                    question_id=str(row["QID"]),
                    question=str(row["Question"]),
                    ground_truth_answers=answers,
                    ground_truths_context_ids=[ground_truth_context_id],
                )
                entries.append(entry)
        return entries
