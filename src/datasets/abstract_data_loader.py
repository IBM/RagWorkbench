import logging
import random
from abc import ABC, abstractmethod
from typing import Literal

from rag_unitxt_cards.data_models.data_sampling_params import DataSamplingParams
from rag_unitxt_cards.data_models.dataset_names import DatasetName
from rag_unitxt_cards.data_models.document_object import DocumentObject
from rag_unitxt_cards.data_models.rag_benchmark import RagBenchmark, RagBenchmarkEntry
from rag_unitxt_cards.data_models.rag_corpus import RagCorpus, RagCorpusMetadata

logger = logging.getLogger("rag_unitxt_cards")


class RagDataLoader(ABC):
    def __init__(
        self,
        dataset_name: DatasetName,
        split: Literal["train", "test"] | None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.all_docs: list[DocumentObject] = self._get_documents()
        self.all_benchmark_entries = self._get_benchmark_entries(split=split)

        sampled_benchmark_entries, sampled_docs = self._load_sample(
            self.all_benchmark_entries, self.all_docs, sampling_params
        )
        self.benchmark = RagBenchmark(benchmark_entries=sampled_benchmark_entries)
        rag_corpus_metadata: RagCorpusMetadata = self._get_corpus_metadata()
        self.rag_corpus = RagCorpus(
            documents=sampled_docs, corpus_metadata=rag_corpus_metadata
        )
        logger.debug(
            f"Loaded {len(self.rag_corpus)} documents and {len(self.benchmark)} labeled examples '{dataset_name}', split '{split}'."
        )

    @abstractmethod
    def _get_documents(self) -> list[DocumentObject]:
        pass

    @abstractmethod
    def _get_benchmark_entries(
        self, split: Literal["train", "test"] | None
    ) -> list[RagBenchmarkEntry]:
        pass

    @abstractmethod
    def _get_corpus_metadata(self) -> RagCorpusMetadata:
        pass

    def get_benchmark(self) -> RagBenchmark:
        return self.benchmark

    def get_corpus(self) -> RagCorpus:
        return self.rag_corpus

    @staticmethod
    def _load_sample(
        benchmark_entries: list[RagBenchmarkEntry],
        full_docs: list[DocumentObject],
        sampling_params: DataSamplingParams,
    ) -> tuple[list[RagBenchmarkEntry], list[DocumentObject]]:
        docs = full_docs.copy()
        benchmark_entries = benchmark_entries.copy()
        random.seed(sampling_params.seed)

        if sampling_params.question_limit:
            # We select a subset of the df:
            if sampling_params.question_limit < len(benchmark_entries):
                benchmark_entries = random.sample(
                    benchmark_entries, sampling_params.question_limit
                )

        if sampling_params.document_factor is not None:
            # Now we must limit the documents!
            # 1 - We get all the gt_doc_ids:
            benchmark_doc_ids = RagBenchmark.get_doc_ids_set(benchmark_entries)

            # 2 - We get all the label doc_ids:
            total_doc_ids = [d.name for d in docs]
            # 2a - We remove duplicates and we remove the benchmark_doc_ids
            total_doc_ids = list(set(total_doc_ids) - benchmark_doc_ids)
            # 2b - We shuffle them according to the seed
            # 2b1 - First we sort
            total_doc_ids.sort()
            # 2b2 : we shuffle
            random.shuffle(total_doc_ids)

            # 3 - We take the number of doc_ids
            other_docs_ids_size = sampling_params.document_factor * len(
                benchmark_doc_ids
            )
            other_doc_ids = total_doc_ids[:other_docs_ids_size]

            # 4 - we merge the two sets:
            all_docs_ids = benchmark_doc_ids | set(other_doc_ids)

            # 5 - we select the documents:
            docs = [d for d in docs if d.name in all_docs_ids]

        return benchmark_entries, docs
