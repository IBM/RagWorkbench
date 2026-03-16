import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.caching.data_loader_cache import DataLoaderCache
from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)
from ragworkbench.datasets_loader.data_models.document_object import DocumentObject
from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    RagBenchmark,
    RagBenchmarkEntry,
)
from ragworkbench.datasets_loader.data_models.rag_corpus import RagCorpus
from ragworkbench.datasets_loader.dataset_names import DatasetName

logger: logging.Logger = logging.getLogger(__name__)


class RagDataLoader(ABC):
    """
    Abstract base class for loading RAG benchmark datasets.

    This class provides a framework for loading document corpora and benchmark
    question-answer pairs from various dataset sources. It handles sampling,
    filtering, and initialization of both the corpus and benchmark components.

    Subclasses must implement methods to retrieve documents and benchmark entries
    specific to their dataset format.

    Attributes:
        dataset_name: Identifier for the dataset being loaded.
        split: Dataset split ('train' or 'test'), or None for full dataset.
        benchmark: The sampled RagBenchmark instance.
        rag_corpus: The sampled RagCorpus instance.

    Example:
        >>> class MyDataLoader(RagDataLoader):
        ...     def _get_documents(self):
        ...         return [...]
        ...     def _get_benchmark_entries(self, split):
        ...         return [...]
        >>> loader = MyDataLoader(
        ...     dataset_name=DatasetName.AI_ARXIV,
        ...     split="test",
        ...     sampling_params=DataSamplingParams(question_limit=100)
        ... )
        >>> corpus = loader.get_corpus()
        >>> benchmark = loader.get_benchmark()
    """

    def __init__(
        self,
        dataset_name: DatasetName | str,
        split: DatasetSplit | None,
        sampling_params: DataSamplingParams = DataSamplingParams(),
        cache_dir: Path | None = None,
    ):
        """
        Initialize the RAG data loader.

        Args:
            dataset_name: Identifier for the dataset to load.
            split: Dataset split to load ('train', 'test', or None for all).
            sampling_params: Parameters controlling question and document sampling.
                           Defaults to no sampling (all data included).
            cache_dir: Optional directory for caching full (unsampled) dataset.

        Note:
            The initialization process:
            1. Loads or retrieves all documents and benchmark entries (full dataset)
            2. Caches the full dataset if not already cached (cache is independent of sampling)
            3. Applies sampling based on sampling_params to the full dataset
            4. Creates RagBenchmark and RagCorpus instances with sampled data
            5. Logs the final dataset size

            The cache stores the complete unsampled dataset, allowing different
            sampling parameters to be applied without reloading the original data.
        """
        self.dataset_name: DatasetName | str = dataset_name
        self.split: DatasetSplit | None = split
        self.sampling_params = sampling_params

        # Step 1: Load or retrieve full (unsampled) data
        self.all_docs: list[DocumentObject]
        all_benchmark_entries: list[RagBenchmarkEntry]

        if cache_dir is not None:
            # Initialize cache (only depends on dataset_name and split, not sampling_params)
            cache = DataLoaderCache(
                cache_dir=cache_dir,
                dataset_name=dataset_name,
                split=split,
            )
            cached_documents, cached_benchmark = cache.get()

            if cached_documents is not None and cached_benchmark is not None:
                # Cache HIT: Use full unsampled data from cache
                self.all_docs = cached_documents
                all_benchmark_entries = cached_benchmark.benchmark_entries
                logger.debug(
                    f"Loaded from cache - {len(self.all_docs)} documents and "
                    f"{len(all_benchmark_entries)} benchmark entries (full dataset) "
                    f"from '{dataset_name}', split '{split}'."
                )
            else:
                # Cache MISS: Load full data from source and cache it
                self.all_docs = self._get_documents()
                all_benchmark_entries = self._get_benchmark_entries(split=split)

                # Cache the full unsampled data
                full_benchmark = RagBenchmark(benchmark_entries=all_benchmark_entries)
                cache.add(full_benchmark, self.all_docs)
                logger.debug(
                    f"Loaded from source and cached - {len(self.all_docs)} documents and "
                    f"{len(all_benchmark_entries)} benchmark entries (full dataset) "
                    f"from '{dataset_name}', split '{split}'."
                )
        else:
            # No cache: Load full data from source
            self.all_docs = self._get_documents()
            all_benchmark_entries = self._get_benchmark_entries(split=split)
            logger.debug(
                f"Loaded from source (no cache) - {len(self.all_docs)} documents and "
                f"{len(all_benchmark_entries)} benchmark entries (full dataset) "
                f"from '{dataset_name}', split '{split}'."
            )

        # Step 2: Apply sampling to the full dataset (always, regardless of cache hit/miss)
        sampled_benchmark_entries: list[RagBenchmarkEntry]
        sampled_docs: list[DocumentObject]
        sampled_benchmark_entries, sampled_docs = self._load_sample(
            all_benchmark_entries, self.all_docs, sampling_params
        )

        # Step 3: Create final benchmark and corpus instances with sampled data
        self.benchmark: RagBenchmark = RagBenchmark(
            benchmark_entries=sampled_benchmark_entries
        )
        self.rag_corpus: RagCorpus = RagCorpus(documents=sampled_docs)

        logger.debug(
            f"After sampling - {len(self.rag_corpus)} documents and "
            f"{len(self.benchmark)} benchmark entries available for use."
        )

    @abstractmethod
    def _get_documents(self) -> list[DocumentObject]:
        """
        Retrieve all documents from the dataset.

        This method must be implemented by subclasses to load documents
        specific to their dataset format.

        Returns:
            List of DocumentObject instances representing the corpus.

        Note:
            This method is called during initialization before sampling is applied.
        """
        pass

    @abstractmethod
    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        """
        Retrieve all benchmark entries from the dataset.

        This method must be implemented by subclasses to load question-answer
        pairs specific to their dataset format.

        Args:
            split: Dataset split to load ('train', 'test', or None for all).

        Returns:
            List of RagBenchmarkEntry instances.

        Note:
            This method is called during initialization before sampling is applied.
        """
        pass

    def get_benchmark(self) -> RagBenchmark:
        """
        Get the loaded and sampled benchmark.

        Returns:
            RagBenchmark instance containing question-answer pairs.

        Example:
            >>> loader = MyDataLoader(...)
            >>> benchmark = loader.get_benchmark()
            >>> questions = benchmark.get_questions()
        """
        return self.benchmark

    def get_corpus(self) -> RagCorpus:
        """
        Get the loaded and sampled document corpus.

        Returns:
            RagCorpus instance containing documents.

        Example:
            >>> loader = MyDataLoader(...)
            >>> corpus = loader.get_corpus()
            >>> print(f"Corpus has {len(corpus)} documents")
        """
        return self.rag_corpus

    @staticmethod
    def _load_sample(
        benchmark_entries: list[RagBenchmarkEntry],
        full_docs: list[DocumentObject],
        sampling_params: DataSamplingParams,
    ) -> tuple[list[RagBenchmarkEntry], list[DocumentObject]]:
        """
        Sample benchmark entries and documents based on sampling parameters.

        This method applies two types of sampling:
        1. Question sampling: Limits the number of benchmark entries
        2. Document sampling: Limits documents to relevant ones plus a factor of non-relevant ones

        Args:
            benchmark_entries: Complete list of benchmark entries to sample from.
            full_docs: Complete list of documents to sample from.
            sampling_params: Parameters controlling the sampling behavior.

        Returns:
            Tuple of (sampled_benchmark_entries, sampled_documents).

        Note:
            Sampling is deterministic based on sampling_params.seed for reproducibility.
            Document sampling ensures all ground truth documents are included, then adds
            additional non-relevant documents based on document_factor.

        Example:
            >>> entries, docs = RagDataLoader._load_sample(
            ...     benchmark_entries,
            ...     all_docs,
            ...     DataSamplingParams(question_limit=50, document_factor=2, seed=42)
            ... )
            >>> print(f"Sampled {len(entries)} questions and {len(docs)} documents")
        """
        # Create copies to avoid modifying original lists
        docs: list[DocumentObject] = full_docs.copy()
        benchmark_entries_copy: list[RagBenchmarkEntry] = benchmark_entries.copy()
        random.seed(sampling_params.seed)

        # Apply question sampling if specified
        if sampling_params.question_limit:
            if sampling_params.question_limit < len(benchmark_entries_copy):
                benchmark_entries_copy = random.sample(
                    benchmark_entries_copy, sampling_params.question_limit
                )

        # Apply document sampling if specified
        if sampling_params.document_factor is not None:
            # Step 1: Get all ground truth document IDs from benchmark entries
            benchmark_doc_ids: set[str] = RagBenchmark.get_doc_ids_set(
                benchmark_entries_copy
            )

            # Step 2: Get all document IDs from the corpus
            total_doc_ids: list[str] = [d.name for d in docs]

            # Step 3: Remove duplicates and ground truth documents to get non-relevant docs
            non_relevant_doc_ids: list[str] = list(
                set(total_doc_ids) - benchmark_doc_ids
            )

            # Step 4: Sort and shuffle non-relevant documents for reproducibility
            non_relevant_doc_ids.sort()
            random.shuffle(non_relevant_doc_ids)

            # Step 5: Calculate how many non-relevant documents to include
            num_non_relevant: int = sampling_params.document_factor * len(
                benchmark_doc_ids
            )
            selected_non_relevant: list[str] = non_relevant_doc_ids[:num_non_relevant]

            # Step 6: Combine ground truth and non-relevant document IDs
            all_selected_doc_ids: set[str] = benchmark_doc_ids | set(
                selected_non_relevant
            )

            # Step 7: Filter documents to only include selected IDs
            docs = [d for d in docs if d.name in all_selected_doc_ids]

        return benchmark_entries_copy, docs
