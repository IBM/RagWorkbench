"""
DataLoader Factory for creating RAG benchmark dataset loaders.

This module provides a factory class for instantiating the appropriate DataLoader
based on the dataset name, simplifying the process of loading different RAG benchmarks.
"""

import logging
from typing import Any, Literal

from ragbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragbench.datasets_loader.clap_nq_data_loader import ClapNqDataLoader
from ragbench.datasets_loader.da_code_data_loader import DaCodeDataLoader
from ragbench.datasets_loader.dabstep_data_loader import DabStepDataLoader
from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets_loader.dataset_names import DatasetName
from ragbench.datasets_loader.hotpot_qa_data_loader import HotpotQaDataLoader
from ragbench.datasets_loader.kramabench_data_loader import KramabenchDataLoader
from ragbench.datasets_loader.miniwiki_data_loader import MiniWikiDataLoader
from ragbench.datasets_loader.mldr_data_loader import MLDRDataLoader
from ragbench.datasets_loader.narrative_qa_data_loader import NarrativeQaDataLoader
from ragbench.datasets_loader.office_qa_data_loader import OfficeQADataLoader
from ragbench.datasets_loader.qasper_data_loader import QasperQaDataLoader
from ragbench.datasets_loader.secque_data_loader import SecqueDataLoader
from ragbench.datasets_loader.watsonx_data_loader import WatsonxDocsQADataLoader

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory class for creating DataLoader instances based on dataset name.

    This factory provides a unified interface for instantiating any of the available
    RAG benchmark dataset loaders. It handles dataset name validation, parameter
    routing, and provides clear error messages for invalid inputs.

    The factory supports all 14 available datasets and their specific parameters,
    while maintaining a consistent API across all loaders.

    Attributes:
        _LOADER_REGISTRY: Internal mapping of DatasetName to loader classes.

    Example:
        Basic usage with enum:
        >>> from ragbench.datasets_loader import DataLoaderFactory
        >>> from ragbench.datasets_loader.dataset_names import DatasetName
        >>> loader = DataLoaderFactory.create_loader(
        ...     dataset_name=DatasetName.BIOASQ,
        ...     split="train"
        ... )
        >>> corpus = loader.get_corpus()
        >>> benchmark = loader.get_benchmark()

        Using string dataset name:
        >>> loader = DataLoaderFactory.create_loader(
        ...     dataset_name="bioasq",
        ...     split="test"
        ... )

        With sampling parameters:
        >>> from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
        >>> loader = DataLoaderFactory.create_loader(
        ...     dataset_name=DatasetName.HOTPOT_QA,
        ...     split="train",
        ...     sampling_params=DataSamplingParams(
        ...         question_limit=100,
        ...         document_factor=2,
        ...         seed=42
        ...     )
        ... )

        With loader-specific parameters:
        >>> # HotpotQA with difficulty level
        >>> loader = DataLoaderFactory.create_loader(
        ...     dataset_name=DatasetName.HOTPOT_QA,
        ...     split="train",
        ...     level="hard"
        ... )
        >>> # Kramabench with verbose output
        >>> loader = DataLoaderFactory.create_loader(
        ...     dataset_name=DatasetName.KRAMABENCH,
        ...     split="test",
        ...     verbose=True,
        ...     progress_every=10
        ... )

    Note:
        Some loaders have specific parameters:
        - HotpotQA: `level` (Literal["easy", "medium", "hard"])
        - Kramabench: `verbose` (bool), `progress_every` (int)
        - MLDR: `language` (str)

        These can be passed as keyword arguments to create_loader().
    """

    # Registry mapping DatasetName to loader classes
    _LOADER_REGISTRY: dict[DatasetName, type[RagDataLoader]] = {
        DatasetName.BIOASQ: BioasqDataLoader,
        DatasetName.CLAP_NQ: ClapNqDataLoader,
        DatasetName.DA_CODE: DaCodeDataLoader,
        DatasetName.DABSTEP: DabStepDataLoader,
        DatasetName.HOTPOT_QA: HotpotQaDataLoader,
        DatasetName.KRAMABENCH: KramabenchDataLoader,
        DatasetName.MINI_WIKI: MiniWikiDataLoader,
        DatasetName.MLDR: MLDRDataLoader,
        DatasetName.NARRATIVE_QA: NarrativeQaDataLoader,
        DatasetName.OFFICEQA: OfficeQADataLoader,
        DatasetName.QASPER: QasperQaDataLoader,
        DatasetName.SECQUE: SecqueDataLoader,
        DatasetName.WATSONX_DOCS_QA: WatsonxDocsQADataLoader,
    }

    @classmethod
    def create_loader(
        cls,
        dataset_name: DatasetName | str,
        split: Literal["train", "test"] | None = None,
        sampling_params: DataSamplingParams | None = None,
        **kwargs: Any,
    ) -> RagDataLoader:
        """
        Create and return a DataLoader instance for the specified dataset.

        This method instantiates the appropriate DataLoader class based on the
        dataset name, passing through all relevant parameters including split,
        sampling parameters, and any loader-specific keyword arguments.

        Args:
            dataset_name: The dataset to load. Can be a DatasetName enum value
                         or a string that matches a valid dataset name.
            split: Dataset split to load ('train', 'test', or None for all data).
                  Default is None.
            sampling_params: Parameters controlling question and document sampling.
                           If None, defaults to DataSamplingParams() (no sampling).
            **kwargs: Additional loader-specific parameters. Examples:
                     - level: For HotpotQA ("easy", "medium", or "hard")
                     - verbose: For Kramabench (bool)
                     - progress_every: For Kramabench (int)
                     - language: For MLDR (str)

        Returns:
            An initialized RagDataLoader instance for the specified dataset.
            The loader can be used to access the corpus via get_corpus() and
            the benchmark via get_benchmark().

        Raises:
            ValueError: If dataset_name is not a valid dataset name.
            TypeError: If the loader class cannot be instantiated with the
                      provided parameters.

        Example:
            >>> # Basic usage
            >>> loader = DataLoaderFactory.create_loader("bioasq", split="train")
            >>> print(f"Loaded {len(loader.get_corpus())} documents")

            >>> # With sampling
            >>> loader = DataLoaderFactory.create_loader(
            ...     dataset_name=DatasetName.NARRATIVE_QA,
            ...     split="test",
            ...     sampling_params=DataSamplingParams(question_limit=50)
            ... )

            >>> # With loader-specific parameters
            >>> loader = DataLoaderFactory.create_loader(
            ...     dataset_name="hotpot_qa",
            ...     split="train",
            ...     level="hard",
            ...     sampling_params=DataSamplingParams(question_limit=100)
            ... )

        Note:
            The factory automatically handles parameter routing to the appropriate
            loader constructor. Not all loaders support all parameters - refer to
            individual loader documentation for specific parameter requirements.
        """
        # Convert string to DatasetName enum if necessary
        if isinstance(dataset_name, str):
            dataset_name = DatasetName.from_string(dataset_name)

        # Validate dataset name is in registry
        if dataset_name not in cls._LOADER_REGISTRY:
            available = ", ".join([name.value for name in cls._LOADER_REGISTRY.keys()])
            raise ValueError(
                f"No loader registered for dataset '{dataset_name.value}'. "
                f"Available datasets: {available}"
            )

        # Get the loader class
        loader_class = cls._LOADER_REGISTRY[dataset_name]

        # Prepare constructor arguments
        # Default sampling_params if not provided
        if sampling_params is None:
            sampling_params = DataSamplingParams()

        # Build constructor arguments based on loader requirements
        constructor_args: dict[str, Any] = {
            "split": split,
        }

        # Add sampling_params with the appropriate parameter name
        # Most loaders use 'data_sampling', but some use 'sampling_params'
        if loader_class in [
            BioasqDataLoader,
            ClapNqDataLoader,
            WatsonxDocsQADataLoader,
        ]:
            constructor_args["sampling_params"] = sampling_params
        else:
            constructor_args["data_sampling"] = sampling_params

        # Add any additional kwargs (loader-specific parameters)
        constructor_args.update(kwargs)

        try:
            # Instantiate and return the loader
            logger.debug(
                f"Creating {loader_class.__name__} with split='{split}', "
                f"sampling_params={sampling_params}, kwargs={kwargs}"
            )
            return loader_class(**constructor_args)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate {loader_class.__name__} with provided parameters. "
                f"Error: {str(e)}. Check that all required parameters are provided and "
                f"that parameter names are correct for this loader."
            ) from e

    @classmethod
    def list_available_datasets(cls) -> list[str]:
        """
        Return a list of all available dataset names.

        Returns:
            List of dataset name strings that can be used with create_loader().

        Example:
            >>> datasets = DataLoaderFactory.list_available_datasets()
            >>> print(f"Available datasets: {', '.join(datasets)}")
            Available datasets: bioasq, clap_nq, da_code, ...
        """
        return [name.value for name in cls._LOADER_REGISTRY.keys()]

    @classmethod
    def get_loader_class(cls, dataset_name: DatasetName | str) -> type[RagDataLoader]:
        """
        Get the loader class for a given dataset name without instantiating it.

        This method is useful for introspection or when you need to access
        class-level attributes or methods before instantiation.

        Args:
            dataset_name: The dataset name. Can be a DatasetName enum value
                         or a string that matches a valid dataset name.

        Returns:
            The loader class (not an instance) for the specified dataset.

        Raises:
            ValueError: If dataset_name is not a valid dataset name.

        Example:
            >>> loader_class = DataLoaderFactory.get_loader_class("bioasq")
            >>> print(loader_class.__name__)
            BioasqDataLoader
            >>> # Check if a loader has a specific method
            >>> hasattr(loader_class, '_get_documents')
            True
        """
        # Convert string to DatasetName enum if necessary
        if isinstance(dataset_name, str):
            dataset_name = DatasetName.from_string(dataset_name)

        # Validate dataset name is in registry
        if dataset_name not in cls._LOADER_REGISTRY:
            available = ", ".join([name.value for name in cls._LOADER_REGISTRY.keys()])
            raise ValueError(
                f"No loader registered for dataset '{dataset_name.value}'. "
                f"Available datasets: {available}"
            )

        return cls._LOADER_REGISTRY[dataset_name]
