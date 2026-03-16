"""
DataLoader Factory for creating RAG benchmark dataset loaders.

This module provides a factory class for instantiating the appropriate DataLoader
based on the dataset name, simplifying the process of loading different RAG benchmarks.
"""

import logging
from pathlib import Path
from typing import Any

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.datasets_loader.abstract_data_loader import RagDataLoader
from ragworkbench.datasets_loader.ait_qa_data_loader import AITQaDataLoader
from ragworkbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragworkbench.datasets_loader.clap_nq_data_loader import ClapNqDataLoader
from ragworkbench.datasets_loader.da_code_data_loader import DaCodeDataLoader
from ragworkbench.datasets_loader.dabstep_data_loader import DabStepDataLoader
from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)
from ragworkbench.datasets_loader.dataset_names import DatasetName
from ragworkbench.datasets_loader.hotpot_qa_data_loader import HotpotQaDataLoader
from ragworkbench.datasets_loader.kramabench_data_loader import KramabenchDataLoader
from ragworkbench.datasets_loader.miniwiki_data_loader import MiniWikiDataLoader
from ragworkbench.datasets_loader.mldr_data_loader import MLDRDataLoader
from ragworkbench.datasets_loader.narrative_qa_data_loader import NarrativeQaDataLoader
from ragworkbench.datasets_loader.office_qa_data_loader import OfficeQADataLoader
from ragworkbench.datasets_loader.qasper_data_loader import QasperQaDataLoader
from ragworkbench.datasets_loader.real_mm_data_loader import RealMMRagDataLoader
from ragworkbench.datasets_loader.secque_data_loader import SecqueDataLoader
from ragworkbench.datasets_loader.watsonx_data_loader import WatsonxDocsQADataLoader

logger = logging.getLogger(__name__)


def data_loader(name: str):
    """
    Decorator to register a data loader with the DataLoaderFactory.

    Usage:
        @data_loader(name="my_dataset")
        class MyDataLoader(RagDataLoader):
            ...
    """

    def decorator(cls: type[RagDataLoader]) -> type[RagDataLoader]:
        DataLoaderFactory.register_loader(name, cls)
        return cls

    return decorator


class DataLoaderFactory:
    """
    Factory class for creating DataLoader instances based on dataset name.

    This factory provides a unified interface for instantiating any of the available
    RAG benchmark dataset loaders. It handles dataset name validation, parameter
    routing, and provides clear error messages for invalid inputs.

    The factory supports all built-in datasets and their specific parameters,
    while maintaining a consistent API across all loaders. Additionally, it allows
    external users to register custom data loaders for their own datasets.

    Attributes:
        _LOADER_REGISTRY: Internal mapping of DatasetName to built-in loader classes.
        _CUSTOM_LOADER_REGISTRY: Internal mapping of string names to custom loader classes.

    Example:
        Basic usage with built-in dataset:
        >>> from ragworkbench.datasets_loader import DataLoaderFactory
        >>> from ragworkbench.datasets_loader.dataset_names import DatasetName
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
        >>> from ragworkbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
        >>> loader = DataLoaderFactory.create_loader(
        ...     dataset_name=DatasetName.HOTPOT_QA,
        ...     split="train",
        ...     sampling_params=DataSamplingParams(
        ...         question_limit=100,
        ...         document_factor=2,
        ...         seed=42
        ...     )
        ... )

        Registering and using a custom loader:
        >>> from ragworkbench.datasets_loader import RagDataLoader
        >>> class MyCustomLoader(RagDataLoader):
        ...     def _get_documents(self):
        ...         return [...]  # Your implementation
        ...     def _get_benchmark_entries(self, split):
        ...         return [...]  # Your implementation
        >>>
        >>> # Register the custom loader
        >>> DataLoaderFactory.register_loader("my_dataset", MyCustomLoader)
        >>>
        >>> # Use it like any built-in dataset
        >>> loader = DataLoaderFactory.create_loader("my_dataset", split="train")
        >>> corpus = loader.get_corpus()
        >>> benchmark = loader.get_benchmark()
        >>>
        >>> # Check if a dataset is registered
        >>> DataLoaderFactory.is_registered("my_dataset")
        True
        >>>
        >>> # List all available datasets (built-in + custom)
        >>> datasets = DataLoaderFactory.list_available_datasets()
        >>>
        >>> # Unregister when done
        >>> DataLoaderFactory.unregister_loader("my_dataset")
        True

    Note:
        Some built-in loaders have specific parameters:
        - HotpotQA: `level` (Literal["easy", "medium", "hard"])
        - Kramabench: `verbose` (bool), `progress_every` (int)
        - MLDR: `language` (str)

        These can be passed as keyword arguments to create_loader().

        Custom loaders must:
        - Extend RagDataLoader
        - Implement _get_documents() and _get_benchmark_entries()
        - Use string names that don't conflict with built-in DatasetName values
    """

    # Registry mapping DatasetName to loader classes
    _LOADER_REGISTRY: dict[DatasetName, type[RagDataLoader]] = {
        DatasetName.AIT_QA: AITQaDataLoader,
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
        DatasetName.REAL_MM_FIN_SLIDES: RealMMRagDataLoader,
        DatasetName.REAL_MM_FIN_REPORT: RealMMRagDataLoader,
        DatasetName.REAL_MM_TECH_REPORT: RealMMRagDataLoader,
        DatasetName.REAL_MM_TECH_SLIDES: RealMMRagDataLoader,
        DatasetName.SECQUE: SecqueDataLoader,
        DatasetName.WATSONX_DOCS_QA_TXT: WatsonxDocsQADataLoader,
        DatasetName.WATSONX_DOCS_QA_HTML: WatsonxDocsQADataLoader,
        DatasetName.WATSONX_DOCS_QA_MD: WatsonxDocsQADataLoader,
    }

    # Custom registry for externally registered loaders (string-based keys)
    _CUSTOM_LOADER_REGISTRY: dict[str, type[RagDataLoader]] = {}

    @classmethod
    def register_loader(
        cls,
        dataset_name: str,
        loader_class: type[RagDataLoader],
    ) -> None:
        """
                Register a custom data loader class for a specific dataset name.

                This method allows external users to register their own RagDataLoader
                implementations for custom datasets. The registered loaders can then be
                used with create_loader() just like built-in datasets.
        g
                Args:
                    dataset_name: The unique identifier for the custom dataset. Must be
                                 a non-empty string that doesn't conflict with built-in
                                 dataset names (DatasetName enum values).
                    loader_class: The loader class to register. Must be a subclass of
                                 RagDataLoader (not the abstract class itself).

                Raises:
                    ValueError: If dataset_name is empty, None, or matches a built-in
                               dataset name from the DatasetName enum.
                    TypeError: If loader_class is not a class, not a subclass of
                              RagDataLoader, or is RagDataLoader itself.

                Example:
                    >>> from ragworkbench.datasets_loader import DataLoaderFactory, RagDataLoader
                    >>> class MyCustomLoader(RagDataLoader):
                    ...     def _get_documents(self):
                    ...         return [...]
                    ...     def _get_benchmark_entries(self, split):
                    ...         return [...]
                    >>> DataLoaderFactory.register_loader("my_dataset", MyCustomLoader)
                    >>> loader = DataLoaderFactory.create_loader("my_dataset", split="train")

                Note:
                    - If dataset_name already exists in the custom registry, it will be
                      overwritten with a warning logged.
                    - Cannot override built-in dataset names.
                    - The loader_class must implement _get_documents() and
                      _get_benchmark_entries() methods.
        """
        # Validate dataset_name is a non-empty string
        if not isinstance(dataset_name, str):
            raise TypeError(
                f"dataset_name must be a string, got {type(dataset_name).__name__}"
            )
        if not dataset_name or not dataset_name.strip():
            raise ValueError("dataset_name cannot be empty")

        # Check if dataset_name conflicts with built-in datasets
        if DatasetName.is_valid(dataset_name):
            available = ", ".join([name.value for name in cls._LOADER_REGISTRY.keys()])
            raise ValueError(
                f"Cannot register custom loader for '{dataset_name}' - this is a "
                f"built-in dataset name. Available built-in datasets: {available}"
            )

        # Validate loader_class is a class
        if not isinstance(loader_class, type):
            raise TypeError(
                f"loader_class must be a class, got {type(loader_class).__name__}"
            )

        # Validate loader_class is a subclass of RagDataLoader
        if not issubclass(loader_class, RagDataLoader):
            raise TypeError(
                f"Loader class must be a subclass of RagDataLoader, got {loader_class}"
            )

        # Prevent registering the abstract base class itself
        if loader_class is RagDataLoader:
            raise TypeError(
                "Cannot register RagDataLoader itself - it is an abstract class. "
                "Please register a concrete subclass."
            )

        # Warn if overwriting existing custom loader
        if dataset_name in cls._CUSTOM_LOADER_REGISTRY:
            logger.warning(
                f"Overwriting existing custom loader for dataset '{dataset_name}'"
            )

        # Register the loader
        cls._CUSTOM_LOADER_REGISTRY[dataset_name] = loader_class
        logger.debug(
            f"Registered custom loader {loader_class.__name__} for dataset '{dataset_name}'"
        )

    @classmethod
    def unregister_loader(cls, dataset_name: str) -> bool:
        """
        Remove a custom data loader from the registry.

        This method removes a previously registered custom loader. It only affects
        the custom registry and cannot remove built-in loaders.

        Args:
            dataset_name: The name of the custom dataset to unregister.

        Returns:
            True if the loader was found and removed, False if not found in
            the custom registry.

        Example:
            >>> DataLoaderFactory.register_loader("my_dataset", MyLoader)
            >>> DataLoaderFactory.unregister_loader("my_dataset")
            True
            >>> DataLoaderFactory.unregister_loader("my_dataset")
            False
            >>> # Cannot unregister built-in datasets
            >>> DataLoaderFactory.unregister_loader("bioasq")
            False

        Note:
            This method will not raise an error if the dataset is not found;
            it simply returns False.
        """
        if dataset_name in cls._CUSTOM_LOADER_REGISTRY:
            del cls._CUSTOM_LOADER_REGISTRY[dataset_name]
            logger.debug(f"Unregistered custom loader for dataset '{dataset_name}'")
            return True
        return False

    @classmethod
    def is_registered(cls, dataset_name: str) -> bool:
        """
        Check if a dataset name is registered (either built-in or custom).

        This method checks both the built-in registry (DatasetName enum) and
        the custom registry for the given dataset name.

        Args:
            dataset_name: The dataset name to check.

        Returns:
            True if the dataset is registered in either registry, False otherwise.

        Example:
            >>> # Check built-in dataset
            >>> DataLoaderFactory.is_registered("bioasq")
            True
            >>> # Check custom dataset
            >>> DataLoaderFactory.register_loader("my_dataset", MyLoader)
            >>> DataLoaderFactory.is_registered("my_dataset")
            True
            >>> # Check non-existent dataset
            >>> DataLoaderFactory.is_registered("unknown")
            False
        """
        # Check if it's a valid built-in dataset name
        if DatasetName.is_valid(dataset_name):
            return True

        # Check if it's in the custom registry
        return dataset_name in cls._CUSTOM_LOADER_REGISTRY

    @classmethod
    def create_loader(
        cls,
        dataset_name: DatasetName | str,
        split: DatasetSplit | None = None,
        sampling_params: DataSamplingParams | None = None,
        cache_dir: Path | None = None,
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
            cache_dir: Directory path for caching downloaded datasets. If None,
                      uses the default cache location. Default is None.
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
        # Determine which registry to use and get the loader class
        loader_class: type[RagDataLoader]
        resolved_dataset_name: DatasetName | str = dataset_name

        if isinstance(dataset_name, DatasetName):
            # DatasetName enum - use built-in registry
            if dataset_name not in cls._LOADER_REGISTRY:
                available = ", ".join(
                    [name.value for name in cls._LOADER_REGISTRY.keys()]
                )
                raise ValueError(
                    f"No loader registered for dataset '{dataset_name.value}'. "
                    f"Available datasets: {available}"
                )
            loader_class = cls._LOADER_REGISTRY[dataset_name]

        elif isinstance(dataset_name, str):
            # String - try built-in first, then custom
            try:
                # Try to convert to DatasetName enum (built-in dataset)
                dataset_name_enum = DatasetName.from_string(dataset_name)
                loader_class = cls._LOADER_REGISTRY[dataset_name_enum]
                resolved_dataset_name = dataset_name_enum
            except ValueError:
                # Not a built-in dataset, check custom registry
                if dataset_name in cls._CUSTOM_LOADER_REGISTRY:
                    loader_class = cls._CUSTOM_LOADER_REGISTRY[dataset_name]
                else:
                    # Not found in either registry
                    available_list = cls.list_available_datasets()
                    raise ValueError(
                        f"No loader registered for dataset '{dataset_name}'. "
                        f"Available datasets: {', '.join(available_list)}"
                    ) from None
        else:
            raise TypeError(
                f"dataset_name must be a DatasetName enum or string, "
                f"got {type(dataset_name).__name__}"
            )

        # Prepare constructor arguments
        # Default sampling_params if not provided
        if sampling_params is None:
            sampling_params = DataSamplingParams()

        # Build constructor arguments based on loader requirements
        constructor_args: dict[str, Any] = {
            "split": split,
            "cache_dir": cache_dir,
        }

        # RealMMRagDataLoader requires dataset_name as a constructor parameter
        if loader_class == RealMMRagDataLoader:
            constructor_args["dataset_name"] = resolved_dataset_name

        # WatsonxDocsQADataLoader requires document_format based on dataset_name
        if loader_class == WatsonxDocsQADataLoader:
            # Map dataset names to document formats
            if isinstance(resolved_dataset_name, DatasetName):
                if resolved_dataset_name == DatasetName.WATSONX_DOCS_QA_HTML:
                    constructor_args["document_format"] = "html"
                elif resolved_dataset_name == DatasetName.WATSONX_DOCS_QA_MD:
                    constructor_args["document_format"] = "markdown"
                else:  # DatasetName.WATSONX_DOCS_QA (default to text)
                    constructor_args["document_format"] = "text"
            else:
                # String-based dataset name
                if dataset_name == "watsonx_docs_qa_html":
                    constructor_args["document_format"] = "html"
                elif dataset_name == "watsonx_docs_qa_md":
                    constructor_args["document_format"] = "markdown"
                else:  # "watsonx_docs_qa" (default to text)
                    constructor_args["document_format"] = "text"

        constructor_args["sampling_params"] = sampling_params

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
        Return a list of all available dataset names (built-in and custom).

        This method returns a combined list of all registered datasets, including
        both built-in datasets (from DatasetName enum) and custom datasets
        registered via register_loader().

        Returns:
            List of dataset name strings that can be used with create_loader().
            Built-in datasets are listed first, followed by custom datasets
            (sorted alphabetically).

        Example:
            >>> datasets = DataLoaderFactory.list_available_datasets()
            >>> print(f"Available datasets: {', '.join(datasets)}")
            Available datasets: bioasq, clap_nq, da_code, ..., my_custom_dataset
            >>> # After registering a custom loader
            >>> DataLoaderFactory.register_loader("my_dataset", MyLoader)
            >>> datasets = DataLoaderFactory.list_available_datasets()
            >>> "my_dataset" in datasets
            True
        """
        # Get built-in dataset names
        builtin_names = [name.value for name in cls._LOADER_REGISTRY.keys()]

        # Get custom dataset names (sorted alphabetically)
        custom_names = sorted(cls._CUSTOM_LOADER_REGISTRY.keys())

        # Return combined list
        return builtin_names + custom_names

    @classmethod
    def get_loader_class(cls, dataset_name: DatasetName | str) -> type[RagDataLoader]:
        """
        Get the loader class for a given dataset name without instantiating it.

        This method is useful for introspection or when you need to access
        class-level attributes or methods before instantiation. It supports
        both built-in and custom datasets.

        Args:
            dataset_name: The dataset name. Can be a DatasetName enum value
                         or a string that matches a valid dataset name
                         (built-in or custom).

        Returns:
            The loader class (not an instance) for the specified dataset.

        Raises:
            ValueError: If dataset_name is not a valid dataset name.
            TypeError: If dataset_name is not a DatasetName enum or string.

        Example:
            >>> # Get built-in loader class
            >>> loader_class = DataLoaderFactory.get_loader_class("bioasq")
            >>> print(loader_class.__name__)
            BioasqDataLoader
            >>> # Get custom loader class
            >>> DataLoaderFactory.register_loader("my_dataset", MyLoader)
            >>> loader_class = DataLoaderFactory.get_loader_class("my_dataset")
            >>> print(loader_class.__name__)
            MyLoader
            >>> # Check if a loader has a specific method
            >>> hasattr(loader_class, '_get_documents')
            True
        """
        if isinstance(dataset_name, DatasetName):
            # DatasetName enum - use built-in registry
            if dataset_name not in cls._LOADER_REGISTRY:
                available = ", ".join(
                    [name.value for name in cls._LOADER_REGISTRY.keys()]
                )
                raise ValueError(
                    f"No loader registered for dataset '{dataset_name.value}'. "
                    f"Available datasets: {available}"
                )
            return cls._LOADER_REGISTRY[dataset_name]

        elif isinstance(dataset_name, str):
            # String - try built-in first, then custom
            try:
                # Try to convert to DatasetName enum (built-in dataset)
                dataset_name_enum = DatasetName.from_string(dataset_name)
                return cls._LOADER_REGISTRY[dataset_name_enum]
            except ValueError:
                # Not a built-in dataset, check custom registry
                if dataset_name in cls._CUSTOM_LOADER_REGISTRY:
                    return cls._CUSTOM_LOADER_REGISTRY[dataset_name]
                else:
                    # Not found in either registry
                    available_list = cls.list_available_datasets()
                    raise ValueError(
                        f"No loader registered for dataset '{dataset_name}'. "
                        f"Available datasets: {', '.join(available_list)}"
                    ) from None
        else:
            raise TypeError(
                f"dataset_name must be a DatasetName enum or string, "
                f"got {type(dataset_name).__name__}"
            )
