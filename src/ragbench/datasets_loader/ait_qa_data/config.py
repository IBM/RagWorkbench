"""Configuration module for AIT QA dataset paths.

This module provides centralized path configuration for the AIT QA dataset,
ensuring both the dataset downloader and data loader use the same location.
This is especially important when ragbench is used as a library dependency.

The module requires:
    - RAGBENCH_DATA_DIR environment variable to be set (can be defined in .env file)

Example:
    >>> import os
    >>> os.environ['RAGBENCH_DATA_DIR'] = '/path/to/data'
    >>> from ragbench.datasets_loader.ait_qa_data.config import get_ait_qa_documents_dir
    >>> docs_dir = get_ait_qa_documents_dir()
    >>> print(docs_dir)
    /path/to/data/ait_qa_pdf/documents

    Or using .env file:
    >>> # Create a .env file in your project root with:
    >>> # RAGBENCH_DATA_DIR=/path/to/data
    >>> from ragbench.datasets_loader.ait_qa_data.config import get_ait_qa_documents_dir
    >>> docs_dir = get_ait_qa_documents_dir()
"""

import os
from pathlib import Path

from dotenv import load_dotenv  # type: ignore[import-not-found]

# Load environment variables from .env file if it exists
# This will not override existing environment variables
load_dotenv()


def get_ait_qa_data_dir() -> Path:
    """Get the AIT QA dataset root directory location.

    This function determines where the AIT QA dataset should be stored by
    reading the RAGBENCH_DATA_DIR environment variable. The AIT QA data
    will be stored in a subdirectory 'ait_qa_pdf' within that location.

    The environment variable can be set either:
    1. As a system environment variable
    2. In a .env file in the project root or current directory

    The function does NOT create the directory - that's the responsibility
    of the caller (either the downloader or data loader).

    Returns:
        Path object pointing to the ait_qa_pdf directory.

    Raises:
        EnvironmentError: If RAGBENCH_DATA_DIR environment variable is not set
                         (neither in environment nor in .env file).

    Example:
        >>> import os
        >>> os.environ['RAGBENCH_DATA_DIR'] = '/custom/data/path'
        >>> get_ait_qa_data_dir()
        PosixPath('/custom/data/path/ait_qa_pdf')

        Or create a .env file:
        >>> # .env file content:
        >>> # RAGBENCH_DATA_DIR=/custom/data/path
        >>> get_ait_qa_data_dir()
        PosixPath('/custom/data/path/ait_qa_pdf')
    """
    data_dir = os.getenv("RAGBENCH_DATA_DIR")
    if not data_dir:
        raise OSError(
            "RAGBENCH_DATA_DIR environment variable is not set. "
            "Please set it to specify where the AIT QA dataset should be stored.\n"
            "You can either:\n"
            "1. Set it as an environment variable: export RAGBENCH_DATA_DIR=/path/to/data\n"
            "2. Create a .env file in your project root with: RAGBENCH_DATA_DIR=/path/to/data"
        )
    return Path(data_dir) / "ait_qa_pdf"


def get_ait_qa_documents_dir() -> Path:
    """Get the AIT QA documents directory containing PDF files.

    This is a convenience function that returns the 'documents' subdirectory
    within the AIT QA data directory. This is where the actual PDF files
    are stored.

    Returns:
        Path object pointing to the documents subdirectory.

    Example:
        >>> get_ait_qa_documents_dir()
        PosixPath('/path/to/data/ait_qa_pdf/documents')
    """
    return get_ait_qa_data_dir() / "documents"


# Made with Bob
