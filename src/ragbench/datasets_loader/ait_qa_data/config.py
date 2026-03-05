"""Configuration module for AIT QA dataset paths.

This module provides centralized path configuration for the AIT QA dataset,
ensuring both the dataset downloader and data loader use the same location.
This is especially important when ragbench is used as a library dependency.

The module supports:
    - Environment variable override (RAGBENCH_DATA_DIR)
    - Default location in user's home directory (~/.ragbench/)
    - Consistent paths across download and loading operations

Example:
    >>> from ragbench.datasets_loader.ait_qa_benchmark.config import get_ait_qa_documents_dir
    >>> docs_dir = get_ait_qa_documents_dir()
    >>> print(docs_dir)
    /Users/username/.ragbench/ait_qa_pdf/documents
"""

import os
from pathlib import Path


def get_ait_qa_data_dir() -> Path:
    """Get the AIT QA dataset root directory location.

    This function determines where the AIT QA dataset should be stored,
    following this priority order:
    1. RAGBENCH_DATA_DIR environment variable (if set) + /ait_qa_pdf
    2. User's home directory: ~/.ragbench/ait_qa_pdf (default)

    The function does NOT create the directory - that's the responsibility
    of the caller (either the downloader or data loader).

    Returns:
        Path object pointing to the ait_qa_pdf directory.

    Example:
        >>> # With environment variable set
        >>> os.environ['RAGBENCH_DATA_DIR'] = '/custom/data/path'
        >>> get_ait_qa_data_dir()
        PosixPath('/custom/data/path/ait_qa_pdf')

        >>> # Without environment variable (default)
        >>> get_ait_qa_data_dir()
        PosixPath('/Users/username/.ragbench/ait_qa_pdf')
    """
    if data_dir := os.getenv("RAGBENCH_DATA_DIR"):
        return Path(data_dir) / "ait_qa_pdf"
    return Path.home() / ".ragbench" / "ait_qa_pdf"


def get_ait_qa_documents_dir() -> Path:
    """Get the AIT QA documents directory containing PDF files.

    This is a convenience function that returns the 'documents' subdirectory
    within the AIT QA data directory. This is where the actual PDF files
    are stored.

    Returns:
        Path object pointing to the documents subdirectory.

    Example:
        >>> get_ait_qa_documents_dir()
        PosixPath('/Users/username/.ragbench/ait_qa_pdf/documents')
    """
    return get_ait_qa_data_dir() / "documents"


# Made with Bob
