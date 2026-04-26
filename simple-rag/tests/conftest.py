"""Pytest configuration for simple-rag tests."""

import logging
import sys


def pytest_configure(config):
    """Configure pytest logging to show logs in console.

    Uses the same format as RagWorkbench's logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging configured for simple-rag tests")


# Made with Bob
