"""Pytest configuration for simple-rag tests."""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv


def pytest_configure(config):
    """Configure pytest logging and load environment variables.

    Uses the same format as RagWorkbench's logging configuration.
    """
    # Load .env file from simple-rag directory
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path, verbose=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging configured for simple-rag tests")
