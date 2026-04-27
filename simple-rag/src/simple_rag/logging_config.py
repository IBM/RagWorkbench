"""Shared logging configuration for Simple RAG."""

import logging


def init_logger() -> None:
    """Initialize logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
