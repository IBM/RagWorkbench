"""Evaluation script for Simple RAG benchmarking.

This script runs the Simple RAG pipeline with the example board configuration
to benchmark performance on the AIT-QA dataset.
"""

import logging
from pathlib import Path

from dotenv import load_dotenv

# Import pipelines to register them with BoardRegistry
from simple_rag.inference_pipeline import SimpleRagInferencePipeline  # noqa: F401
from simple_rag.ingest_pipeline import SimpleRagIngestPipeline  # noqa: F401

from ragworkbench.boards.board_generator import BoardGenerator
from ragworkbench.logging_config import init_logger

logger = logging.getLogger(__name__)


def main() -> None:
    """Run Simple RAG evaluation with the example board."""

    init_logger()

    # Load environment variables from .env file
    load_dotenv(verbose=True)

    # Get the boards directory path
    boards_directory = Path(__file__).parent / "boards"

    # Create board generator with the simple_rag_example board
    board = BoardGenerator(board_path=boards_directory / "simple_rag_example")

    logger.info("Starting Simple RAG evaluation...")
    board.process()

    logger.info(f"Evaluation complete. Output written to '{board.output_path}'")


if __name__ == "__main__":
    main()
