"""
Utility functions for dataset operations.

This module provides helper functions for working with RAG benchmark datasets,
including splitting datasets into train/test sets with reproducible randomization.
"""

from __future__ import annotations

import base64
import mimetypes
import random
from dataclasses import dataclass
from typing import Any, cast

import requests

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.datasets_loader.data_models.rag_benchmark import RagBenchmarkEntry

# Default train/test split ratio
DEFAULT_TRAIN_RATIO = 0.7
# Default random seed for reproducible splits
DEFAULT_SPLIT_SEED = 42


def get_benchmark_split(
    benchmark_entries: list[RagBenchmarkEntry],
    split: DatasetSplit | None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    seed: int = DEFAULT_SPLIT_SEED,
) -> list[RagBenchmarkEntry]:
    """
    Split benchmark entries into train or test sets.

    This function creates a reproducible train/test split of benchmark entries
    using a fixed random seed. The split is performed by shuffling the entries
    and dividing them according to the specified train ratio.

    Args:
        benchmark_entries: List of benchmark entries to split.
        split: Which split to return - "train", "test", or None for all entries.
        train_ratio: Proportion of entries to use for training (default: 0.7).
        seed: Random seed for reproducible shuffling (default: 42).

    Returns:
        List of benchmark entries for the requested split:
        - If split is None: returns all entries unchanged
        - If split is "train": returns first train_ratio% of shuffled entries
        - If split is "test": returns remaining entries after train split

    Raises:
        ValueError: If train_ratio is not between 0 and 1.

    Example:
        >>> entries = [entry1, entry2, entry3, entry4, entry5]
        >>> train = get_benchmark_split(entries, split="train", train_ratio=0.6)
        >>> test = get_benchmark_split(entries, split="test", train_ratio=0.6)
        >>> len(train) + len(test) == len(entries)
        True
    """
    # Return all entries if no split is requested
    if split is None:
        return benchmark_entries

    # Validate train_ratio
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    # Create reproducible shuffle
    rng = random.Random(seed)
    shuffled = benchmark_entries.copy()
    rng.shuffle(shuffled)

    # Calculate split index
    split_idx = int(train_ratio * len(shuffled))

    # Return appropriate split
    if split == "train":
        return shuffled[:split_idx]
    else:  # split == "test"
        return shuffled[split_idx:]


def encode_bytes(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def guess_mime(name: str) -> str:
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


@dataclass(frozen=True)
class GitHubRef:
    repo: str
    ref: str = "main"


class GitHubClient:
    """List directory contents via GitHub API + fetch raw bytes via raw.githubusercontent.com."""

    def __init__(
        self, token: str | None = None, session: requests.Session | None = None
    ):
        self._s = session or requests.Session()
        self._s.headers.update({"User-Agent": "da-code-loader"})
        if token:
            self._s.headers.update({"Authorization": f"Bearer {token}"})

    def list_dir(self, gh: GitHubRef, path: str) -> list[dict[str, Any]]:
        api = f"https://api.github.com/repos/{gh.repo}/contents/{path.lstrip('/')}"
        r = self._s.get(api, params={"ref": gh.ref}, timeout=60)
        if r.status_code == 404:
            raise FileNotFoundError(f"GitHub path not found: {gh.repo}@{gh.ref}:{path}")
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and data.get("type") == "file":
            return [data]
        if not isinstance(data, list):
            raise ValueError(f"Unexpected GitHub API response for {api}: {type(data)}")
        return data

    def read_file(self, gh: GitHubRef, path: str) -> bytes:
        url = f"https://raw.githubusercontent.com/{gh.repo}/{gh.ref}/{path.lstrip('/')}"
        r = self._s.get(url, timeout=120)
        if r.status_code == 404:
            raise FileNotFoundError(f"GitHub file not found: {gh.repo}@{gh.ref}:{path}")
        r.raise_for_status()
        return cast(bytes, r.content)
