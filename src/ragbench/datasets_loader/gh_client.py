from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import requests


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
