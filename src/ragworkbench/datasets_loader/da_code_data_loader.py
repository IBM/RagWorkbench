from __future__ import annotations

import io
import json
import logging
import os
import zipfile
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import gdown  # type: ignore[import-not-found]

from ragworkbench.api.dataset import DatasetSplit
from ragworkbench.datasets_loader import RagDataLoader
from ragworkbench.datasets_loader.data_models.data_sampling_params import (
    DataSamplingParams,
)
from ragworkbench.datasets_loader.data_models.document_object import DocumentObject
from ragworkbench.datasets_loader.data_models.rag_benchmark import (
    GroundTruthContextId,
    RagBenchmarkEntry,
)
from ragworkbench.datasets_loader.dataset_names import DatasetName
from ragworkbench.datasets_loader.datasets_utils import (
    GitHubClient,
    GitHubRef,
    encode_bytes,
    get_benchmark_split,
    guess_mime,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Helpers
# =============================================================================


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _iter_jsonl_rows_from_bytes(data: bytes) -> Iterable[dict[str, Any]]:
    """Robust JSONL parser; tolerates concatenated JSON objects on one line."""
    decoder = json.JSONDecoder()

    def parse_many(s: str) -> Iterable[Any]:
        i = 0
        n = len(s)
        while i < n:
            while i < n and s[i].isspace():
                i += 1
            if i >= n:
                return
            try:
                obj, end = decoder.raw_decode(s, i)
            except json.JSONDecodeError:
                return
            yield obj
            i = end

    for raw in data.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        text = raw.decode("utf-8", errors="replace")
        for obj in parse_many(text):
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item


# =============================================================================
# DA-Code Loader (GitHub + Drive via gdown)
# =============================================================================
class DaCodeDataLoader(RagDataLoader):
    """
    DA-Code loader without Hugging Face.

    Sources:
      - GitHub configs:
          tasks: da_code/configs/task/*.jsonl
          evals: da_code/configs/eval/*.jsonl
      - Google Drive:
          gold.zip (+ optional source.zip) via gdown

    Offline eval:
      Each RagBenchmarkEntry embeds task metadata, eval config, and gold artifacts (base64).
    """

    def __init__(
        self,
        split: DatasetSplit | None,
        cache_dir: Path | None = None,
        *,
        sampling_params: DataSamplingParams | None = None,
        github_token: str | None = None,
    ):
        self.gh = GitHubRef(repo="yiyihum/da-code", ref="main")
        self._gh_client = GitHubClient(token=github_token)

        self.task_cfg_dir = "da_code/configs/task".strip("/")
        self.eval_cfg_dir = "da_code/configs/eval".strip("/")

        self.gold_drive_file_id = "1WxcrijbCgdHzFSSSt2HVlkJqQrBWQ2IL"
        self.source_drive_file_id = "1eM_FVT1tlY4XXp6b7TrKzgTWOvskrjTs"

        download_cache_dir = Path.home() / ".cache" / "da_code"
        self.download_cache_dir = Path(download_cache_dir)
        self.download_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using cache dir: {cache_dir}")

        # Fail fast with a helpful error if paths are wrong
        _ = self._gh_client.list_dir(self.gh, self.task_cfg_dir)
        _ = self._gh_client.list_dir(self.gh, self.eval_cfg_dir)

        super().__init__(
            dataset_name=DatasetName.DA_CODE,
            split=split,
            sampling_params=sampling_params or DataSamplingParams(),
            cache_dir=cache_dir,
        )

    # -------------------------------------------------------------------------
    # Drive downloads (gdown) + caching
    # -------------------------------------------------------------------------
    def _cached_path(self, name: str) -> Path:
        return self.download_cache_dir / name

    def _ensure_drive_file(self, *, name: str, drive_file_id: str) -> Path:
        path = self._cached_path(name)
        if path.exists() and path.stat().st_size > 0:
            return path

        logger.info(f"Downloading {name} via gdown -> {path}")
        url = f"https://drive.google.com/uc?id={drive_file_id}"
        out = gdown.download(url=url, output=str(path), quiet=True, fuzzy=True)
        if not out or not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"gdown failed: name={name}, file_id={drive_file_id}")
        return path

    # -------------------------------------------------------------------------
    # GitHub config listing/reading
    # -------------------------------------------------------------------------
    def _list_jsonl_files(self, dir_path: str) -> list[str]:
        entries = self._gh_client.list_dir(self.gh, dir_path)
        paths: list[str] = []
        for e in entries:
            if e.get("type") == "file" and (e.get("name") or "").endswith(".jsonl"):
                p = e.get("path")
                if p:
                    paths.append(p)
        return sorted(paths)

    def _read_jsonl_rows(self, repo_path: str) -> Iterable[dict[str, Any]]:
        data = self._gh_client.read_file(self.gh, repo_path)
        yield from _iter_jsonl_rows_from_bytes(data)

    # -------------------------------------------------------------------------
    # Gold indexing
    # -------------------------------------------------------------------------
    @staticmethod
    def _index_gold(gold_zip_path: Path) -> dict[str, list[zipfile.ZipInfo]]:
        """
        Index gold.zip members by task_id.

        Expected layouts seen in DA-Code zips:
          - gold/<task_id>/<file...>
          - source/<task_id>/<file...>

        Returns:
          { "<task_id>": [ZipInfo, ...], ... }
        """
        index: dict[str, list[zipfile.ZipInfo]] = {}

        with zipfile.ZipFile(gold_zip_path) as zf:
            for zi in zf.infolist():
                if zi.is_dir():
                    continue

                parts = zi.filename.split("/")
                if len(parts) < 3:
                    continue

                root, task_id = parts[0], parts[1]
                if root not in {"gold", "source"}:
                    continue

                index.setdefault(task_id, []).append(zi)
        return index

    # -------------------------------------------------------------------------
    # Benchmark entries
    # -------------------------------------------------------------------------
    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        skip_names = {".DS_Store", "DS_Store"}

        # 0) docs: task_id -> [doc.name]
        task_id_to_gold_doc_ids: dict[str, list[str]] = defaultdict(list)
        for doc in self.all_docs:
            task_id_to_gold_doc_ids[str(doc.metadata["task_id"])].append(doc.name)

        # 1) task_id_to_task_cfgs: task_id -> task bundle
        task_id_to_task_cfgs: dict[str, dict[str, Any]] = {}
        for tp in (
            p for p in self._list_jsonl_files(self.task_cfg_dir) if _stem(p) != "all"
        ):
            family = _stem(tp)
            for i, row in enumerate(self._read_jsonl_rows(tp)):
                tid = str(row.get("id") or row.get("task_id") or f"{family}:{i}")
                if tid in task_id_to_task_cfgs:
                    raise ValueError(f"Duplicate task_id: {tid}")
                row["family"] = family
                task_id_to_task_cfgs[tid] = {"task_config": row}

        # 2) task_id_to_eval_cfgs: task_id -> eval row
        task_id_to_eval_cfgs: dict[str, dict[str, Any]] = {}
        for ep in self._list_jsonl_files(self.eval_cfg_dir):
            if _stem(ep) in {"eval_all", "all"}:
                continue
            for row in self._read_jsonl_rows(ep):
                key = (
                    row.get("id")
                    or row.get("task_id")
                    or row.get("question_id")
                    or row.get("qid")
                )
                if key is not None:
                    task_id_to_eval_cfgs[str(key)] = row

        # 3) task_id_to_gold_files: task_id -> gold payload
        gold_zip_path = self._ensure_drive_file(
            name="gold.zip", drive_file_id=self.gold_drive_file_id
        )
        gold_index = self._index_gold(gold_zip_path)

        task_id_to_gold_files: dict[str, dict[str, Any]] = {}
        with zipfile.ZipFile(gold_zip_path) as zf:
            for tid, members in gold_index.items():
                files, total = [], 0
                for zi in sorted(members, key=lambda z: z.filename):
                    fname = os.path.basename(zi.filename)
                    if fname in skip_names:
                        continue

                    raw = zf.read(zi.filename)
                    total += len(raw)
                    files.append(
                        {
                            "path": zi.filename,
                            "name": fname,
                            "mime_type": guess_mime(fname),
                            "size_bytes": zi.file_size,
                            "b64": encode_bytes(raw),
                        }
                    )
                task_id_to_gold_files[tid] = {
                    "count": len(files),
                    "total_bytes": total,
                    "files": files,
                }

        # 4) validate same IDs (no missing, no extras) incl docs
        task_ids, eval_ids, gold_ids, doc_ids = (
            set(task_id_to_task_cfgs),
            set(task_id_to_eval_cfgs),
            set(task_id_to_gold_files),
            set(task_id_to_gold_doc_ids),
        )
        if task_ids != eval_ids or task_ids != gold_ids or task_ids != doc_ids:
            raise ValueError(
                "Task ID mismatch:\n"
                f"missing_in_eval={sorted(task_ids - eval_ids)[:20]} extra_in_eval={sorted(eval_ids - task_ids)[:20]}\n"
                f"missing_in_gold={sorted(task_ids - gold_ids)[:20]} extra_in_gold={sorted(gold_ids - task_ids)[:20]}\n"
                f"missing_in_docs={sorted(task_ids - doc_ids)[:20]} extra_in_docs={sorted(doc_ids - task_ids)[:20]}"
            )

        # 5) simple join -> out
        out: list[RagBenchmarkEntry] = []
        for tid in sorted(task_ids):
            ground_truth_context_ids: list[GroundTruthContextId] = []
            for filename in task_id_to_gold_doc_ids[tid]:
                ground_truth_context_ids.append(
                    GroundTruthContextId(document_id=filename)
                )

            out.append(
                RagBenchmarkEntry(
                    question_id=tid,
                    question=str(
                        task_id_to_task_cfgs[tid]["task_config"]["instruction"]
                    ),
                    ground_truth_answers=[],
                    ground_truths_context_ids=ground_truth_context_ids,
                    additional_information={
                        "task_config": task_id_to_task_cfgs[tid]["task_config"],
                        "eval_config": task_id_to_eval_cfgs[tid],
                        "gold_answer_documents": task_id_to_gold_files[tid],
                    },
                )
            )

            if len(out) % 50 == 0:
                logger.info(f"Loaded {len(out)} task_id_to_task_cfgs (last={tid})")

        logger.info(f"DONE — {len(out)} benchmark entries ready")
        return get_benchmark_split(benchmark_entries=out, split=split)

    # -------------------------------------------------------------------------
    # Documents (optional corpus)
    # -------------------------------------------------------------------------
    def _get_documents(self) -> list[DocumentObject]:
        """
        Loads allowed files from source.zip.
        """
        source_path = self._ensure_drive_file(
            name="source.zip", drive_file_id=self.source_drive_file_id
        )
        docs: list[DocumentObject] = []

        with zipfile.ZipFile(source_path) as zf:
            members = [zi for zi in zf.infolist() if not zi.is_dir()]
            total = len(members)
            logger.info(f"Loading source.zip corpus: {total} files")

            for i, zi in enumerate(members, 1):
                file_str = zi.filename
                raw = zf.read(file_str)

                file_path = Path(file_str)
                filename = file_path.name
                task_id = file_path.parts[1]
                if task_id == "dm-source":
                    continue

                file_name_with_task = Path(*file_path.parts[1:]).as_posix()

                docs.append(
                    DocumentObject(
                        stream=io.BytesIO(raw),
                        name=file_name_with_task,
                        mime_type=guess_mime(filename),
                        metadata={
                            "filename": filename,
                            "file_path": file_str,
                            "task_id": task_id,
                        },
                    )
                )
                # progress every 100 files or at end
                if i % 100 == 0 or i == total:
                    pct = (i / total) * 100
                    logger.info(f"Progress: {i}/{total} ({pct:.1f}%)")

        logger.info(f"Source corpus loaded, total={total}")

        return docs
