import fnmatch
import gzip
import io
import json
import logging
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

from huggingface_hub import HfFileSystem  # type: ignore[import-not-found]

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
from ragworkbench.datasets_loader.datasets_utils import get_benchmark_split

REPO_ID = "eugenie-y/KramaBench"
SEED = 42
NUMBER_OF_THREADS = 5

logger = logging.getLogger(__name__)


class KramabenchDataLoader(RagDataLoader):
    def __init__(
        self,
        split: DatasetSplit | None = None,
        sampling_params: DataSamplingParams | None = None,
        cache_dir: Path | None = None,
        verbose: bool = True,
        progress_every: int = 50,
    ):
        if sampling_params is None:
            sampling_params = DataSamplingParams()

        self.verbose = verbose
        self.progress_every = max(1, int(progress_every))

        super().__init__(
            dataset_name=DatasetName.KRAMABENCH,
            split=split,
            sampling_params=sampling_params,
            cache_dir=cache_dir,
        )

    @staticmethod
    def _filter_paths(all_docs: list[str], pattern: str) -> list[str]:
        return [p for p in all_docs if fnmatch.fnmatch(p, pattern)]

    def _get_documents(self) -> list[DocumentObject]:
        base = f"hf://datasets/{REPO_ID}/data"
        # Deterministic order
        paths_str: list[str] = sorted(
            p
            for p in HfFileSystem().glob(f"{base}/**/*")
            if not p.endswith("/")
            and not p.endswith("resources.txt")
            and Path(p).suffix
        )

        total = len(paths_str)

        if self.verbose:
            logger.info(f"[Kramabench] Loading documents from: {base}")
            logger.info(f"[Kramabench] Found {total} document files")
            logger.info(
                f"[Kramabench] Reading up to {NUMBER_OF_THREADS} files in parallel"
            )

        def load_one(path: str) -> tuple[str, DocumentObject]:
            fs = HfFileSystem()  # thread-safe: one FS per worker
            with fs.open(path, "rb") as f:
                content = f.read()

            # We have to keep all the path from input/ as document name.
            name = path.split("/input/", 1)[1]
            # Secondly, we have to remove some prefix
            name = name.removeprefix("csn-data-book-2024-csv/CSVs/")
            mime_type = mimetypes.guess_type(name)[0] or "application/octet-stream"

            doc_obj = DocumentObject(
                name=name,
                mime_type=mime_type,
                stream=io.BytesIO(content),
                metadata={"path": path},
            )
            return path, doc_obj

        # Submit all jobs
        results: dict[str, DocumentObject] = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=NUMBER_OF_THREADS) as executor:
            future_to_path = {
                executor.submit(load_one, path): path for path in paths_str
            }

            for future in as_completed(future_to_path):
                path, doc = future.result()
                results[path] = doc
                completed += 1

                if self.verbose and (
                    completed == 1
                    or completed % self.progress_every == 0
                    or completed == total
                ):
                    logger.info(f"[Kramabench] Documents loaded: {completed}/{total}")

        # Preserve deterministic order
        documents = [results[path] for path in paths_str]
        return documents

    @staticmethod
    def _is_gzip_file(first_bytes: bytes) -> bool:
        # Gzip files start with magic bytes 0x1f 0x8b
        return len(first_bytes) >= 2 and first_bytes[:2] == b"\x1f\x8b"

    @staticmethod
    def _local_fixes(gt_str: str) -> str | None:
        to_ignore_gts = {
            "STORM-AI/warmup/v2/Sat_Density/swarma-wu334-20161022_to_20161024.csv",
            "WeatherEvents_Jan2016-Dec2022.csv",
            "STORM-AI/warmup/v2/Sat_Density/swarma-wu335-20161025_to_20161029.csv",
            "STORM-AI/warmup/v2/OMNI2/omni2-wu334-20161022_to_20161024.csv",
            "STORM-AI/warmup/v2/GOES/goes-wu334-20161022_to_20161024.csv",
            "STORM-AI/warmup/v2/GOES/goes-wu335-20161025_to_20161029.csv",
            "STORM-AI/warmup/v2/OMNI2/omni2-wu335-20161025_to_20161029.csv",
            "ZHVI.csv",
        }
        if (
            gt_str == "all csv in State MSA Identity Theft data/"
            or gt_str == "State MSA Identity Theft Data/"
        ):
            return "State MSA Identity Theft data/*.csv"
        elif gt_str == "m-street-beach-datasheet.csv":
            return "m_street_beach_datasheet.csv"
        elif gt_str == "carson-beach-datasheet.csv":
            return "carson_beach_datasheet.csv"
        elif gt_str == "omni2_low_res/omni2.txt":
            return "omni2_low_res/omni2.text"
        elif gt_str in to_ignore_gts:
            return None
        return gt_str

    def _get_benchmark_entries(
        self, split: DatasetSplit | None
    ) -> list[RagBenchmarkEntry]:
        all_docs_names: list[str] = [d.name for d in self.all_docs]
        benchmark_entries: list[RagBenchmarkEntry] = []
        base = f"hf://datasets/{REPO_ID}/workload"
        fs = HfFileSystem()

        excluded = {"legal-tiny.json", "quick-start-questions.json"}

        paths_str: list[str] = sorted(
            p
            for p in fs.glob(f"{base}/*")
            if p.endswith(".json") and Path(p).name not in excluded
        )

        total_files = len(paths_str)

        if self.verbose:
            logger.info(f"[Kramabench] Loading benchmark entries from: {base}")
            logger.info(
                f"[Kramabench] Found {len(paths_str)} workload JSON files (excluded: {sorted(excluded)})"
            )

        for file_i, path in enumerate(paths_str, start=1):

            with fs.open(path, "rb") as src:
                # Peek the first few bytes to decide gzip vs plain
                # Use a buffered reader so we can unread the peeked bytes
                buffered = io.BufferedReader(src)
                head = buffered.peek(
                    4
                )  # returns up to 4 bytes without advancing position

                if self._is_gzip_file(head):
                    # Wrap in gzip and decode as text
                    with gzip.GzipFile(fileobj=buffered, mode="rb") as gz:
                        text_stream = io.TextIOWrapper(
                            cast(io.BufferedReader, gz), encoding="utf-8"
                        )
                        json_contents = json.load(text_stream)
                else:
                    # Plain JSON, decode directly
                    text_stream = io.TextIOWrapper(buffered, encoding="utf-8")
                    json_contents = json.load(text_stream)

                # Some datasets store list[dict]; this assumes that shape
                for json_content in json_contents:
                    ground_truth_context_id_lst: list[GroundTruthContextId] = []
                    for gt in json_content["data_sources"]:
                        gt = self._local_fixes(gt)
                        if gt is not None:
                            if "?" in gt or "*" in gt:
                                # it is a glob pattern
                                pattern = gt
                                gt_docs = self._filter_paths(all_docs_names, pattern)
                            else:
                                gt_docs = [gt]
                            for gt_doc in gt_docs:
                                if gt_doc is not None:
                                    ground_truth_context_id_lst.append(
                                        GroundTruthContextId(document_id=gt_doc)
                                    )
                    # We can have a case of no ground-truth! In this case, we just skip this entry
                    if len(ground_truth_context_id_lst) > 0:
                        benchmark_entries.append(
                            RagBenchmarkEntry(
                                question_id=json_content["id"],
                                question=json_content["query"],
                                ground_truth_answers=[str(json_content["answer"])],
                                ground_truths_context_ids=ground_truth_context_id_lst,
                                is_answerable=True,
                                additional_information={
                                    "answer_type": json_content.get("answer_type"),
                                    "sub_tasks": json_content.get("subtasks"),
                                    "source_file": Path(path).name,
                                },
                            )
                        )

                if self.verbose and (
                    len(benchmark_entries) == 1
                    or len(benchmark_entries) % self.progress_every == 0
                ):
                    logger.info(
                        f"[Kramabench] Entries loaded: {len(benchmark_entries)} (latest file: {Path(path).name})"
                    )

            if self.verbose:
                logger.info(
                    f"[Kramabench] Workload files processed: {file_i}/{total_files} ({Path(path).name})"
                )

        if self.verbose:
            logger.info(
                f"[Kramabench] Total benchmark entries loaded: {len(benchmark_entries)}"
            )

        return get_benchmark_split(benchmark_entries=benchmark_entries, split=split)
