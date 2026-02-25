import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yaml

from ragbench import DataLoaderFactory, Experiment
from ragbench.boards.board_model import Board
from ragbench.boards.board_registry import BoardRegistry
from ragbench.eval import (
    MetricDefinition,
    MetricDefinitionsConfig,
    load_metric_definitions,
)

logger = logging.getLogger(__name__)


class BoardGenerator:

    BOARD_YAML = "board.yaml"
    RESULTS_CSV = "results.csv"

    def __init__(self, board_path: Path):
        self.input_path = board_path
        self.output_path = board_path / "output"

        yaml_file = self.input_path / self.BOARD_YAML
        with yaml_file.open("r") as f:
            data = yaml.full_load(f)

        self.board = Board(**data)
        self.pipelines = []
        for config in self.board.configurations:
            # create the ingest instance
            ingest_pipeline = BoardRegistry.create_ingest_pipeline(
                name=config.ingest.name, params=config.ingest.params
            )

            # create the inference instance
            inference_pipeline = BoardRegistry.create_inference_pipeline(
                name=config.inference.name, params=config.inference.params
            )

            # save for later
            self.pipelines.append((ingest_pipeline, inference_pipeline))
            metric_definition_config: MetricDefinitionsConfig = (
                load_metric_definitions()
            )

            self.metric_definitions: list[MetricDefinition] = [
                metric_definition_config.get_metric_definition(metric_name)
                for metric_name in self.board.metrics
            ]

        self.results = pd.DataFrame()

    def process(self) -> None:
        # iterate over configurations and then over datasets
        for config_seq, config in enumerate(self.board.configurations):
            logger.info(
                f"Running configuration: {config.name} ({config_seq+1}/{len(self.board.configurations)})"
            )
            ingest_pipeline, inference_pipeline = self.pipelines[config_seq]
            for dataset_seq, dataset in enumerate(self.board.datasets):
                logger.info(
                    f"Running dataset: {dataset.name} ({dataset_seq+1}/{len(self.board.datasets)})"
                )

                data_loader = DataLoaderFactory.create_loader(
                    dataset_name=dataset.name,
                    split=dataset.split,
                    sampling_params=dataset.sampling,
                )

                experiment = Experiment(
                    name=f"experiment_{config_seq}_{dataset_seq}",
                    data_loader=data_loader,
                    ingest_pipeline=ingest_pipeline,
                    inference_pipeline=inference_pipeline,
                    eval_metrics=self.metric_definitions,
                )

                experiment.run()

        #             df = pd.read_csv(results_path)
        #             df["board_configuration_seq"] = config_seq
        #             df["board_dataset_seq"] = dataset_seq
        #             df["board_configuration_name"] = config.name
        #             df["board_dataset_id"] = dataset.id
        #             df["board_experiment_yaml"] = yaml.dump(full_config, sort_keys=False)
        #             df["board_experiment_id"] = f"exp_{config_seq}_{dataset_seq}"
        #
        #             self.results.append(df)
        #
        self.results = pd.DataFrame()

    def export_results(self) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        self.results.to_csv(self.output_path / self.RESULTS_CSV, index=False)

    def load_results(self) -> None:
        self.results = pd.read_csv(self.output_path / self.RESULTS_CSV)

    # def get_board_results(self) -> BoardResults:
    #     return BoardResults(board=self.board, results=self.results)

    def clean_output(self):
        path = self.output_path / "output"
        if os.path.exists(path):
            shutil.rmtree(path)

    def clean_cache(self):
        path = self.output_path / "cache"
        if os.path.exists(path):
            shutil.rmtree(path)

    def export_md(self):

        os.makedirs(self.output_path, exist_ok=True)

        # define the structure of the markdown
        md_struct = [
            (1, self.board.name, lambda x: x.description),
            (2, "Results", self.serialize_results),
            (2, "Configurations", self.serialize_configs),
            (2, "Datasets", self.serialize_datasets),
        ]

        # create the md
        md = "\n".join(
            f"{'#' * times} {key}\n" f"{value(self.board)}\n"
            for times, key, value in md_struct
        )

        # save it
        os.makedirs(self.output_path, exist_ok=True)
        with open(self.output_path / "board.md", "w") as f:
            f.write(md)

    @classmethod
    def tuples_to_md_table(
        cls,
        rows: list[Sequence],
        headers: list[str],
    ) -> str:
        rows = list(rows)
        if not rows:
            raise ValueError("rows must not be empty")

        width = len(rows[0])
        if len(headers) != width:
            raise ValueError("headers length must match tuple size")

        if any(len(row) != width for row in rows):
            raise ValueError("all tuples must be of equal length")

        def esc(x):
            return str(x).replace("|", "\\|").replace("\n", "<br>")

        lines = [
            "| " + " | ".join(map(esc, headers)) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]

        for row in rows:
            lines.append("| " + " | ".join(esc(v) for v in row) + " |")

        return "\n".join(lines)

    @classmethod
    def serialize_datasets(cls, board: Board):
        return cls.tuples_to_md_table(
            headers=["Name", "Question Sample Size", "Document Sample Factor", "Split"],
            rows=[
                (d.name, d.sampling.question_limit, d.sampling.document_factor, d.split)
                for d in board.datasets
            ],
        )

    @classmethod
    def serialize_configs(cls, board: Board):
        return cls.tuples_to_md_table(
            headers=["Name", "Description"],
            rows=[(c.name, c.description) for c in board.configurations],
        )

    def serialize_results(self, board: Board):
        md = ""
        for screen in board.report.screens:
            config_row_span = 1 + (1 if len(screen.scores) > 1 else 0)
            md += f"### {screen.title}\n"
            md += "<table>\n"
            md += "<tr>\n"
            md += (
                "\n".join(
                    [f'\t<th rowspan="{config_row_span}">Configuration</th>']
                    + [
                        f'\t<th colspan="{len(screen.scores)}">{dataset.name}</th>'
                        for dataset in board.datasets
                    ]
                )
                + "\n"
            )
            md += "</tr>\n"
            if len(screen.scores) > 1:
                md += "<tr>\n"
                for _ in self.board.datasets:
                    for score_title in screen.scores.values():
                        md += f"\t<td>{score_title}</td>\n"
                md += "</tr>\n"
            for config_seq, config in enumerate(self.board.configurations):
                md += "<tr>\n"
                md += f"\t<td>{config.name}</td>\n"
                for dataset_seq, _dataset in enumerate(self.board.datasets):
                    for score in screen.scores.keys():
                        df = self.results[
                            (self.results["board_configuration_seq"] == config_seq)
                            & (self.results["board_dataset_seq"] == dataset_seq)
                        ]
                        md += f"\t<td>{df.iloc[0][score]:.2f}</td>\n"
                md += "</tr>\n"
            md += "</table>\n\n"
        return md
