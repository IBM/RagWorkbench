import json
import logging
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import yaml
from pandas import DataFrame

from ragbench import DataLoaderFactory, Experiment
from ragbench.api.inference_result import InferenceResult
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
        self.input_path: Path = board_path
        self.output_path: Path = board_path / "output"
        self.cache_dir: Path = board_path / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
                name=config.inference.name,
                params=config.inference.params,
                cache_dir=self.cache_dir,
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

        self.results: DataFrame = pd.DataFrame()

    def process(self) -> None:
        # iterate over configurations and then over datasets
        results_list = []
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
                    cache_dir=self.cache_dir,
                )

                experiment = Experiment(
                    name=f"experiment_{config_seq}_{dataset_seq}",
                    data_loader=data_loader,
                    ingest_pipeline=ingest_pipeline,
                    inference_pipeline=inference_pipeline,
                    eval_metrics=self.metric_definitions,
                    cache_dir=self.cache_dir,
                )

                # Run experiment and capture both inference and evaluation results
                inference_results: list[InferenceResult]
                evaluation_results: dict
                inference_results, evaluation_results = experiment.run()

                # Export inference results to CSV for this experiment
                experiment_id = f"exp_{config_seq}_{dataset_seq}"
                self._export_inference_results_csv(
                    inference_results,
                    experiment_id,
                    config_seq,
                    dataset_seq,
                    config.name,
                    dataset.id(),
                    evaluation_results,
                )

                # Export combined results to JSON for this experiment
                self._export_combined_results_json(
                    inference_results,
                    evaluation_results,
                    experiment_id,
                    config_seq,
                    dataset_seq,
                    config.name,
                    dataset.id(),
                )

                config_dataset_df = pd.DataFrame(
                    [
                        {
                            "board_configuration_seq": config_seq,
                            "board_dataset_seq": dataset_seq,
                            "board_configuration_name": config.name,
                            "board_dataset_id": dataset.id(),
                            "board_experiment_id": f"exp_{config_seq}_{dataset_seq}",
                        }
                    ]
                )
                for metric_name in evaluation_results.keys():
                    metric_stats: dict[str, dict] = evaluation_results[metric_name][
                        "statistics"
                    ]
                    for sub_metric_name in metric_stats.keys():
                        for k, v in metric_stats[sub_metric_name].items():
                            config_dataset_df[f"{sub_metric_name}_{k}"] = v

                results_list.append(config_dataset_df)
        #
        self.results = pd.concat(results_list)

        # Export results and markdown at the end of processing
        self.export_results()
        self.export_md()

    def _export_inference_results_csv(
        self,
        inference_results: list[InferenceResult],
        experiment_id: str,
        config_seq: int,
        dataset_seq: int,
        config_name: str,
        dataset_id: str,
        evaluation_results: dict,
    ) -> None:
        """Export inference results to a CSV file for a single experiment.
        
        Args:
            inference_results: List of inference results
            experiment_id: Experiment identifier
            config_seq: Configuration sequence number
            dataset_seq: Dataset sequence number
            config_name: Configuration name
            dataset_id: Dataset identifier
            evaluation_results: Dictionary mapping metric IDs to their evaluation results,
                              where each result contains 'per_question' scores
        """
        os.makedirs(self.output_path, exist_ok=True)

        inference_data = []
        for inf_result in inference_results:
            inference_dict = {
                "board_configuration_seq": config_seq,
                "board_dataset_seq": dataset_seq,
                "board_configuration_name": config_name,
                "board_dataset_id": dataset_id,
                "board_experiment_id": experiment_id,
                "question_id": inf_result.question_id,
                "question": inf_result.question,
                "answer": inf_result.answer,
                "ground_truth_answers": (
                    str(inf_result.ground_truth_answers)
                    if inf_result.ground_truth_answers
                    else None
                ),
                "is_answerable": inf_result.is_answerable,
                "context_ids": (
                    str(inf_result.context_ids) if inf_result.context_ids else None
                ),
                "num_contexts": len(inf_result.contexts) if inf_result.contexts else 0,
            }
            
            # Add metric scores per question
            for metric_id, metric_result in evaluation_results.items():
                per_question_scores = metric_result.get("per_question", {})
                question_scores = per_question_scores.get(inf_result.question_id, {})
                
                # Add each metric score as a separate column
                for score_name, score_value in question_scores.items():
                    column_name = f"{metric_id}_{score_name}"
                    inference_dict[column_name] = score_value
            
            inference_data.append(inference_dict)

        inference_df = pd.DataFrame(inference_data)
        csv_filename = f"inference_results_{experiment_id}.csv"
        inference_df.to_csv(self.output_path / csv_filename, index=False)
        logger.info(f"Exported inference results to {self.output_path / csv_filename}")

    def _export_combined_results_json(
        self,
        inference_results: list[InferenceResult],
        evaluation_results: dict,
        experiment_id: str,
        config_seq: int,
        dataset_seq: int,
        config_name: str,
        dataset_id: str,
    ) -> None:
        """Export combined inference and evaluation results to a JSON file for a single experiment."""
        os.makedirs(self.output_path, exist_ok=True)

        combined_data = {
            "board_configuration_seq": config_seq,
            "board_dataset_seq": dataset_seq,
            "board_configuration_name": config_name,
            "board_dataset_id": dataset_id,
            "board_experiment_id": experiment_id,
            "inference_results": [
                {
                    "question_id": inf_result.question_id,
                    "question": inf_result.question,
                    "answer": inf_result.answer,
                    "ground_truth_answers": inf_result.ground_truth_answers,
                    "is_answerable": inf_result.is_answerable,
                    "context_ids": inf_result.context_ids,
                    "contexts": inf_result.contexts,
                    "trajectory": inf_result.trajectory,
                }
                for inf_result in inference_results
            ],
            "evaluation_results": evaluation_results,
        }

        json_filename = f"combined_results_{experiment_id}.json"
        with open(self.output_path / json_filename, "w") as f:
            json.dump(combined_data, f, indent=2)
        logger.info(f"Exported combined results to {self.output_path / json_filename}")

    def export_results(self) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        self.results.to_csv(self.output_path / self.RESULTS_CSV, index=False)

    def load_results(self) -> None:
        self.results = pd.read_csv(self.output_path / self.RESULTS_CSV)

    # def get_board_results(self) -> BoardResults:
    #     return BoardResults(board=self.board, results=self.results)

    def clean_output(self):
        path = self.output_path
        if os.path.exists(path):
            shutil.rmtree(path)

    def clean_cache(self):
        path = self.cache_dir
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
