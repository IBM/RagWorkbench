import hashlib
import json
import logging
import os
import shutil
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pandas import DataFrame

from ragworkbench import DataLoaderFactory, Experiment
from ragworkbench.boards.board_model import Board
from ragworkbench.boards.board_registry import BoardRegistry
from ragworkbench.eval import (
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
        # Create unique output directory name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path: Path = board_path / f"output_{timestamp}"
        self.cache_dir: Path = board_path / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        yaml_file = self.input_path / self.BOARD_YAML
        with yaml_file.open("r") as f:
            data = yaml.full_load(f)

        # Copy board.yaml to output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(yaml_file, self.output_path / self.BOARD_YAML)

        # Expand configurations with list parameters into multiple configurations
        data["configurations"] = self._expand_configurations(data["configurations"])

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
                cache_mode=self.board.experiment.cache,
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
        self.experiment_id_mapping: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _find_list_params(
        params: dict[str, Any], prefix: str = ""
    ) -> list[tuple[str, list]]:
        """Recursively find all parameters that have list values.

        Args:
            params: Dictionary of parameters to search
            prefix: Current path prefix for nested keys

        Returns:
            List of tuples (param_path, list_values) where param_path uses dot notation
        """
        list_params = []
        for key, value in params.items():
            current_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, list):
                list_params.append((current_path, value))
            elif isinstance(value, dict):
                list_params.extend(
                    BoardGenerator._find_list_params(value, current_path)
                )
        return list_params

    @staticmethod
    def _set_nested_value(params: dict[str, Any], path: str, value: Any) -> None:
        """Set a value in a nested dictionary using dot notation path.

        Args:
            params: Dictionary to modify
            path: Dot-separated path (e.g., "embedding_model.model_id")
            value: Value to set
        """
        keys = path.split(".")
        current = params
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    @staticmethod
    def _expand_configurations(
        configurations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expand configurations that contain list parameters into multiple configurations.

        When a parameter value is a list, this creates multiple configurations with all
        combinations of the list values (Cartesian product).

        Args:
            configurations: List of configuration dictionaries from YAML

        Returns:
            Expanded list of configurations with all parameter combinations

        Example:
            Input configuration with:
                embedding_model.model_id: [model-1, model-2]
                chunking.chunk_size: [512, 256]

            Will generate 4 configurations (2 × 2):
                1. model-1, chunk_size=512
                2. model-1, chunk_size=256
                3. model-2, chunk_size=512
                4. model-2, chunk_size=256
        """
        expanded_configs = []

        for config in configurations:
            # Find all parameters with list values in both ingest and inference
            ingest_list_params = BoardGenerator._find_list_params(
                config.get("ingest", {}).get("params", {})
            )
            inference_list_params = BoardGenerator._find_list_params(
                config.get("inference", {}).get("params", {})
            )

            # Combine all list parameters
            all_list_params = ingest_list_params + inference_list_params

            if not all_list_params:
                # No list parameters, keep configuration as-is
                expanded_configs.append(config)
                continue

            # Generate all combinations of list parameter values
            param_paths = [path for path, _ in all_list_params]
            param_values = [values for _, values in all_list_params]
            combinations = list(product(*param_values))

            logger.info(
                f"Expanding configuration '{config['name']}' with {len(combinations)} "
                f"parameter combinations"
            )

            # Create a new configuration for each combination
            for combo_idx, combo_values in enumerate(combinations):
                # Deep copy the original configuration
                new_config = deepcopy(config)

                # Build a suffix for the configuration name based on parameter values
                param_suffix_parts = []
                for param_path, param_value in zip(
                    param_paths, combo_values, strict=True
                ):
                    # Extract the last part of the path for the suffix
                    param_name = param_path.split(".")[-1]
                    param_suffix_parts.append(f"{param_name}={param_value}")

                # Update configuration name with parameter values
                original_name = config["name"]
                new_config["name"] = f"{original_name}__{combo_idx + 1}"

                # Update description to include parameter values
                param_description = ", ".join(param_suffix_parts)
                original_description = config.get("description", "")
                new_config["description"] = (
                    f"{original_description} [{param_description}]"
                    if original_description
                    else f"[{param_description}]"
                )

                # Set each parameter value in the new configuration
                for param_path, param_value in zip(
                    param_paths, combo_values, strict=True
                ):
                    # Determine if this is an ingest or inference parameter
                    if param_path in [p for p, _ in ingest_list_params]:
                        BoardGenerator._set_nested_value(
                            new_config["ingest"]["params"], param_path, param_value
                        )
                    else:
                        BoardGenerator._set_nested_value(
                            new_config["inference"]["params"], param_path, param_value
                        )

                expanded_configs.append(new_config)
                logger.debug(
                    f"Created configuration '{new_config['name']}' with parameters: "
                    f"{param_description}"
                )

        logger.info(
            f"Configuration expansion complete: {len(configurations)} original → "
            f"{len(expanded_configs)} expanded"
        )

        return expanded_configs

    @staticmethod
    def _generate_experiment_id(config, dataset) -> tuple[str, dict[str, Any]]:
        """Generate a unique experiment ID based on all configuration and dataset parameters.

        Args:
            config: Configuration object containing ingest and inference parameters
            dataset: Dataset object containing name, split, and sampling parameters

        Returns:
            A tuple of (experiment_id, experiment_params) where:
            - experiment_id: A unique string identifier based on a hash of all parameters
            - experiment_params: Dictionary containing all the parameters
        """
        # Collect all parameters that define the experiment
        experiment_params = {
            # "config_name": config.name,
            # "config_description": config.description,
            # "ingest_name": config.ingest.name,
            "ingest_params": config.ingest.params,
            "inference_name": config.inference.name,
            "inference_params": config.inference.params,
            "dataset_name": (
                dataset.name if isinstance(dataset.name, str) else dataset.name.value
            ),
            "dataset_split": dataset.split.value if dataset.split else None,
            "sampling_question_limit": dataset.sampling.question_limit,
            "sampling_document_factor": dataset.sampling.document_factor,
            "sampling_seed": dataset.sampling.seed,
        }

        # Create a stable JSON representation (sorted keys for consistency)
        params_json = json.dumps(experiment_params, sort_keys=True, default=str)

        # Generate a hash of the parameters (8 characters)
        params_hash = hashlib.sha256(params_json.encode()).hexdigest()[:8]

        # Create a readable experiment ID with the hash
        experiment_id = f"exp_{params_hash}"

        return experiment_id, experiment_params

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

                # Generate unique experiment ID based on all parameters
                experiment_id, experiment_params = (
                    BoardGenerator._generate_experiment_id(config, dataset)
                )

                # Store the mapping from experiment_id to full configuration
                self.experiment_id_mapping[experiment_id] = experiment_params

                experiment = Experiment(
                    experiment_id=experiment_id,
                    data_loader=data_loader,
                    ingest_pipeline=ingest_pipeline,
                    inference_pipeline=inference_pipeline,
                    eval_metrics=self.metric_definitions,
                    experiment_config=self.board.experiment,
                    cache_dir=self.cache_dir,
                )

                # Run experiment and get ExperimentResult object
                experiment_result = experiment.run()

                # Log cost data if tracking is enabled
                if self.board.experiment.usage_tracking and experiment_result.cost_data:
                    logger.info(
                        f"Experiment {experiment_id} cost: "
                        f"${experiment_result.cost_data.total_cost:.4f}, "
                        f"{experiment_result.cost_data.total_tokens} tokens"
                    )

                # Export results using ExperimentResult methods

                experiment_result.export_all(
                    self.output_path,
                    config_seq,
                    dataset_seq,
                    config.name,
                    dataset.id(),
                )

                # Create experiment summary using the new method
                summary_dict = experiment_result.create_summary(
                    config_seq=config_seq,
                    dataset_seq=dataset_seq,
                    config_name=config.name,
                    dataset_id=dataset.id(),
                )
                config_dataset_df = pd.DataFrame([summary_dict])

                results_list.append(config_dataset_df)
        #
        self.results = pd.concat(results_list)

        # Export results and markdown at the end of processing
        self.export_results()
        self.export_experiment_id_mapping()
        self.export_md()

    def export_results(self) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        self.results.to_csv(self.output_path / self.RESULTS_CSV, index=False)

    def export_experiment_id_mapping(self) -> None:
        """Export the mapping from experiment IDs to their full configurations."""
        os.makedirs(self.output_path, exist_ok=True)
        mapping_file = self.output_path / "experiment_id_mapping.json"
        with open(mapping_file, "w") as f:
            json.dump(self.experiment_id_mapping, f, indent=2, default=str)
        logger.info(f"Exported experiment ID mapping to {mapping_file}")

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
            config_row_span = 1 + (1 if len(screen.columns) > 1 else 0)
            md += f"### {screen.title}\n"
            md += "<table>\n"
            md += "<tr>\n"
            md += (
                "\n".join(
                    [f'\t<th rowspan="{config_row_span}">Configuration</th>']
                    + [
                        f'\t<th colspan="{len(screen.columns)}">{dataset.name}</th>'
                        for dataset in board.datasets
                    ]
                )
                + "\n"
            )
            md += "</tr>\n"
            if len(screen.columns) > 1:
                md += "<tr>\n"
                for _ in self.board.datasets:
                    for score_title in screen.columns.values():
                        md += f"\t<td>{score_title}</td>\n"
                md += "</tr>\n"
            for config_seq, config in enumerate(self.board.configurations):
                md += "<tr>\n"
                md += f"\t<td>{config.name}</td>\n"
                for dataset_seq, _dataset in enumerate(self.board.datasets):
                    df = self.results[
                        (self.results["board_configuration_seq"] == config_seq)
                        & (self.results["board_dataset_seq"] == dataset_seq)
                    ]

                    if len(df) > 1:
                        raise ValueError(
                            f"Expected at most 1 result for config_seq={config_seq} and "
                            f"dataset_seq={dataset_seq}, but found {len(df)} results:\n{df}"
                        )

                    row = df.iloc[0]
                    for score in screen.columns.keys():
                        try:
                            value = row[score]
                            # Handle different value types
                            if isinstance(value, (int, float)):
                                md += f"\t<td>{value:.2f}</td>\n"
                            elif isinstance(value, list):
                                # Format lists as comma-separated strings
                                md += f"\t<td>{', '.join(str(v) for v in value)}</td>\n"
                            else:
                                # Handle strings and other types
                                md += f"\t<td>{value}</td>\n"
                        except KeyError as e:
                            available_keys = list(row.index)
                            raise KeyError(
                                f"Score key '{score}' not found in DataFrame row. "
                                f"Available keys: {available_keys}"
                            ) from e
                md += "</tr>\n"
            md += "</table>\n\n"
        return md
