import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from ragworkbench.api.inference_result import InferenceResult
from ragworkbench.eval.cost_tracking import UsageData

logger = logging.getLogger(__name__)


class ExperimentResult(BaseModel):
    """Encapsulates all results from running an experiment including inference, evaluation, and cost data."""

    experiment_id: str = Field(description="Unique identifier for the experiment")
    inference_results: list[InferenceResult] = Field(
        description="List of inference results from the experiment"
    )
    evaluation_results: dict[str, Any] = Field(
        description="Dictionary of evaluation metrics and scores"
    )
    cost_data: UsageData = Field(
        default_factory=UsageData,
        description="Usage data containing cost tracking information",
    )

    def create_summary(
        self,
        config_seq: int,
        dataset_seq: int,
        config_name: str,
        dataset_id: str,
    ) -> dict[str, Any]:
        """
        Create a summary dictionary of the experiment including evaluation metrics and usage data.

        Args:
            config_seq: Configuration sequence number
            dataset_seq: Dataset sequence number
            config_name: Name of the configuration used
            dataset_id: ID of the dataset used

        Returns:
            Dictionary containing experiment summary with metrics statistics and usage data
        """
        summary = {
            "board_configuration_seq": config_seq,
            "board_dataset_seq": dataset_seq,
            "board_configuration_name": config_name,
            "board_dataset_id": dataset_id,
            "board_experiment_id": self.experiment_id,
        }

        # Add evaluation metrics statistics
        for metric_name in self.evaluation_results.keys():
            metric_stats: dict[str, dict] = self.evaluation_results[metric_name].get(
                "statistics", {}
            )
            for sub_metric_name in metric_stats.keys():
                for k, v in metric_stats[sub_metric_name].items():
                    summary[f"{sub_metric_name}_{k}"] = v

        # Add usage/cost data
        if self.cost_data:
            summary["total_cost"] = self.cost_data.total_cost
            summary["total_tokens"] = self.cost_data.total_tokens
            summary["prompt_tokens"] = self.cost_data.prompt_tokens
            summary["completion_tokens"] = self.cost_data.completion_tokens
            summary["requests"] = self.cost_data.requests
            summary["models_used"] = self.cost_data.models_used

        return summary

    def export_inference_results_csv(
        self,
        output_path: Path,
        config_seq: int,
        dataset_seq: int,
        config_name: str,
        dataset_id: str,
    ) -> None:
        """
        Export inference results to a CSV file with evaluation metrics.

        Args:
            output_path: Directory path where the CSV file will be saved
        """
        os.makedirs(output_path, exist_ok=True)

        inference_data = []
        for inf_result in self.inference_results:
            inference_dict = {
                "board_configuration_seq": config_seq,
                "board_dataset_seq": dataset_seq,
                "board_configuration_name": config_name,
                "board_dataset_id": dataset_id,
                "board_experiment_id": self.experiment_id,
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
            for metric_name, metric_result in self.evaluation_results.items():
                per_question_scores = metric_result.get("per_question", {})
                question_scores = per_question_scores.get(inf_result.question_id, {})

                # Add each metric score as a separate column
                for score_name, score_value in question_scores.items():
                    # Avoid duplication: if score_name equals metric_name, use it as-is
                    # Otherwise, concatenate metric_name with score_name
                    if score_name == metric_name:
                        column_name = metric_name
                    else:
                        column_name = f"{metric_name}.{score_name}"
                    inference_dict[column_name] = score_value

            inference_data.append(inference_dict)

        inference_df = pd.DataFrame(inference_data)
        csv_filename = f"experiment_inference_results_{self.experiment_id}.csv"
        inference_df.to_csv(output_path / csv_filename, index=False)
        logger.info(f"Exported inference results to {output_path / csv_filename}")

    def export_to_json(
        self,
        output_path: Path,
        config_seq: int,
        dataset_seq: int,
        config_name: str,
        dataset_id: str,
    ) -> None:
        """
        Export combined inference, evaluation, and cost tracking results to a JSON file.

        Args:
            output_path: Directory path where the JSON file will be saved
        """
        os.makedirs(output_path, exist_ok=True)

        combined_data = {
            "board_configuration_seq": config_seq,
            "board_dataset_seq": dataset_seq,
            "board_configuration_name": config_name,
            "board_dataset_id": dataset_id,
            "board_experiment_id": self.experiment_id,
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
                for inf_result in self.inference_results
            ],
            "evaluation_results": self.evaluation_results,
            "cost_data": self.cost_data.model_dump(),
        }

        json_filename = f"experiment_results_{self.experiment_id}.json"
        with open(output_path / json_filename, "w") as f:
            json.dump(combined_data, f, indent=2)
        logger.info(f"Exported combined results to {output_path / json_filename}")

    def export_all(
        self,
        output_path: Path,
        config_seq: int,
        dataset_seq: int,
        config_name: str,
        dataset_id: str,
    ) -> None:
        """
        Export both CSV and JSON results.

        Args:
            output_path: Directory path where files will be saved
            config_seq: Configuration sequence number
            dataset_seq: Dataset sequence number
            config_name: Name of the configuration used
            dataset_id: ID of the dataset used
        """
        self.export_inference_results_csv(
            output_path, config_seq, dataset_seq, config_name, dataset_id
        )
        self.export_to_json(
            output_path, config_seq, dataset_seq, config_name, dataset_id
        )
