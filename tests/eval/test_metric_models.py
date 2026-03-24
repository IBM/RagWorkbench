"""
Unit tests for metric_models module.

Tests the functionality of loading metric definitions from YAML files
and retrieving metric names.
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from ragworkbench.eval import (
    MetricDefinition,
    MetricDefinitionsConfig,
    load_metric_definitions,
)


class TestMetricDefinition:
    """Test cases for MetricDefinition model."""

    def test_valid_metric_definition(self):
        """Test creating a valid MetricDefinition."""
        metric = MetricDefinition(
            metric_name="unitxt.context_correctness.retrieval_at_k",
            metric_id="metrics.rag.context_correctness.retrieval_at_k",
            metric_params={"sub_scores": ["match_at_1", "match_at_3"]},
            metric_fields=["context_ids", "ground_truths_context_ids"],
            vendor="unitxt",
        )
        assert metric.metric_name == "unitxt.context_correctness.retrieval_at_k"
        assert metric.metric_id == "metrics.rag.context_correctness.retrieval_at_k"
        assert metric.metric_params == {"sub_scores": ["match_at_1", "match_at_3"]}
        assert metric.metric_fields == ["context_ids", "ground_truths_context_ids"]
        assert metric.vendor == "unitxt"


class TestMetricDefinitionsConfig:
    """Test cases for MetricDefinitionsConfig model."""

    def test_valid_config(self):
        """Test creating a valid MetricDefinitionsConfig."""
        definitions = {
            "unitxt.context_correctness.retrieval_at_k": MetricDefinition(
                metric_name="unitxt.context_correctness.retrieval_at_k",
                metric_id="metrics.rag.context_correctness.retrieval_at_k",
                metric_params={"sub_scores": ["match_at_1"]},
                metric_fields=["context_ids", "ground_truths_context_ids"],
                vendor="unitxt",
            ),
            "unitxt.context_correctness.map": MetricDefinition(
                metric_name="unitxt.context_correctness.map",
                metric_id="metrics.rag.context_correctness.map",
                metric_params={},
                metric_fields=["context_ids", "ground_truths_context_ids"],
                vendor="unitxt",
            ),
        }
        config = MetricDefinitionsConfig(definitions=definitions)
        assert len(config.definitions) == 2
        assert "unitxt.context_correctness.retrieval_at_k" in config.definitions
        assert "unitxt.context_correctness.map" in config.definitions

    def test_get_metric_names(self):
        """Test get_metric_names method returns all metric names."""
        definitions = {
            "unitxt.context_correctness.retrieval_at_k": MetricDefinition(
                metric_name="unitxt.context_correctness.retrieval_at_k",
                metric_id="metrics.rag.context_correctness.retrieval_at_k",
                metric_params={},
                metric_fields=["context_ids", "ground_truths_context_ids"],
                vendor="unitxt",
            ),
            "unitxt.context_correctness.map": MetricDefinition(
                metric_name="unitxt.context_correctness.map",
                metric_id="metrics.rag.context_correctness.map",
                metric_params={},
                metric_fields=["context_ids", "ground_truths_context_ids"],
                vendor="unitxt",
            ),
            "unitxt.answer_correctness": MetricDefinition(
                metric_name="unitxt.answer_correctness",
                metric_id="metrics.rag.answer_correctness",
                metric_params={},
                metric_fields=["ground_truths", "answer"],
                vendor="unitxt",
            ),
        }
        config = MetricDefinitionsConfig(definitions=definitions)
        metric_names = config.get_metric_names()

        assert isinstance(metric_names, list)
        assert len(metric_names) == 3
        assert "unitxt.context_correctness.retrieval_at_k" in metric_names
        assert "unitxt.context_correctness.map" in metric_names
        assert "unitxt.answer_correctness" in metric_names

    def test_get_metric_names_order(self):
        """Test that get_metric_names returns names in consistent order."""
        definitions = {
            "metric_c": MetricDefinition(
                metric_name="metric_c",
                metric_id="metrics.c",
                metric_params={},
                metric_fields=["field1"],
                vendor="unitxt",
            ),
            "metric_a": MetricDefinition(
                metric_name="metric_a",
                metric_id="metrics.a",
                metric_params={},
                metric_fields=["field1"],
                vendor="unitxt",
            ),
            "metric_b": MetricDefinition(
                metric_name="metric_b",
                metric_id="metrics.b",
                metric_params={},
                metric_fields=["field1"],
                vendor="unitxt",
            ),
        }
        config = MetricDefinitionsConfig(definitions=definitions)
        metric_names = config.get_metric_names()

        # Should contain all keys
        assert set(metric_names) == {"metric_a", "metric_b", "metric_c"}

    def test_get_metric_definition(self):
        """Test get_metric_definition method returns correct metric."""
        definitions = {
            "unitxt.context_correctness.retrieval_at_k": MetricDefinition(
                metric_name="unitxt.context_correctness.retrieval_at_k",
                metric_id="metrics.rag.context_correctness.retrieval_at_k",
                metric_params={"sub_scores": ["match_at_1"]},
                metric_fields=["context_ids", "ground_truths_context_ids"],
                vendor="unitxt",
            ),
            "unitxt.context_correctness.map": MetricDefinition(
                metric_name="unitxt.context_correctness.map",
                metric_id="metrics.rag.context_correctness.map",
                metric_params={},
                metric_fields=["context_ids", "ground_truths_context_ids"],
                vendor="unitxt",
            ),
        }
        config = MetricDefinitionsConfig(definitions=definitions)

        # Test retrieving existing metric
        metric_def = config.get_metric_definition("unitxt.context_correctness.map")
        assert metric_def.metric_id == "metrics.rag.context_correctness.map"
        assert metric_def.vendor == "unitxt"
        assert metric_def.metric_fields == ["context_ids", "ground_truths_context_ids"]

    def test_get_metric_definition_via_direct_access(self):
        """Test that direct dictionary access still works."""
        definitions = {
            "unitxt.context_correctness.map": MetricDefinition(
                metric_name="unitxt.context_correctness.map",
                metric_id="metrics.rag.context_correctness.map",
                metric_params={},
                metric_fields=["context_ids", "ground_truths_context_ids"],
                vendor="unitxt",
            ),
        }
        config = MetricDefinitionsConfig(definitions=definitions)

        # Test direct access (original way)
        metric_def = config.definitions["unitxt.context_correctness.map"]
        assert metric_def.metric_id == "metrics.rag.context_correctness.map"


class TestLoadMetricDefinitions:
    """Test cases for load_metric_definitions function."""

    def test_load_default_yaml(self):
        """Test loading the default metric_defs.yaml file."""
        config = load_metric_definitions()

        assert isinstance(config, MetricDefinitionsConfig)
        assert len(config.definitions) > 0

        # Check that some expected metrics are present
        metric_names = config.get_metric_names()
        assert "unitxt.context_correctness.retrieval_at_k" in metric_names
        assert "unitxt.context_correctness.map" in metric_names
        assert "unitxt.answer_correctness" in metric_names

    def test_load_custom_yaml(self):
        """Test loading a custom YAML file."""
        # Create a temporary YAML file
        test_data = {
            "test.metric.one": {
                "metric_id": "metrics.test.one",
                "metric_params": {"param1": "value1"},
                "metric_fields": ["field1", "field2"],
                "vendor": "unitxt",
            },
            "test.metric.two": {
                "metric_id": "metrics.test.two",
                "metric_params": {},
                "metric_fields": ["field3"],
                "vendor": "unitxt",
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_data, f)
            temp_path = f.name

        try:
            config = load_metric_definitions(temp_path)

            assert isinstance(config, MetricDefinitionsConfig)
            assert len(config.definitions) == 2

            metric_names = config.get_metric_names()
            assert "test.metric.one" in metric_names
            assert "test.metric.two" in metric_names

            # Verify the loaded data
            metric_one = config.definitions["test.metric.one"]
            assert metric_one.metric_id == "metrics.test.one"
            assert metric_one.metric_params == {"param1": "value1"}
            assert metric_one.metric_fields == ["field1", "field2"]
        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_load_with_path_object(self):
        """Test loading with a Path object instead of string."""
        # Create a temporary YAML file
        test_data = {
            "test.metric": {
                "metric_id": "metrics.test",
                "metric_params": {},
                "metric_fields": ["field1"],
                "vendor": "unitxt",
            }
        }

        with NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            config = load_metric_definitions(temp_path)
            assert isinstance(config, MetricDefinitionsConfig)
            assert len(config.definitions) == 1
        finally:
            temp_path.unlink()


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_complete_workflow(self):
        """Test the complete workflow of loading and accessing metrics."""
        # Load the default configuration
        config = load_metric_definitions()

        # Get all metric names
        metric_names = config.get_metric_names()
        assert len(metric_names) > 0

        # Access a specific metric
        first_metric_name = metric_names[0]
        metric_def = config.definitions[first_metric_name]

        # Verify the metric definition has all required attributes
        assert hasattr(metric_def, "metric_id")
        assert hasattr(metric_def, "metric_params")
        assert hasattr(metric_def, "metric_fields")
        assert hasattr(metric_def, "vendor")

        # Verify types
        assert isinstance(metric_def.metric_id, str)
        assert isinstance(metric_def.metric_params, dict)
        assert isinstance(metric_def.metric_fields, list)
        assert isinstance(metric_def.vendor, str)
