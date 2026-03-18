"""Tests for BoardGenerator configuration expansion functionality."""

from ragworkbench.boards.board_generator import BoardGenerator


class TestConfigurationExpansion:
    """Test suite for configuration parameter expansion."""

    def test_find_list_params_simple(self):
        """Test finding list parameters in a simple dictionary."""
        params = {
            "model_id": ["model-1", "model-2"],
            "temperature": 0.7,
        }

        list_params = BoardGenerator._find_list_params(params)

        assert len(list_params) == 1
        assert list_params[0] == ("model_id", ["model-1", "model-2"])

    def test_find_list_params_nested(self):
        """Test finding list parameters in nested dictionaries."""
        params = {
            "embedding_model": {
                "model_id": ["model-1", "model-2"],
                "provider": "openai",
            },
            "chunking": {
                "chunk_size": [512, 256],
                "overlap": 50,
            },
        }

        list_params = BoardGenerator._find_list_params(params)

        assert len(list_params) == 2
        paths = [path for path, _ in list_params]
        assert "embedding_model.model_id" in paths
        assert "chunking.chunk_size" in paths

    def test_find_list_params_deeply_nested(self):
        """Test finding list parameters in deeply nested structures."""
        params = {
            "level1": {
                "level2": {
                    "level3": {
                        "param": ["value1", "value2"],
                    }
                }
            }
        }

        list_params = BoardGenerator._find_list_params(params)

        assert len(list_params) == 1
        assert list_params[0][0] == "level1.level2.level3.param"

    def test_find_list_params_no_lists(self):
        """Test that no list parameters are found when none exist."""
        params = {
            "model_id": "single-model",
            "temperature": 0.7,
            "nested": {
                "value": "test",
            },
        }

        list_params = BoardGenerator._find_list_params(params)

        assert len(list_params) == 0

    def test_set_nested_value_simple(self):
        """Test setting a value in a simple dictionary."""
        params = {"key": "old_value"}

        BoardGenerator._set_nested_value(params, "key", "new_value")

        assert params["key"] == "new_value"

    def test_set_nested_value_nested(self):
        """Test setting a value in a nested dictionary."""
        params = {
            "embedding_model": {
                "model_id": "old_model",
            }
        }

        BoardGenerator._set_nested_value(
            params, "embedding_model.model_id", "new_model"
        )

        assert params["embedding_model"]["model_id"] == "new_model"

    def test_set_nested_value_creates_path(self):
        """Test that setting a nested value creates missing intermediate dictionaries."""
        params = {}

        BoardGenerator._set_nested_value(params, "level1.level2.level3", "value")

        assert params == {"level1": {"level2": {"level3": "value"}}}

    def test_expand_configurations_no_lists(self):
        """Test that configurations without list parameters remain unchanged."""
        configs = [
            {
                "name": "config1",
                "description": "Test config",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "model_id": "single-model",
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {
                        "temperature": 0.7,
                    },
                },
            }
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        assert len(expanded) == 1
        assert expanded[0]["name"] == "config1"

    def test_expand_configurations_single_list_param(self):
        """Test expansion with a single list parameter."""
        configs = [
            {
                "name": "config1",
                "description": "Test config",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "embedding_model": {
                            "model_id": ["model-1", "model-2"],
                        },
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {},
                },
            }
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        assert len(expanded) == 2
        assert expanded[0]["name"] == "config1__1"
        assert expanded[1]["name"] == "config1__2"
        assert (
            expanded[0]["ingest"]["params"]["embedding_model"]["model_id"] == "model-1"
        )
        assert (
            expanded[1]["ingest"]["params"]["embedding_model"]["model_id"] == "model-2"
        )

    def test_expand_configurations_multiple_list_params(self):
        """Test expansion with multiple list parameters (Cartesian product)."""
        configs = [
            {
                "name": "config1",
                "description": "Test config",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "embedding_model": {
                            "model_id": ["model-1", "model-2"],
                        },
                        "chunking": {
                            "chunk_size": [512, 256],
                        },
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {},
                },
            }
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        # Should create 2 × 2 = 4 configurations
        assert len(expanded) == 4

        # Check names
        assert expanded[0]["name"] == "config1__1"
        assert expanded[1]["name"] == "config1__2"
        assert expanded[2]["name"] == "config1__3"
        assert expanded[3]["name"] == "config1__4"

        # Check first combination: model-1, chunk_size=512
        assert (
            expanded[0]["ingest"]["params"]["embedding_model"]["model_id"] == "model-1"
        )
        assert expanded[0]["ingest"]["params"]["chunking"]["chunk_size"] == 512

        # Check last combination: model-2, chunk_size=256
        assert (
            expanded[3]["ingest"]["params"]["embedding_model"]["model_id"] == "model-2"
        )
        assert expanded[3]["ingest"]["params"]["chunking"]["chunk_size"] == 256

    def test_expand_configurations_inference_params(self):
        """Test expansion with list parameters in inference section."""
        configs = [
            {
                "name": "config1",
                "description": "Test config",
                "ingest": {
                    "name": "openrag",
                    "params": {},
                },
                "inference": {
                    "name": "openrag",
                    "params": {
                        "generative_model": {
                            "model_id": ["gpt-4", "gpt-3.5"],
                        },
                    },
                },
            }
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        assert len(expanded) == 2
        assert (
            expanded[0]["inference"]["params"]["generative_model"]["model_id"]
            == "gpt-4"
        )
        assert (
            expanded[1]["inference"]["params"]["generative_model"]["model_id"]
            == "gpt-3.5"
        )

    def test_expand_configurations_mixed_ingest_inference(self):
        """Test expansion with list parameters in both ingest and inference."""
        configs = [
            {
                "name": "config1",
                "description": "Test config",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "chunking": {
                            "chunk_size": [512, 256],
                        },
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {
                        "generative_model": {
                            "model_id": ["gpt-4", "gpt-3.5"],
                        },
                    },
                },
            }
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        # Should create 2 × 2 = 4 configurations
        assert len(expanded) == 4

        # Check first combination
        assert expanded[0]["ingest"]["params"]["chunking"]["chunk_size"] == 512
        assert (
            expanded[0]["inference"]["params"]["generative_model"]["model_id"]
            == "gpt-4"
        )

        # Check last combination
        assert expanded[3]["ingest"]["params"]["chunking"]["chunk_size"] == 256
        assert (
            expanded[3]["inference"]["params"]["generative_model"]["model_id"]
            == "gpt-3.5"
        )

    def test_expand_configurations_description_updated(self):
        """Test that descriptions are updated with parameter values."""
        configs = [
            {
                "name": "config1",
                "description": "Original description",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "model_id": ["model-1", "model-2"],
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {},
                },
            }
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        assert "model_id=model-1" in expanded[0]["description"]
        assert "model_id=model-2" in expanded[1]["description"]
        assert "Original description" in expanded[0]["description"]

    def test_expand_configurations_multiple_configs(self):
        """Test expansion with multiple original configurations."""
        configs = [
            {
                "name": "config1",
                "description": "Config 1",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "model_id": ["model-1", "model-2"],
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {},
                },
            },
            {
                "name": "config2",
                "description": "Config 2",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "chunk_size": [512, 256],
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {},
                },
            },
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        # Should create 2 + 2 = 4 configurations
        assert len(expanded) == 4

        # Check that both original configs were expanded
        config1_expanded = [c for c in expanded if c["name"].startswith("config1")]
        config2_expanded = [c for c in expanded if c["name"].startswith("config2")]

        assert len(config1_expanded) == 2
        assert len(config2_expanded) == 2

    def test_expand_configurations_preserves_other_fields(self):
        """Test that expansion preserves fields not related to parameters."""
        configs = [
            {
                "name": "config1",
                "description": "Test config",
                "custom_field": "custom_value",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "model_id": ["model-1", "model-2"],
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {},
                },
            }
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        assert expanded[0]["custom_field"] == "custom_value"
        assert expanded[1]["custom_field"] == "custom_value"
        assert expanded[0]["ingest"]["name"] == "openrag"
        assert expanded[1]["ingest"]["name"] == "openrag"

    def test_expand_configurations_three_params(self):
        """Test expansion with three list parameters."""
        configs = [
            {
                "name": "config1",
                "description": "Test config",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "param1": ["a", "b"],
                        "param2": ["x", "y"],
                        "param3": [1, 2],
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {},
                },
            }
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        # Should create 2 × 2 × 2 = 8 configurations
        assert len(expanded) == 8

        # Verify all combinations exist
        param1_values = [c["ingest"]["params"]["param1"] for c in expanded]
        param2_values = [c["ingest"]["params"]["param2"] for c in expanded]
        param3_values = [c["ingest"]["params"]["param3"] for c in expanded]

        # Check that we have all combinations
        assert set(param1_values) == {"a", "b"}
        assert set(param2_values) == {"x", "y"}
        assert set(param3_values) == {1, 2}

    def test_expand_configurations_empty_list(self):
        """Test that empty lists are handled gracefully."""
        configs = [
            {
                "name": "config1",
                "description": "Test config",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "model_id": [],
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {},
                },
            }
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        # Empty list should result in no configurations
        assert len(expanded) == 0

    def test_expand_configurations_single_value_list(self):
        """Test expansion with a single-value list."""
        configs = [
            {
                "name": "config1",
                "description": "Test config",
                "ingest": {
                    "name": "openrag",
                    "params": {
                        "model_id": ["single-model"],
                    },
                },
                "inference": {
                    "name": "openrag",
                    "params": {},
                },
            }
        ]

        expanded = BoardGenerator._expand_configurations(configs)

        assert len(expanded) == 1
        assert expanded[0]["name"] == "config1__1"
        assert expanded[0]["ingest"]["params"]["model_id"] == "single-model"


# Made with Bob
