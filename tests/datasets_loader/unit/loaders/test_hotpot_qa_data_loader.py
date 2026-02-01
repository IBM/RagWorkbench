"""
Unit tests for HotpotQaDataLoader.

This module tests the HotpotQA data loader implementation with mocked
HuggingFace datasets to avoid requiring internet access. Tests cover:
- Initialization
- Document loading from context paragraphs
- Benchmark entry loading from questions with supporting facts
- Split handling (train, validation, test, None)
- Data format validation
- Integration with parent class functionality
"""

from io import BytesIO
from unittest.mock import MagicMock, patch

from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets_loader.data_models.dataset_names import DatasetName
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus
from ragbench.datasets_loader.hotpot_qa_data_loader import HotpotQaDataLoader


class TestHotpotQaDataLoaderInitialization:
    """Test suite for HotpotQaDataLoader initialization."""

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_initialization_with_defaults(self, mock_load_dataset):
        """Test initialization with default parameters."""
        # Mock the HuggingFace dataset structure
        mock_train = MagicMock()
        mock_train.__len__ = MagicMock(return_value=2)
        mock_train.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "What is the capital?",
                        "answer": "Paris",
                        "type": "bridge",
                        "level": "easy",
                        "context": {
                            "title": ["France", "Europe"],
                            "sentences": [
                                ["France is a country.", "Paris is its capital."],
                                ["Europe is a continent."],
                            ],
                        },
                        "supporting_facts": {
                            "title": ["France"],
                            "sent_id": [0],
                        },
                    }
                ]
            )
        )

        mock_val = MagicMock()
        mock_val.__len__ = MagicMock(return_value=1)
        mock_val.__iter__ = MagicMock(return_value=iter([]))

        mock_test = MagicMock()
        mock_test.__len__ = MagicMock(return_value=1)
        mock_test.__iter__ = MagicMock(return_value=iter([]))

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader()

        assert loader.dataset_name == DatasetName.HOTPOT_QA
        assert loader.split is None
        assert isinstance(loader.sampling_params, DataSamplingParams)
        assert loader.dataset_path is None

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_initialization_with_custom_parameters(self, mock_load_dataset):
        """Test initialization with custom parameters."""
        # Mock the HuggingFace dataset
        mock_train = MagicMock()
        mock_train.__len__ = MagicMock(return_value=5)
        mock_train.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "Test?",
                        "answer": "Answer",
                        "type": "comparison",
                        "level": "hard",
                        "context": {
                            "title": ["Doc1"],
                            "sentences": [["Sentence 1."]],
                        },
                        "supporting_facts": {"title": [], "sent_id": []},
                    }
                ]
            )
        )

        mock_val = MagicMock()
        mock_val.__len__ = MagicMock(return_value=0)
        mock_val.__iter__ = MagicMock(return_value=iter([]))

        mock_test = MagicMock()
        mock_test.__len__ = MagicMock(return_value=0)
        mock_test.__iter__ = MagicMock(return_value=iter([]))

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        sampling_params = DataSamplingParams(question_limit=3, seed=42)
        loader = HotpotQaDataLoader(
            dataset_name=DatasetName.HOTPOT_QA,
            split="train",
            sampling_params=sampling_params,
            dataset_path="/custom/path",
        )

        assert loader.dataset_name == DatasetName.HOTPOT_QA
        assert loader.split == "train"
        assert loader.sampling_params.question_limit == 3
        assert loader.dataset_path == "/custom/path"


class TestHotpotQaDataLoaderDocumentLoading:
    """Test suite for document loading functionality."""

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_get_documents_loads_unique_contexts(self, mock_load_dataset):
        """Test that _get_documents extracts unique documents from contexts."""
        # Mock dataset with multiple questions sharing some contexts
        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "Question 1?",
                        "answer": "Answer 1",
                        "type": "bridge",
                        "level": "easy",
                        "context": {
                            "title": ["France", "Paris"],
                            "sentences": [
                                ["France is a country.", "It is in Europe."],
                                ["Paris is the capital.", "It has the Eiffel Tower."],
                            ],
                        },
                        "supporting_facts": {"title": ["France"], "sent_id": [0]},
                    },
                    {
                        "id": "q2",
                        "question": "Question 2?",
                        "answer": "Answer 2",
                        "type": "comparison",
                        "level": "medium",
                        "context": {
                            "title": ["Paris", "London"],
                            "sentences": [
                                ["Paris is the capital.", "It has the Eiffel Tower."],
                                ["London is the capital of UK."],
                            ],
                        },
                        "supporting_facts": {"title": ["Paris"], "sent_id": [0]},
                    },
                ]
            )
        )

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader()
        documents = loader._get_documents()

        # Should have 3 unique documents: France, Paris, London
        assert len(documents) == 3
        doc_names = {doc.name for doc in documents}
        assert doc_names == {"France", "Paris", "London"}

        # Verify content
        doc_dict = {doc.name: doc for doc in documents}
        assert (
            doc_dict["France"].stream.read() == b"France is a country. It is in Europe."
        )
        assert (
            doc_dict["Paris"].stream.read()
            == b"Paris is the capital. It has the Eiffel Tower."
        )

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_get_documents_concatenates_sentences(self, mock_load_dataset):
        """Test that sentences are properly concatenated with spaces."""
        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "Test?",
                        "answer": "Answer",
                        "type": "bridge",
                        "level": "easy",
                        "context": {
                            "title": ["TestDoc"],
                            "sentences": [
                                [
                                    "First sentence.",
                                    "Second sentence.",
                                    "Third sentence.",
                                ]
                            ],
                        },
                        "supporting_facts": {"title": ["TestDoc"], "sent_id": [0]},
                    }
                ]
            )
        )

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader()
        documents = loader._get_documents()

        assert len(documents) == 1
        doc = documents[0]
        assert doc.name == "TestDoc"
        assert doc.mime_type == "text/plain"
        assert isinstance(doc.stream, BytesIO)
        assert doc.stream.read() == b"First sentence. Second sentence. Third sentence."

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_get_documents_processes_all_splits(self, mock_load_dataset):
        """Test that documents are extracted from all splits."""
        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "Train question?",
                        "answer": "Train answer",
                        "type": "bridge",
                        "level": "easy",
                        "context": {
                            "title": ["TrainDoc"],
                            "sentences": [["Train content."]],
                        },
                        "supporting_facts": {"title": ["TrainDoc"], "sent_id": [0]},
                    }
                ]
            )
        )

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q2",
                        "question": "Val question?",
                        "answer": "Val answer",
                        "type": "comparison",
                        "level": "medium",
                        "context": {
                            "title": ["ValDoc"],
                            "sentences": [["Val content."]],
                        },
                        "supporting_facts": {"title": ["ValDoc"], "sent_id": [0]},
                    }
                ]
            )
        )

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q3",
                        "question": "Test question?",
                        "answer": "Test answer",
                        "type": "bridge",
                        "level": "hard",
                        "context": {
                            "title": ["TestDoc"],
                            "sentences": [["Test content."]],
                        },
                        "supporting_facts": {"title": ["TestDoc"], "sent_id": [0]},
                    }
                ]
            )
        )

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader()
        documents = loader._get_documents()

        # Should have documents from all three splits
        assert len(documents) == 3
        doc_names = {doc.name for doc in documents}
        assert doc_names == {"TrainDoc", "ValDoc", "TestDoc"}


class TestHotpotQaDataLoaderBenchmarkLoading:
    """Test suite for benchmark entry loading functionality."""

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_get_benchmark_entries_train_split(self, mock_load_dataset):
        """Test loading benchmark entries with train split."""
        mock_train_data = [
            {
                "id": "train_q1",
                "question": "What is the capital of France?",
                "answer": "Paris",
                "type": "bridge",
                "level": "easy",
                "context": {
                    "title": ["France", "Paris"],
                    "sentences": [
                        ["France is a country."],
                        ["Paris is the capital."],
                    ],
                },
                "supporting_facts": {"title": ["France", "Paris"], "sent_id": [0, 0]},
            },
            {
                "id": "train_q2",
                "question": "Which is larger?",
                "answer": "France",
                "type": "comparison",
                "level": "medium",
                "context": {
                    "title": ["France", "Belgium"],
                    "sentences": [["France is large."], ["Belgium is small."]],
                },
                "supporting_facts": {"title": ["France"], "sent_id": [0]},
            },
        ]

        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter(mock_train_data))
        mock_train.__len__ = MagicMock(return_value=len(mock_train_data))

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))
        mock_val.__len__ = MagicMock(return_value=0)

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))
        mock_test.__len__ = MagicMock(return_value=0)

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader(split="train")
        entries = loader._get_benchmark_entries("train")

        assert len(entries) == 2
        assert entries[0].question_id == "train_q1"
        assert entries[0].question == "What is the capital of France?"
        assert entries[0].ground_truth_answers == ["Paris"]
        assert len(entries[0].ground_truth_context_ids) == 2
        assert entries[0].is_answerable is True
        assert entries[0].additional_information["question_type"] == "bridge"
        assert entries[0].additional_information["level"] == "easy"

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_get_benchmark_entries_validation_split(self, mock_load_dataset):
        """Test loading benchmark entries with validation split."""
        mock_val_data = [
            {
                "id": "val_q1",
                "question": "Validation question?",
                "answer": "Validation answer",
                "type": "bridge",
                "level": "hard",
                "context": {
                    "title": ["Doc1"],
                    "sentences": [["Content."]],
                },
                "supporting_facts": {"title": ["Doc1"], "sent_id": [0]},
            }
        ]

        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter([]))
        mock_train.__len__ = MagicMock(return_value=0)

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter(mock_val_data))
        mock_val.__len__ = MagicMock(return_value=len(mock_val_data))

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))
        mock_test.__len__ = MagicMock(return_value=0)

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader(split="validation")
        entries = loader._get_benchmark_entries("validation")

        assert len(entries) == 1
        assert entries[0].question_id == "val_q1"

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_get_benchmark_entries_test_split(self, mock_load_dataset):
        """Test loading benchmark entries with test split."""
        mock_test_data = [
            {
                "id": "test_q1",
                "question": "Test question?",
                "answer": "Test answer",
                "type": "comparison",
                "level": "easy",
                "context": {
                    "title": ["Doc1"],
                    "sentences": [["Content."]],
                },
                "supporting_facts": {"title": ["Doc1"], "sent_id": [0]},
            }
        ]

        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter([]))
        mock_train.__len__ = MagicMock(return_value=0)

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))
        mock_val.__len__ = MagicMock(return_value=0)

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter(mock_test_data))
        mock_test.__len__ = MagicMock(return_value=len(mock_test_data))

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader(split="test")
        entries = loader._get_benchmark_entries("test")

        assert len(entries) == 1
        assert entries[0].question_id == "test_q1"

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.concatenate_datasets")
    def test_get_benchmark_entries_all_splits(self, mock_concat, mock_load_dataset):
        """Test loading benchmark entries with None split (all data)."""
        mock_train_data = [
            {
                "id": "train_q1",
                "question": "Train question?",
                "answer": "Train answer",
                "type": "bridge",
                "level": "easy",
                "context": {"title": ["Doc1"], "sentences": [["Content."]]},
                "supporting_facts": {"title": ["Doc1"], "sent_id": [0]},
            }
        ]

        mock_val_data = [
            {
                "id": "val_q1",
                "question": "Val question?",
                "answer": "Val answer",
                "type": "comparison",
                "level": "medium",
                "context": {"title": ["Doc2"], "sentences": [["Content."]]},
                "supporting_facts": {"title": ["Doc2"], "sent_id": [0]},
            }
        ]

        mock_test_data = [
            {
                "id": "test_q1",
                "question": "Test question?",
                "answer": "Test answer",
                "type": "bridge",
                "level": "hard",
                "context": {"title": ["Doc3"], "sentences": [["Content."]]},
                "supporting_facts": {"title": ["Doc3"], "sent_id": [0]},
            }
        ]

        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter(mock_train_data))
        mock_train.__len__ = MagicMock(return_value=1)

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter(mock_val_data))
        mock_val.__len__ = MagicMock(return_value=1)

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter(mock_test_data))
        mock_test.__len__ = MagicMock(return_value=1)

        # Mock concatenated dataset
        mock_combined = MagicMock()
        mock_combined.__iter__ = MagicMock(
            return_value=iter(mock_train_data + mock_val_data + mock_test_data)
        )
        mock_combined.__len__ = MagicMock(return_value=3)
        mock_concat.return_value = mock_combined

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader(split=None)
        entries = loader._get_benchmark_entries(None)

        assert len(entries) == 3
        mock_concat.assert_called_once()

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_benchmark_entries_deduplicate_supporting_facts(self, mock_load_dataset):
        """Test that duplicate supporting fact titles are deduplicated."""
        mock_train_data = [
            {
                "id": "q1",
                "question": "Question?",
                "answer": "Answer",
                "type": "bridge",
                "level": "easy",
                "context": {
                    "title": ["Doc1", "Doc2"],
                    "sentences": [["Sentence 1.", "Sentence 2."], ["Sentence 3."]],
                },
                # Supporting facts reference Doc1 multiple times (different sentences)
                "supporting_facts": {
                    "title": ["Doc1", "Doc1", "Doc2"],
                    "sent_id": [0, 1, 0],
                },
            }
        ]

        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter(mock_train_data))
        mock_train.__len__ = MagicMock(return_value=1)

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader(split="train")
        entries = loader._get_benchmark_entries("train")

        entry = entries[0]
        # Should have only 2 unique ground truth context IDs (Doc1 and Doc2)
        assert len(entry.ground_truth_context_ids) == 2
        gt_doc_ids = {ctx.document_id for ctx in entry.ground_truth_context_ids}
        assert gt_doc_ids == {"Doc1", "Doc2"}


class TestHotpotQaDataLoaderIntegration:
    """Test suite for integration with parent class functionality."""

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_get_corpus_returns_rag_corpus(self, mock_load_dataset):
        """Test that get_corpus returns a RagCorpus instance."""
        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "Question?",
                        "answer": "Answer",
                        "type": "bridge",
                        "level": "easy",
                        "context": {
                            "title": ["Doc1"],
                            "sentences": [["Content."]],
                        },
                        "supporting_facts": {"title": ["Doc1"], "sent_id": [0]},
                    }
                ]
            )
        )

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader()
        corpus = loader.get_corpus()

        assert isinstance(corpus, RagCorpus)
        assert len(corpus) == 1

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_get_benchmark_returns_rag_benchmark(self, mock_load_dataset):
        """Test that get_benchmark returns a RagBenchmark instance."""
        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "Question?",
                        "answer": "Answer",
                        "type": "bridge",
                        "level": "easy",
                        "context": {
                            "title": ["Doc1"],
                            "sentences": [["Content."]],
                        },
                        "supporting_facts": {"title": ["Doc1"], "sent_id": [0]},
                    }
                ]
            )
        )

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader()
        benchmark = loader.get_benchmark()

        assert isinstance(benchmark, RagBenchmark)
        assert len(benchmark) == 1

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_sampling_integration(self, mock_load_dataset):
        """Test that sampling parameters are applied correctly."""
        # Create 10 questions with various documents
        mock_train_data = [
            {
                "id": f"q{i}",
                "question": f"Question {i}?",
                "answer": f"Answer {i}",
                "type": "bridge" if i % 2 == 0 else "comparison",
                "level": "easy",
                "context": {
                    "title": [f"Doc{i}", f"Doc{i + 10}"],
                    "sentences": [[f"Content {i}."], [f"Content {i + 10}."]],
                },
                "supporting_facts": {"title": [f"Doc{i}"], "sent_id": [0]},
            }
            for i in range(10)
        ]

        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter(mock_train_data))
        mock_train.__len__ = MagicMock(return_value=10)

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        # Apply sampling: limit to 3 questions
        sampling_params = DataSamplingParams(question_limit=3, seed=42)
        loader = HotpotQaDataLoader(split="train", sampling_params=sampling_params)

        benchmark = loader.get_benchmark()
        corpus = loader.get_corpus()

        # Should have exactly 3 questions
        assert len(benchmark) == 3

        # All ground truth documents should be in corpus
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        assert gt_doc_ids.issubset(corpus_doc_ids)

    @patch("ragbench.datasets_loader.hotpot_qa_data_loader.load_dataset")
    def test_consistency_of_multiple_calls(self, mock_load_dataset):
        """Test that multiple calls return the same instances."""
        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "Question?",
                        "answer": "Answer",
                        "type": "bridge",
                        "level": "easy",
                        "context": {
                            "title": ["Doc1"],
                            "sentences": [["Content."]],
                        },
                        "supporting_facts": {"title": ["Doc1"], "sent_id": [0]},
                    }
                ]
            )
        )

        mock_val = MagicMock()
        mock_val.__iter__ = MagicMock(return_value=iter([]))

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))

        mock_load_dataset.return_value = {
            "train": mock_train,
            "validation": mock_val,
            "test": mock_test,
        }

        loader = HotpotQaDataLoader()

        corpus1 = loader.get_corpus()
        corpus2 = loader.get_corpus()
        assert corpus1 is corpus2

        benchmark1 = loader.get_benchmark()
        benchmark2 = loader.get_benchmark()
        assert benchmark1 is benchmark2
