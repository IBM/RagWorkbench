"""
Unit tests for BioasqDataLoader.

This module tests the BioASQ data loader implementation with mocked
HuggingFace datasets to avoid requiring internet access. Tests cover:
- Initialization
- Document loading from text corpus
- Benchmark entry loading from question-answer-passages
- Split handling (train, test, None)
- Data format validation
- Integration with parent class functionality
"""

from io import BytesIO
from unittest.mock import MagicMock, patch

from ragbench.datasets_loader.bioasq_data_loader import BioasqDataLoader
from ragbench.datasets_loader.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets_loader.data_models.dataset_names import DatasetName
from ragbench.datasets_loader.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets_loader.data_models.rag_corpus import RagCorpus


class TestBioasqDataLoaderInitialization:
    """Test suite for BioasqDataLoader initialization."""

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_initialization_with_defaults(self, mock_load_dataset):
        """Test initialization with default parameters."""
        # Mock the HuggingFace dataset
        mock_corpus = MagicMock()
        mock_corpus.__len__ = MagicMock(return_value=10)
        mock_corpus.__iter__ = MagicMock(
            return_value=iter([{"id": "0", "passage": "Test passage 0"}])
        )

        mock_qa = MagicMock()
        mock_qa.__len__ = MagicMock(return_value=5)
        mock_qa.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q0",
                        "question": "Test question?",
                        "answer": "Test answer",
                        "relevant_passage_ids": ["0"],
                    }
                ]
            )
        )

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},  # For corpus
            {"train": mock_qa, "test": mock_qa},  # For benchmark
        ]

        loader = BioasqDataLoader()

        assert loader.dataset_name == DatasetName.BIOASQ
        assert loader.split is None
        assert isinstance(loader.sampling_params, DataSamplingParams)
        assert loader.dataset_path is None

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_initialization_with_custom_parameters(self, mock_load_dataset):
        """Test initialization with custom parameters."""
        # Mock the HuggingFace dataset
        mock_corpus = MagicMock()
        mock_corpus.__len__ = MagicMock(return_value=10)
        mock_corpus.__iter__ = MagicMock(
            return_value=iter([{"id": "0", "passage": "Test passage 0"}])
        )

        mock_qa = MagicMock()
        mock_qa.__len__ = MagicMock(return_value=5)
        mock_qa.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q0",
                        "question": "Test question?",
                        "answer": "Test answer",
                        "relevant_passage_ids": ["0"],
                    }
                ]
            )
        )

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_qa, "test": mock_qa},
        ]

        sampling_params = DataSamplingParams(question_limit=5, seed=42)
        loader = BioasqDataLoader(
            dataset_name=DatasetName.BIOASQ,
            split="train",
            sampling_params=sampling_params,
            dataset_path="/custom/path",
        )

        assert loader.dataset_name == DatasetName.BIOASQ
        assert loader.split == "train"
        assert loader.sampling_params.question_limit == 5
        assert loader.dataset_path == "/custom/path"


class TestBioasqDataLoaderDocumentLoading:
    """Test suite for document loading functionality."""

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_get_documents_loads_from_huggingface(self, mock_load_dataset):
        """Test that _get_documents loads from HuggingFace correctly."""
        # Mock corpus data
        mock_corpus_data = [
            {"id": "doc_0", "passage": "First biomedical passage"},
            {"id": "doc_1", "passage": "Second biomedical passage"},
            {"id": "doc_2", "passage": "Third biomedical passage"},
        ]

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(return_value=iter(mock_corpus_data))
        mock_corpus.__len__ = MagicMock(return_value=len(mock_corpus_data))

        mock_qa = MagicMock()
        mock_qa.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q0",
                        "question": "Test?",
                        "answer": "Answer",
                        "relevant_passage_ids": ["doc_0"],
                    }
                ]
            )
        )
        mock_qa.__len__ = MagicMock(return_value=1)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_qa, "test": mock_qa},
        ]

        loader = BioasqDataLoader()
        documents = loader._get_documents()

        assert len(documents) == 3
        assert all(doc.name in ["doc_0", "doc_1", "doc_2"] for doc in documents)
        assert all(doc.mime_type == "text/plain" for doc in documents)

        # Verify content
        doc_dict = {doc.name: doc for doc in documents}
        assert doc_dict["doc_0"].stream.read() == b"First biomedical passage"

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_get_documents_handles_integer_ids(self, mock_load_dataset):
        """Test that document IDs are converted to strings."""
        # Mock corpus with integer IDs
        mock_corpus_data = [
            {"id": 123, "passage": "Passage with integer ID"},
            {"id": 456, "passage": "Another passage"},
        ]

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(return_value=iter(mock_corpus_data))
        mock_corpus.__len__ = MagicMock(return_value=len(mock_corpus_data))

        mock_qa = MagicMock()
        mock_qa.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q0",
                        "question": "Test?",
                        "answer": "Answer",
                        "relevant_passage_ids": ["123"],
                    }
                ]
            )
        )
        mock_qa.__len__ = MagicMock(return_value=1)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_qa, "test": mock_qa},
        ]

        loader = BioasqDataLoader()
        documents = loader._get_documents()

        # IDs should be converted to strings
        assert all(isinstance(doc.name, str) for doc in documents)
        assert any(doc.name == "123" for doc in documents)
        assert any(doc.name == "456" for doc in documents)

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_get_documents_creates_valid_document_objects(self, mock_load_dataset):
        """Test that created DocumentObjects are valid."""
        mock_corpus_data = [
            {"id": "test_doc", "passage": "Test passage content"},
        ]

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(return_value=iter(mock_corpus_data))
        mock_corpus.__len__ = MagicMock(return_value=1)

        mock_qa = MagicMock()
        mock_qa.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q0",
                        "question": "Test?",
                        "answer": "Answer",
                        "relevant_passage_ids": ["test_doc"],
                    }
                ]
            )
        )
        mock_qa.__len__ = MagicMock(return_value=1)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_qa, "test": mock_qa},
        ]

        loader = BioasqDataLoader()
        documents = loader._get_documents()

        doc = documents[0]
        assert doc.name == "test_doc"
        assert doc.mime_type == "text/plain"
        assert isinstance(doc.stream, BytesIO)
        assert doc.stream.read() == b"Test passage content"


class TestBioasqDataLoaderBenchmarkLoading:
    """Test suite for benchmark entry loading functionality."""

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_get_benchmark_entries_train_split(self, mock_load_dataset):
        """Test loading benchmark entries with train split."""
        mock_train_data = [
            {
                "id": "train_q1",
                "question": "What is protein folding?",
                "answer": "Process of protein structure formation",
                "relevant_passage_ids": ["doc_1", "doc_2"],
            },
            {
                "id": "train_q2",
                "question": "What causes diabetes?",
                "answer": "Insulin deficiency or resistance",
                "relevant_passage_ids": ["doc_3"],
            },
        ]

        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter(mock_train_data))
        mock_train.__len__ = MagicMock(return_value=len(mock_train_data))

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter([]))
        mock_test.__len__ = MagicMock(return_value=0)

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"id": "doc_1", "passage": "Passage 1"},
                    {"id": "doc_2", "passage": "Passage 2"},
                    {"id": "doc_3", "passage": "Passage 3"},
                ]
            )
        )
        mock_corpus.__len__ = MagicMock(return_value=3)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_train, "test": mock_test},
        ]

        loader = BioasqDataLoader(split="train")
        entries = loader._get_benchmark_entries("train")

        assert len(entries) == 2
        assert entries[0].question_id == "train_q1"
        assert entries[0].question == "What is protein folding?"
        assert entries[0].ground_truth_answers == [
            "Process of protein structure formation"
        ]
        assert len(entries[0].ground_truth_context_ids) == 2
        assert entries[0].is_answerable is True

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_get_benchmark_entries_test_split(self, mock_load_dataset):
        """Test loading benchmark entries with test split."""
        mock_test_data = [
            {
                "id": "test_q1",
                "question": "What is DNA?",
                "answer": "Deoxyribonucleic acid",
                "relevant_passage_ids": ["doc_5"],
            }
        ]

        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter([]))
        mock_train.__len__ = MagicMock(return_value=0)

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter(mock_test_data))
        mock_test.__len__ = MagicMock(return_value=len(mock_test_data))

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(
            return_value=iter([{"id": "doc_5", "passage": "DNA passage"}])
        )
        mock_corpus.__len__ = MagicMock(return_value=1)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_train, "test": mock_test},
        ]

        loader = BioasqDataLoader(split="test")
        entries = loader._get_benchmark_entries("test")

        assert len(entries) == 1
        assert entries[0].question_id == "test_q1"

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    @patch("ragbench.datasets_loader.bioasq_data_loader.concatenate_datasets")
    def test_get_benchmark_entries_all_splits(self, mock_concat, mock_load_dataset):
        """Test loading benchmark entries with None split (all data)."""
        mock_train_data = [
            {
                "id": "train_q1",
                "question": "Train question?",
                "answer": "Train answer",
                "relevant_passage_ids": ["doc_1"],
            }
        ]

        mock_test_data = [
            {
                "id": "test_q1",
                "question": "Test question?",
                "answer": "Test answer",
                "relevant_passage_ids": ["doc_2"],
            }
        ]

        mock_train = MagicMock()
        mock_train.__iter__ = MagicMock(return_value=iter(mock_train_data))
        mock_train.__len__ = MagicMock(return_value=1)

        mock_test = MagicMock()
        mock_test.__iter__ = MagicMock(return_value=iter(mock_test_data))
        mock_test.__len__ = MagicMock(return_value=1)

        # Mock concatenated dataset
        mock_combined = MagicMock()
        mock_combined.__iter__ = MagicMock(
            return_value=iter(mock_train_data + mock_test_data)
        )
        mock_combined.__len__ = MagicMock(return_value=2)
        mock_concat.return_value = mock_combined

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"id": "doc_1", "passage": "Passage 1"},
                    {"id": "doc_2", "passage": "Passage 2"},
                ]
            )
        )
        mock_corpus.__len__ = MagicMock(return_value=2)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_train, "test": mock_test},
        ]

        loader = BioasqDataLoader(split=None)
        entries = loader._get_benchmark_entries(None)

        assert len(entries) == 2
        mock_concat.assert_called_once()

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_benchmark_entries_have_correct_structure(self, mock_load_dataset):
        """Test that benchmark entries have the correct structure."""
        mock_qa_data = [
            {
                "id": "q1",
                "question": "What is apoptosis?",
                "answer": "Programmed cell death",
                "relevant_passage_ids": ["doc_10", "doc_11"],
            }
        ]

        mock_qa = MagicMock()
        mock_qa.__iter__ = MagicMock(return_value=iter(mock_qa_data))
        mock_qa.__len__ = MagicMock(return_value=1)

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"id": "doc_10", "passage": "Passage 10"},
                    {"id": "doc_11", "passage": "Passage 11"},
                ]
            )
        )
        mock_corpus.__len__ = MagicMock(return_value=2)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_qa, "test": mock_qa},
        ]

        loader = BioasqDataLoader(split="train")
        entries = loader._get_benchmark_entries("train")

        entry = entries[0]
        assert entry.question_id == "q1"
        assert entry.question == "What is apoptosis?"
        assert entry.ground_truth_answers == ["Programmed cell death"]
        assert len(entry.ground_truth_context_ids) == 2
        assert entry.ground_truth_context_ids[0].document_id == "doc_10"
        assert entry.ground_truth_context_ids[1].document_id == "doc_11"
        assert entry.is_answerable is True
        assert entry.additional_information["source"] == "bioasq"
        assert entry.additional_information["question_type"] == "factoid"


class TestBioasqDataLoaderIntegration:
    """Test suite for integration with parent class functionality."""

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_get_corpus_returns_rag_corpus(self, mock_load_dataset):
        """Test that get_corpus returns a RagCorpus instance."""
        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(
            return_value=iter([{"id": "doc_1", "passage": "Passage 1"}])
        )
        mock_corpus.__len__ = MagicMock(return_value=1)

        mock_qa = MagicMock()
        mock_qa.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "Question?",
                        "answer": "Answer",
                        "relevant_passage_ids": ["doc_1"],
                    }
                ]
            )
        )
        mock_qa.__len__ = MagicMock(return_value=1)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_qa, "test": mock_qa},
        ]

        loader = BioasqDataLoader()
        corpus = loader.get_corpus()

        assert isinstance(corpus, RagCorpus)
        assert len(corpus) == 1

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_get_benchmark_returns_rag_benchmark(self, mock_load_dataset):
        """Test that get_benchmark returns a RagBenchmark instance."""
        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(
            return_value=iter([{"id": "doc_1", "passage": "Passage 1"}])
        )
        mock_corpus.__len__ = MagicMock(return_value=1)

        mock_qa = MagicMock()
        mock_qa.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "Question?",
                        "answer": "Answer",
                        "relevant_passage_ids": ["doc_1"],
                    }
                ]
            )
        )
        mock_qa.__len__ = MagicMock(return_value=1)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_qa, "test": mock_qa},
        ]

        loader = BioasqDataLoader()
        benchmark = loader.get_benchmark()

        assert isinstance(benchmark, RagBenchmark)
        assert len(benchmark) == 1

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_sampling_integration(self, mock_load_dataset):
        """Test that sampling parameters are applied correctly."""
        # Create 10 documents and 8 questions
        mock_corpus_data = [
            {"id": f"doc_{i}", "passage": f"Passage {i}"} for i in range(10)
        ]

        mock_qa_data = [
            {
                "id": f"q{i}",
                "question": f"Question {i}?",
                "answer": f"Answer {i}",
                "relevant_passage_ids": [f"doc_{i % 5}"],  # Use first 5 docs as GT
            }
            for i in range(8)
        ]

        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(return_value=iter(mock_corpus_data))
        mock_corpus.__len__ = MagicMock(return_value=10)

        mock_qa = MagicMock()
        mock_qa.__iter__ = MagicMock(return_value=iter(mock_qa_data))
        mock_qa.__len__ = MagicMock(return_value=8)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_qa, "test": mock_qa},
        ]

        # Apply sampling: limit to 3 questions
        sampling_params = DataSamplingParams(question_limit=3, seed=42)
        loader = BioasqDataLoader(split="train", sampling_params=sampling_params)

        benchmark = loader.get_benchmark()
        corpus = loader.get_corpus()

        # Should have exactly 3 questions
        assert len(benchmark) == 3

        # All ground truth documents should be in corpus
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        assert gt_doc_ids.issubset(corpus_doc_ids)

    @patch("ragbench.datasets_loader.bioasq_data_loader.load_dataset")
    def test_consistency_of_multiple_calls(self, mock_load_dataset):
        """Test that multiple calls return the same instances."""
        mock_corpus = MagicMock()
        mock_corpus.__iter__ = MagicMock(
            return_value=iter([{"id": "doc_1", "passage": "Passage 1"}])
        )
        mock_corpus.__len__ = MagicMock(return_value=1)

        mock_qa = MagicMock()
        mock_qa.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "id": "q1",
                        "question": "Question?",
                        "answer": "Answer",
                        "relevant_passage_ids": ["doc_1"],
                    }
                ]
            )
        )
        mock_qa.__len__ = MagicMock(return_value=1)

        mock_load_dataset.side_effect = [
            {"test": mock_corpus},
            {"train": mock_qa, "test": mock_qa},
        ]

        loader = BioasqDataLoader()

        corpus1 = loader.get_corpus()
        corpus2 = loader.get_corpus()
        assert corpus1 is corpus2

        benchmark1 = loader.get_benchmark()
        benchmark2 = loader.get_benchmark()
        assert benchmark1 is benchmark2
