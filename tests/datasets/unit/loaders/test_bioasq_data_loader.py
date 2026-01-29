"""
Tests for BioasqDataLoader implementation.

This module tests the BioasqDataLoader class, which provides a concrete
implementation for loading the BioASQ biomedical question answering dataset.
"""

from ragbench.datasets.data_models.data_sampling_params import DataSamplingParams
from ragbench.datasets.data_models.dataset_names import DatasetName
from ragbench.datasets.data_models.rag_benchmark import RagBenchmark
from ragbench.datasets.data_models.rag_corpus import RagCorpus
from src.ragbench.datasets.bioasq_data_loader import BioasqDataLoader


class TestBioasqDataLoaderInitialization:
    """Test suite for BioasqDataLoader initialization."""

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        loader = BioasqDataLoader()

        assert loader.dataset_name == DatasetName.BIOASQ
        assert loader.split is None
        assert isinstance(loader.get_corpus(), RagCorpus)
        assert isinstance(loader.get_benchmark(), RagBenchmark)

    def test_initialization_with_train_split(self):
        """Test initialization with train split."""
        loader = BioasqDataLoader(split="train")

        assert loader.split == "train"
        assert len(loader.get_benchmark()) > 0

    def test_initialization_with_test_split(self):
        """Test initialization with test split."""
        loader = BioasqDataLoader(split="test")

        assert loader.split == "test"
        assert len(loader.get_benchmark()) > 0

    def test_initialization_with_sampling_params(self):
        """Test initialization with sampling parameters."""
        sampling_params = DataSamplingParams(question_limit=3, seed=42)
        loader = BioasqDataLoader(sampling_params=sampling_params)

        benchmark = loader.get_benchmark()
        assert len(benchmark) == 3

    def test_initialization_with_dataset_path(self):
        """Test initialization with custom dataset path."""
        loader = BioasqDataLoader(dataset_path="/path/to/bioasq")

        assert loader.dataset_path == "/path/to/bioasq"


class TestBioasqDataLoaderDocuments:
    """Test suite for BioasqDataLoader document loading."""

    def test_get_documents_returns_list(self):
        """Test that _get_documents returns a list of DocumentObject."""
        loader = BioasqDataLoader()
        corpus = loader.get_corpus()

        assert isinstance(corpus, RagCorpus)
        assert len(corpus) > 0
        assert all(hasattr(doc, "name") for doc in corpus.documents)

    def test_documents_have_pubmed_format(self):
        """Test that documents follow PubMed ID naming convention."""
        loader = BioasqDataLoader()
        corpus = loader.get_corpus()

        # Check that document names follow PMID format
        for doc in corpus.documents:
            assert doc.name.startswith("PMID_")

    def test_documents_have_bioasq_metadata(self):
        """Test that documents contain BioASQ-specific metadata."""
        loader = BioasqDataLoader()
        corpus = loader.get_corpus()

        for doc in corpus.documents:
            assert "source" in doc.metadata
            assert doc.metadata["source"] == "bioasq"


class TestBioasqDataLoaderBenchmark:
    """Test suite for BioasqDataLoader benchmark loading."""

    def test_get_benchmark_entries_returns_list(self):
        """Test that _get_benchmark_entries returns a list of entries."""
        loader = BioasqDataLoader()
        benchmark = loader.get_benchmark()

        assert isinstance(benchmark, RagBenchmark)
        assert len(benchmark) > 0

    def test_benchmark_entries_have_bioasq_metadata(self):
        """Test that benchmark entries contain BioASQ-specific metadata."""
        loader = BioasqDataLoader()
        benchmark = loader.get_benchmark()

        entries = benchmark.get_benchmark_entries()
        for entry in entries:
            assert entry.additional_information is not None
            assert "source" in entry.additional_information
            assert entry.additional_information["source"] == "bioasq"

    def test_split_affects_benchmark_size(self):
        """Test that different splits return different numbers of entries."""
        loader_train = BioasqDataLoader(split="train")
        loader_test = BioasqDataLoader(split="test")
        loader_all = BioasqDataLoader(split=None)

        len_train = len(loader_train.get_benchmark())
        len_test = len(loader_test.get_benchmark())
        len_all = len(loader_all.get_benchmark())

        # Train and test should have different sizes
        assert len_train != len_test
        # All should be larger than individual splits
        assert len_all >= len_train
        assert len_all >= len_test

    def test_ground_truth_context_ids_reference_documents(self):
        """Test that ground truth context IDs reference actual documents."""
        loader = BioasqDataLoader()
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Get all document IDs from corpus
        corpus_doc_ids = {doc.name for doc in corpus.documents}

        # Get all ground truth document IDs from benchmark
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)

        # All ground truth documents should be in corpus
        assert gt_doc_ids.issubset(corpus_doc_ids)


class TestBioasqDataLoaderSampling:
    """Test suite for BioasqDataLoader sampling functionality."""

    def test_question_sampling(self):
        """Test that question sampling works correctly."""
        sampling_params = DataSamplingParams(question_limit=5, seed=42)
        loader = BioasqDataLoader(sampling_params=sampling_params)

        benchmark = loader.get_benchmark()
        assert len(benchmark) == 5

    def test_document_sampling(self):
        """Test that document sampling works correctly."""
        sampling_params = DataSamplingParams(
            question_limit=5, document_factor=2, seed=42
        )
        loader = BioasqDataLoader(sampling_params=sampling_params)

        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Get ground truth document IDs
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)

        # Corpus should include ground truth docs plus additional ones
        assert len(corpus) >= len(gt_doc_ids)

    def test_sampling_reproducibility(self):
        """Test that sampling is reproducible with same seed."""
        sampling_params_1 = DataSamplingParams(question_limit=5, seed=123)
        sampling_params_2 = DataSamplingParams(question_limit=5, seed=123)

        loader_1 = BioasqDataLoader(sampling_params=sampling_params_1)
        loader_2 = BioasqDataLoader(sampling_params=sampling_params_2)

        questions_1 = loader_1.get_benchmark().get_question_ids()
        questions_2 = loader_2.get_benchmark().get_question_ids()

        assert questions_1 == questions_2


class TestBioasqDataLoaderIntegration:
    """Test suite for BioasqDataLoader integration scenarios."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from initialization to data access."""
        sampling_params = DataSamplingParams(
            question_limit=5, document_factor=1, seed=42
        )
        loader = BioasqDataLoader(split="train", sampling_params=sampling_params)

        # Get corpus and benchmark
        corpus = loader.get_corpus()
        benchmark = loader.get_benchmark()

        # Verify basic properties
        assert len(benchmark) == 5
        assert len(corpus) > 0

        # Verify questions can be retrieved
        questions = benchmark.get_questions()
        assert len(questions) == 5

        # Verify ground truth documents are in corpus
        gt_doc_ids = RagBenchmark.get_doc_ids_set(benchmark.benchmark_entries)
        corpus_doc_ids = {doc.name for doc in corpus.documents}
        assert gt_doc_ids.issubset(corpus_doc_ids)

    def test_placeholder_warning_logged(self, caplog):
        """Test that placeholder implementation logs warnings."""
        import logging

        caplog.set_level(logging.WARNING)

        loader = BioasqDataLoader()

        # Check that warnings were logged about placeholder implementation
        assert any(
            "not fully implemented" in record.message for record in caplog.records
        )
        assert any("placeholder" in record.message.lower() for record in caplog.records)
