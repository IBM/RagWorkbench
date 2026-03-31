# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] - 2026-03-31

### Added
-

### Changed
-

### Fixed
-

## [0.1.1] - 2026-03-24

### Added
-

### Changed
-

### Fixed
-

## [0.1.0-beta.2] - 2026-03-22

### Added
-

### Changed
-

### Fixed
-

## [0.1.0-beta.1] - 2026-03-22

### Added
-

### Changed
-

### Fixed
-

### Added
- Initial release mechanism with GitHub Actions workflow
- Automated PyPI publishing on version tags
- GitHub Release creation with release notes

## [0.1.0] - 2024-XX-XX

### Added
- Initial release of RAGWorkbench
- Support for 18+ RAG benchmark datasets (AIT-QA, BioASQ, HotpotQA, NarrativeQA, QASPER, etc.)
- Comprehensive evaluation metrics for context and answer correctness
- Modular pipeline architecture for ingest and inference
- File-system based caching for data loading, generation, and evaluation
- Interactive web-based dataset explorer
- End-to-end experiment management
- Integration with Unitxt for evaluation metrics
- Support for multiple modalities (TEXT, TABLE, IMAGE)
- Multi-hop retrieval support
- Multilingual dataset support (MLDR)

### Features
- **Data Loaders**: Factory pattern for loading diverse RAG benchmarks
- **Evaluation Metrics**:
  - Context correctness: Retrieval@K, MRR, MAP
  - Answer correctness: BERT Score, Sentence-BERT, LLM-as-a-Judge
- **Caching System**: Smart caching for data, generation, and evaluation
- **Dataset Explorer**: Interactive NiceGUI-based exploration tool
- **Experiment Framework**: Complete workflow orchestration

### Documentation
- Comprehensive README with quick start guide
- Environment setup documentation
- API documentation for core components
- Dataset exploration guide

### Development
- Pre-commit hooks for code quality
- Pytest test suite with coverage
- Type checking with mypy
- Code formatting with black and ruff
- CI/CD pipeline with GitHub Actions

[Unreleased]: https://github.com/IBM/RagWorkbench/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/IBM/RagWorkbench/releases/tag/v0.1.0
