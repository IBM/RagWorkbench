# Contributing to RAGWorkbench

We welcome contributions to RAGWorkbench! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [License](#license)

## Code of Conduct

This project adheres to the IBM Open Source Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RagWorkbench.git
   cd RagWorkbench
   ```
3. Set up your development environment:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Describe the bug in detail
- Include steps to reproduce
- Specify your environment (OS, Python version, etc.)

### Suggesting Enhancements

- Use the GitHub issue tracker
- Clearly describe the enhancement and its benefits
- Provide examples if possible

### Contributing Code

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes following our [coding standards](#coding-standards)
3. Add or update tests as needed
4. Ensure all tests pass:
   ```bash
   pytest
   ```
5. Run pre-commit hooks:
   ```bash
   pre-commit run --all-files
   ```
6. Commit your changes with a DCO sign-off (see below)
7. Push to your fork and submit a pull request

## Developer Certificate of Origin (DCO)

All contributions to this project must be accompanied by a Developer Certificate of Origin (DCO) sign-off. This certifies that you have the right to submit the contribution and agree to the terms of the [DCO](https://developercertificate.org/).

### DCO Sign-Off

To sign off on a commit, add the `-s` flag to your git commit command:

```bash
git commit -s -m "Your commit message"
```

This adds a "Signed-off-by" line to your commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

### DCO Requirements

- Every commit must be signed off
- The sign-off must use your real name (no pseudonyms or anonymous contributions)
- The email address must match the email in your Git configuration

### Amending Commits

If you forgot to sign off a commit, you can amend it:

```bash
git commit --amend -s
```

For multiple commits, you can rebase and sign off:

```bash
git rebase --signoff HEAD~N  # where N is the number of commits
```

## Pull Request Process

1. **Ensure DCO Sign-Off**: All commits must include a DCO sign-off
2. **Update Documentation**: Update the README.md or other docs if needed
3. **Add Tests**: Include tests for new features or bug fixes
4. **Pass CI Checks**: Ensure all automated checks pass
5. **Code Review**: Address feedback from maintainers
6. **Squash Commits**: Consider squashing commits before merge (maintainers may do this)

### Pull Request Checklist

- [ ] All commits are signed off with DCO
- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] No merge conflicts

## Coding Standards

### Python Style

- Follow PEP 8 style guide
- Use Black for code formatting (line length: 88)
- Use Ruff for linting
- Use type hints where appropriate
- Use MyPy for type checking

### Code Quality Tools

The project uses pre-commit hooks to enforce code quality:

- **Black**: Code formatting
- **Ruff**: Linting and import sorting
- **MyPy**: Static type checking

Run these tools before committing:

```bash
pre-commit run --all-files
```

### Documentation

- Add docstrings to all public functions, classes, and modules
- Use Google-style docstrings
- Update README.md for user-facing changes
- Add inline comments for complex logic

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Reference issue numbers when applicable
- Include DCO sign-off

Example:
```
Add support for new dataset format

- Implement DataLoader for XYZ dataset
- Add tests for new functionality
- Update documentation

Fixes #123

Signed-off-by: Your Name <your.email@example.com>
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/path/to/test_file.py

# Run only unit tests
pytest tests/datasets_loader/unit

# Run only integration tests
pytest -m integration
```

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Follow the existing test structure
- Use fixtures for common test data

## License

By contributing to RAGWorkbench, you agree that your contributions will be licensed under the Apache License 2.0. All source files must include the Apache 2.0 license header.

## Questions?

If you have questions about contributing, please:
- Open an issue on GitHub
- Contact the maintainers listed in the README.md

## Acknowledgments

Thank you for contributing to RAGWorkbench! Your contributions help make this project better for everyone.

---

**Important**: All contributions must include a DCO sign-off. Contributions without proper sign-off cannot be accepted.
