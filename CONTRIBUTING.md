# Contributing to LocalMod

Thank you for your interest in contributing to LocalMod! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/localmod.git
   cd localmod
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[all]"
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run fast tests only (skip ML model downloads)
make test-fast

# Run specific test file
pytest tests/test_pii.py -v
```

### Code Style

We use:
- `flake8` for linting
- `mypy` for type checking
- `black` for formatting (optional)

```bash
# Check code style
make lint

# Format code
make format
```

### Pre-commit Hooks

```bash
pre-commit install
```

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add support for custom PII patterns

- Added PIIDetector.add_pattern() method
- Updated documentation
- Added tests for custom patterns
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring

## Adding a New Classifier

1. Create `src/localmod/classifiers/your_classifier.py`:

```python
from localmod.models.base import BaseClassifier, ClassificationResult, Severity

class YourClassifier(BaseClassifier):
    name = "your_classifier"
    version = "1.0.0"
    
    def load(self) -> None:
        # Load model/patterns
        pass
    
    def predict(self, text: str) -> ClassificationResult:
        # Implement prediction logic
        pass
```

2. Register in `src/localmod/classifiers/__init__.py`

3. Add tests in `tests/test_classifiers/test_your_classifier.py`

4. Update documentation

## Pull Request Process

1. Ensure tests pass: `make test`
2. Update documentation if needed
3. Create a pull request with a clear description
4. Wait for review

## Questions?

Open an issue for questions or discussions.

