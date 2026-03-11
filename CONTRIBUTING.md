# Contributing to prompt-hunter

Thank you for your interest in contributing! This document provides guidelines
for submitting changes.

## Development Setup

```bash
# Clone
git clone https://github.com/alanpeng/prompt-hunter.git
cd prompt-hunter

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Standards

- **Type hints** — all public functions must have complete type annotations.
- **Docstrings** — use NumPy-style docstrings for all public classes and methods.
- **Linting** — `ruff check src/` must pass with zero warnings.
- **Formatting** — `ruff format src/` is enforced via pre-commit.
- **Tests** — all new features must include unit tests. Run `pytest` before submitting.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Run the full suite locally**:
   ```bash
   ruff check src/
   ruff format --check src/
   pytest --cov=prompt_hunter
   ```
4. **Write a clear PR description** explaining *what* changed and *why*.
5. Keep commits atomic and well-described.

## Issue Reports

When reporting bugs, please include:
- Python version and OS
- Full traceback
- Minimal reproduction steps
- Input file format (if applicable)

## Architecture Overview

```
src/prompt_hunter/
├── __init__.py     # Public API
├── cli.py          # Command-line interface
├── cropper.py      # Instance cropping from annotations
├── evaluator.py    # Grounding-model prompt scoring
├── hunter.py       # High-level pipeline orchestrator
└── miner.py        # VLM-based prompt candidate generation
```

The pipeline flows: **Cropper → Miner → Evaluator**.
Each module can also be used independently via the Python API.
