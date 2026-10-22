# Contributing to `KBank`

Contributions are welcome, and they are greatly appreciated! Every improvement — from curated bacterial polysaccharide structures and parser edge cases to biosynthetic alignment rules and MPNN predictors — helps advance open bacterial glycomics.

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/SeroVerse/KBank/issues.

If you are reporting a bug, please include:

- Your operating system name, Python version (`>=3.14`), and `serogen`/`numba` versions.
- Minimal `CSDB` / `IUPAC` / `KEGG` glycan strings or FASTA inputs sufficient to reproduce the issue.
- Expected chemical or biosynthetic behavior vs. actual output.

### Fix Bugs & Implement Features

Look through the GitHub issues for open bugs or enhancements at https://github.com/SeroVerse/KBank/issues. If you are proposing a new feature:

- Explain how it fits into `KBank`'s modular architecture (`kbank.glycan`, `kbank.sugars`, `kbank.io`, `kbank.db`, `kbank.biosynthesis`).
- Keep hot-path graph and sequence containers in contiguous Structure-of-Arrays (SoA) NumPy buffers (`Glycan`, `Glycans`, `GlycanRecords`) compatible with `serogen.core`.
- Preserve strict module boundaries (keep `__init__.py` files empty and use explicit submodule imports).

## Local Development Setup

`KBank` uses [`uv`](https://docs.astral.sh/uv/) for dependency and environment management.

1. Fork and clone the repository:

```bash
git clone git@github.com:YOUR_USERNAME/KBank.git
cd KBank
```

2. Sync all development, test, and documentation dependency groups:

```bash
uv sync --all-groups
```

3. Create a feature branch for your changes:

```bash
git checkout -b feat/your-feature-name
```

## Quality Checks & Testing

Before opening a pull request, verify that formatting, linting, static type checking, unit tests, and strict documentation builds all pass:

```bash
# 1. Lint and check formatting with Ruff
uv run ruff check .
uv run ruff format --check .

# 2. Run static type checking with ty
uv run ty check src/

# 3. Run the pytest test suite
uv run pytest

# 4. Build the Zensical documentation site in strict mode
uv run zensical build --clean --strict
```

## Pull Request Guidelines

1. Include unit tests in `tests/` covering any new parsers, biochemical rules, or predictors.
2. Document public classes, methods, and dataclasses using Google-style docstrings so `mkdocstrings` renders complete API references.
3. Ensure all CI checks (`ci.yml` and `docs.yml`) pass cleanly.
