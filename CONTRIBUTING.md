# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Your Python version (`python --version`) and package versions (`pip freeze`).
* A minimal, reproducible example that triggers the issue.
* The full traceback if applicable.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or even
on the web in blog posts, articles, and such.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started

Ready to contribute? Here's how to set up `dgufs` for local development.

1. Fork and clone the repository:

    ```bash
    git clone https://github.com/gsel9/dgufs.git
    cd dgufs
    ```

2. Install in editable mode with the testing extras:

    ```bash
    pip install -e ".[testing]"
    ```

3. Create a branch for your changes:

    ```bash
    git checkout -b name-of-your-bugfix-or-feature
    ```

4. Make your changes. Run the test suite to check nothing is broken:

    ```bash
    pytest
    ```

5. Commit your changes and open a pull request against `master`.

## Code Style

This project uses [Black](https://black.readthedocs.io) for formatting and
[Ruff](https://docs.astral.sh/ruff/) for linting. Before opening a PR, run:

```bash
pip install black ruff
black src/ tests/
ruff src/ tests/
```

Pre-commit hooks are also available:

```bash
pip install pre-commit
pre-commit install
```

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. New or changed behaviour must be covered by tests under `tests/`.
2. If the pull request adds functionality, update the docstrings and README accordingly.
3. The CI pipeline must pass (tests + linting).

## Code of Conduct

Please note that the `dgufs` project is released with a
Code of Conduct. By contributing to this project you agree to abide by its terms.
