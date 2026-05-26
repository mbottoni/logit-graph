# Contributing to Logit Graph

Thank you for your interest in contributing! This project welcomes bug reports, documentation improvements, tests, and feature proposals.

## Quick start

```bash
git clone https://github.com/mbottoni/logit-graph.git
cd logit-graph
make install-dev    # .venv + editable install + pytest/ruff/mypy
make test           # run the test suite
```

Manual install:

```bash
pip install -e ".[viz,notebook,progress]"
pip install pytest ruff mypy
pytest -q
```

## Development workflow

| Task | Command |
|------|---------|
| Run tests | `make test` |
| Coverage | `make test-cov` |
| Lint | `make lint` |
| Format | `make format` |
| Type check | `make typecheck` |
| All checks | `make check` |
| Build wheel | `make build` |

CI runs on **Python 3.10 and 3.11** (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Make focused changes with tests when behavior changes.
3. Run `make check` (or at minimum `make test`).
4. Update [`CHANGELOG.md`](CHANGELOG.md) under **Unreleased** if user-facing.
5. Open a PR using the [pull request template](.github/pull_request_template.md).

## Notebooks

- **PyPI tutorials** → [`examples/`](examples/) (self-contained, install from PyPI).
- **Research / batch runs** → [`notebooks/`](notebooks/) (repo checkout + local data).

When adding analysis notebooks, place them in the appropriate `notebooks/` subdirectory. Batch platform fits live under `notebooks/refactors/`.

## Code style

- Match existing naming and import style in `src/logit_graph/`.
- Prefer minimal, focused diffs.
- Ruff is the linter/formatter (`make lint`, `make format`).

## Reporting issues

Use [GitHub Issues](https://github.com/mbottoni/logit-graph/issues) with the bug or feature template. For reproducible GIC mismatches, include `random_state`, package version (`pip show logit-graph`), and a minimal graph or seed.

## Releases

Maintainers bump `version` in `pyproject.toml`, update `CHANGELOG.md`, tag (`v0.x.y`), and publish to PyPI with `make publish`.
