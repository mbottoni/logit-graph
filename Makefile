.DEFAULT_GOAL := help
SHELL  := /bin/bash
PYTHON := .venv/bin/python
UV     := uv

# ─────────────────────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────────────────────

.venv:  ## Create virtual environment
	$(UV) venv .venv --python 3.11

install: .venv  ## Install package in editable mode with all extras
	$(UV) pip install -e ".[viz,notebook,progress]" --python $(PYTHON)

install-dev: install  ## Install dev / test dependencies
	$(UV) pip install pytest pytest-cov ruff mypy --python $(PYTHON)

install-torch: install  ## Install with optional PyTorch support
	$(UV) pip install -e ".[torch]" --python $(PYTHON)

lock:  ## Regenerate uv.lock
	$(UV) lock

sync:  ## Sync environment from lockfile
	$(UV) sync

# ─────────────────────────────────────────────────────────────
#  Quality
# ─────────────────────────────────────────────────────────────

test:  ## Run test suite
	$(PYTHON) -m pytest tests/ -v

test-cov:  ## Run tests with coverage report
	$(PYTHON) -m pytest tests/ -v --cov=src/logit_graph --cov-report=term-missing

lint:  ## Lint source code with ruff
	$(PYTHON) -m ruff check src/ tests/

lint-fix:  ## Auto-fix lint issues
	$(PYTHON) -m ruff check src/ tests/ --fix

format:  ## Format code with ruff
	$(PYTHON) -m ruff format src/ tests/

typecheck:  ## Run mypy type checking
	$(PYTHON) -m mypy src/logit_graph --ignore-missing-imports

check: lint typecheck test  ## Run all checks (lint + types + tests)

# ─────────────────────────────────────────────────────────────
#  Build & Publish
# ─────────────────────────────────────────────────────────────

build: clean-dist  ## Build sdist and wheel
	$(UV) build

clean-dist:  ## Remove previous build artifacts
	rm -rf dist/

publish-test: build  ## Upload to TestPyPI
	$(UV) publish --index-url https://test.pypi.org/simple/

publish: build  ## Upload to PyPI
	$(UV) publish

# ─────────────────────────────────────────────────────────────
#  Notebooks
# ─────────────────────────────────────────────────────────────

nb-citation:  ## Run the citation network notebook
	$(PYTHON) -m jupyter nbconvert --to notebook --execute \
		notebooks/citation/0-citation.ipynb \
		--output 0-citation.ipynb \
		--ExecutePreprocessor.timeout=7200

nb-playground:  ## Run all playground test notebooks
	@for nb in notebooks/playground/*.ipynb; do \
		echo "── Running $$nb ──"; \
		$(PYTHON) -m jupyter nbconvert --to notebook --execute \
			"$$nb" --output "$$(basename $$nb)" \
			--ExecutePreprocessor.timeout=600 || exit 1; \
	done

# ─────────────────────────────────────────────────────────────
#  Cleanup
# ─────────────────────────────────────────────────────────────

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/

clean-all: clean  ## Remove everything including .venv
	rm -rf .venv

# ─────────────────────────────────────────────────────────────
#  Help
# ─────────────────────────────────────────────────────────────

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
