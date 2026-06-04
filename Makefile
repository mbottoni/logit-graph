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
#  Experiments
# ─────────────────────────────────────────────────────────────

# Parallel-job count for AIC sweeps; override with `make aic-paper JOBS=8`
JOBS ?= 4

aic-efficient:  ## Run EFFICIENT AIC sweep (n=[50,100], ~1 min on 4 cores)
	LG_EXPERIMENT_MODE=EFFICIENT \
	LG_AIC_USE_CACHE=1 \
	LG_AIC_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_aic_experiments.py

aic-paper-fast:  ## Run PAPER_FAST AIC sweep (n=[100,500,1000], ~1.5h fresh)
	LG_EXPERIMENT_MODE=PAPER_FAST \
	LG_AIC_USE_CACHE=1 \
	LG_AIC_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_aic_experiments.py

aic-paper-fast-fresh:  ## PAPER_FAST sweep, force-discard cache (~1.5h)
	LG_EXPERIMENT_MODE=PAPER_FAST \
	LG_AIC_USE_CACHE=0 \
	LG_AIC_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_aic_experiments.py

convergence-diagnostics:  ## Run MCMC convergence diagnostics (n=750, ~30 min)
	$(UV) run python scripts/diagnostics/run_convergence_diagnostics.py

convergence-diagnostics-quick:  ## Quick smoke (n=200, 50k iter, ~1 min)
	LG_CONV_QUICK=1 \
		$(UV) run python scripts/diagnostics/run_convergence_diagnostics.py

gic-facebook-ego:  ## Rank LG vs ER/WS/BA on the 10 SNAP Facebook ego networks (~30-60s)
	LG_FBEGO_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_facebook_ego_gic.py

gic-facebook-ego-quick:  ## Smoke run on small SNAP Facebook ego networks (~15s)
	LG_FBEGO_QUICK=1 \
	LG_FBEGO_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_facebook_ego_gic.py

gic-facebook-ego-closedform:  ## Closed-form vs grid baselines + fair LG on the 10 Facebook ego nets (reproducible, ~2-4 min)
	$(UV) run python scripts/closedform/run_facebook_ego_closedform.py

gic-facebook-ego-closedform-quick:  ## Smoke run of the Facebook-ego closed-form experiment (~20s)
	LG_FBE_QUICK=1 \
		$(UV) run python scripts/closedform/run_facebook_ego_closedform.py

gic-facebook:  ## Rank LG vs ER/WS/BA by GIC on full MUSAE Facebook page-page graph (~2-3 min)
	LG_FB_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_facebook_gic.py

gic-facebook-quick:  ## Smoke run on MUSAE Facebook (~30-60s, fewer MCMC iters)
	LG_FB_QUICK=1 \
	LG_FB_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_facebook_gic.py

gic-arxiv:  ## Rank LG vs ER/WS/BA by GIC on full cit-HepTh citation network (~3-5 min)
	LG_ARXIV_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_arxiv_gic.py

gic-arxiv-quick:  ## Smoke run on cit-HepTh (~30-60s, fewer MCMC iters)
	LG_ARXIV_QUICK=1 \
	LG_ARXIV_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_arxiv_gic.py

gic-arxiv-closedform:  ## Closed-form vs grid baselines + fair LG on cit-HepTh BFS subgraphs (reproducible, ~2-4 min)
	$(UV) run python scripts/closedform/run_arxiv_closedform.py

gic-arxiv-closedform-quick:  ## Smoke run of the arXiv closed-form experiment (~30s)
	LG_ARCF_QUICK=1 \
		$(UV) run python scripts/closedform/run_arxiv_closedform.py

gic-human-connectomes:  ## Rank LG vs ER/WS/BA by GIC on OASIS-3 human brain nets (~3-5 min)
	LG_HCONN_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_human_connectomes_gic.py

gic-human-connectomes-quick:  ## Smoke run on coarse parcellation (~30s)
	LG_HCONN_QUICK=1 \
	LG_HCONN_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_human_connectomes_gic.py

gic-human-connectomes-closedform:  ## Closed-form vs grid baselines + fair LG on OASIS-3 human connectomes (reproducible, ~2-4 min)
	$(UV) run python scripts/closedform/run_human_connectomes_closedform.py

gic-human-connectomes-closedform-quick:  ## Smoke run of the human-connectome closed-form experiment (~15s)
	LG_HCF_QUICK=1 \
		$(UV) run python scripts/closedform/run_human_connectomes_closedform.py

gic-connectomes:  ## Rank LG vs ER/WS/BA by GIC on animal connectomes (~3-5 min)
	LG_CONN_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_connectomes_gic.py

gic-connectomes-quick:  ## Smoke run on small connectomes only (~30s)
	LG_CONN_QUICK=1 \
	LG_CONN_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_connectomes_gic.py

gic-connectomes-closedform:  ## Closed-form vs grid baselines + fair LG on animal connectomes (reproducible, ~1-2 min)
	$(UV) run python scripts/closedform/run_connectomes_closedform.py

gic-connectomes-closedform-quick:  ## Smoke run of the connectomes closed-form experiment (~10s)
	LG_CCF_QUICK=1 \
		$(UV) run python scripts/closedform/run_connectomes_closedform.py

gic-twitch:  ## Rank LG vs ER/WS/BA by GIC on Twitch country networks (6 graphs, ~2-5 min)
	LG_TWITCH_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_twitch_gic.py

gic-twitch-quick:  ## Smoke run on smaller Twitch country graphs (~30-60s)
	LG_TWITCH_QUICK=1 \
	LG_TWITCH_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_twitch_gic.py

gic-twitch-closedform:  ## Closed-form vs grid baselines + fair LG on Twitch BFS subgraphs (reproducible, ~2-4 min)
	$(UV) run python scripts/closedform/run_twitch_closedform.py

gic-twitch-closedform-quick:  ## Smoke run of the Twitch closed-form experiment (~15s)
	LG_TWCF_QUICK=1 \
		$(UV) run python scripts/closedform/run_twitch_closedform.py

gic-twitter:  ## Rank LG vs ER/WS/BA by GIC on Twitter SNAP ego nets (973 graphs, ~3-5 min)
	LG_TWITTER_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_twitter_gic.py

gic-twitter-quick:  ## Smoke run on small Twitter ego nets (~15-30s)
	LG_TWITTER_QUICK=1 \
	LG_TWITTER_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_twitter_gic.py

gic-twitter-closedform:  ## Closed-form vs grid baselines + fair LG on Twitter ego nets (reproducible, ~1-2 min)
	$(UV) run python scripts/closedform/run_twitter_closedform.py

gic-twitter-closedform-quick:  ## Smoke run of the Twitter closed-form experiment (~10s)
	LG_TCF_QUICK=1 \
		$(UV) run python scripts/closedform/run_twitter_closedform.py

gic-gplus:  ## Rank LG vs ER/WS/BA by GIC on Google+ ego nets (~3-5 min)
	LG_GPLUS_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_gplus_gic.py

gic-gplus-quick:  ## Smoke run on small gplus subset (~30s)
	LG_GPLUS_QUICK=1 \
	LG_GPLUS_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_gplus_gic.py

gic-gplus-closedform:  ## Closed-form vs grid baselines + fair LG on gplus (reproducible, ~30s)
	$(UV) run python scripts/closedform/run_gplus_closedform.py

gic-gplus-closedform-quick:  ## Smoke run of the closed-form baseline experiment (~5s)
	LG_CF_QUICK=1 \
		$(UV) run python scripts/closedform/run_gplus_closedform.py

roc-paper-smoke:  ## Fast probe of ROC curve shape (~30s, n_eff=200, exps=20)
	LG_EXPERIMENT_MODE=PAPER_ROC_SMOKE \
	LG_ROC_USE_CACHE=0 \
	LG_ROC_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_roc_experiments.py

roc-paper:  ## Reproduce paper Fig 3 + Fig 4: ANOVA ROC curves (~5 min)
	LG_EXPERIMENT_MODE=PAPER_ROC \
	LG_ROC_USE_CACHE=1 \
	LG_ROC_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_roc_experiments.py

sigma-convergence:  ## Reproduce paper Fig 2: σ̂ → σ as n grows (~10-15 min)
	LG_SIGMA_USE_CACHE=1 \
	LG_SIGMA_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_sigma_convergence.py

sigma-convergence-quick:  ## Smoke (n∈{20,50,100}, n_reps=2, ~30 sec)
	LG_SIGMA_QUICK=1 \
	LG_SIGMA_USE_CACHE=0 \
	LG_SIGMA_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_sigma_convergence.py

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
