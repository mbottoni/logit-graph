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

lg-aic-efficient:  ## Run EFFICIENT AIC sweep (n=[50,100], ~1 min on 4 cores)
	LG_EXPERIMENT_MODE=EFFICIENT \
	LG_AIC_USE_CACHE=1 \
	LG_AIC_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_lg_aic_experiments.py

lg-aic-paper-fast:  ## Run PAPER_FAST AIC sweep (n=[100,500,1000], ~1.5h fresh)
	LG_EXPERIMENT_MODE=PAPER_FAST \
	LG_AIC_USE_CACHE=1 \
	LG_AIC_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_lg_aic_experiments.py

lg-aic-paper-fast-fresh:  ## PAPER_FAST sweep, force-discard cache (~1.5h)
	LG_EXPERIMENT_MODE=PAPER_FAST \
	LG_AIC_USE_CACHE=0 \
	LG_AIC_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_lg_aic_experiments.py

lg-convergence-diagnostics:  ## Run MCMC convergence diagnostics (n=750, ~30 min)
	$(UV) run python scripts/diagnostics/run_lg_convergence_diagnostics.py

lg-convergence-diagnostics-quick:  ## Quick smoke (n=200, 50k iter, ~1 min)
	LG_CONV_QUICK=1 \
		$(UV) run python scripts/diagnostics/run_lg_convergence_diagnostics.py

lg-anova-validation-robust:  ## Validate single-graph dyadic-robust Wald: Type-I calibration + power vs effect/n
	$(UV) run python scripts/experiments/run_lg_anova_validation_robust.py

lg-anova-validation-robust-quick:  ## Smoke run of the robust-Wald validation (d=0, few reps, ~30s)
	LG_AVR_QUICK=1 \
		$(UV) run python scripts/experiments/run_lg_anova_validation_robust.py

lg-gic-facebook-ego:  ## Rank LG vs ER/WS/BA on the 10 SNAP Facebook ego networks (~30-60s)
	LG_FBEGO_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_lg_facebook_ego_gic.py

lg-gic-facebook-ego-quick:  ## Smoke run on small SNAP Facebook ego networks (~15s)
	LG_FBEGO_QUICK=1 \
	LG_FBEGO_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_lg_facebook_ego_gic.py

lg-gic-facebook-ego-closedform:  ## Closed-form vs grid baselines + fair LG on the 10 Facebook ego nets (reproducible, ~2-4 min)
	$(UV) run python scripts/closedform/run_lg_facebook_ego_closedform.py

lg-gic-facebook-ego-closedform-quick:  ## Smoke run of the Facebook-ego closed-form experiment (~20s)
	LG_FBE_QUICK=1 \
		$(UV) run python scripts/closedform/run_lg_facebook_ego_closedform.py

lg-gic-facebook:  ## Rank LG vs ER/WS/BA by GIC on full MUSAE Facebook page-page graph (~2-3 min)
	LG_FB_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_lg_facebook_gic.py

lg-gic-facebook-quick:  ## Smoke run on MUSAE Facebook (~30-60s, fewer MCMC iters)
	LG_FB_QUICK=1 \
	LG_FB_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_lg_facebook_gic.py

lg-gic-arxiv:  ## Rank LG vs ER/WS/BA by GIC on full cit-HepTh citation network (~3-5 min)
	LG_ARXIV_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_lg_arxiv_gic.py

lg-gic-arxiv-quick:  ## Smoke run on cit-HepTh (~30-60s, fewer MCMC iters)
	LG_ARXIV_QUICK=1 \
	LG_ARXIV_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_lg_arxiv_gic.py

lg-gic-arxiv-closedform:  ## Closed-form vs grid baselines + fair LG on cit-HepTh BFS subgraphs (reproducible, ~2-4 min)
	$(UV) run python scripts/closedform/run_lg_arxiv_closedform.py

lg-gic-arxiv-closedform-quick:  ## Smoke run of the arXiv closed-form experiment (~30s)
	LG_ARCF_QUICK=1 \
		$(UV) run python scripts/closedform/run_lg_arxiv_closedform.py

lg-gic-human-connectomes:  ## Rank LG vs ER/WS/BA by GIC on OASIS-3 human brain nets (~3-5 min)
	LG_HCONN_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_lg_human_connectomes_gic.py

lg-gic-human-connectomes-quick:  ## Smoke run on coarse parcellation (~30s)
	LG_HCONN_QUICK=1 \
	LG_HCONN_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_lg_human_connectomes_gic.py

lg-gic-human-connectomes-closedform:  ## Closed-form vs grid baselines + fair LG on OASIS-3 human connectomes (reproducible, ~2-4 min)
	$(UV) run python scripts/closedform/run_lg_human_connectomes_closedform.py

lg-gic-human-connectomes-closedform-quick:  ## Smoke run of the human-connectome closed-form experiment (~15s)
	LG_HCF_QUICK=1 \
		$(UV) run python scripts/closedform/run_lg_human_connectomes_closedform.py

lg-gic-connectomes:  ## Rank LG vs ER/WS/BA by GIC on animal connectomes (~3-5 min)
	LG_CONN_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_lg_connectomes_gic.py

lg-gic-connectomes-quick:  ## Smoke run on small connectomes only (~30s)
	LG_CONN_QUICK=1 \
	LG_CONN_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_lg_connectomes_gic.py

lg-gic-connectomes-closedform:  ## Closed-form vs grid baselines + fair LG on animal connectomes (reproducible, ~1-2 min)
	$(UV) run python scripts/closedform/run_lg_connectomes_closedform.py

lg-gic-connectomes-closedform-quick:  ## Smoke run of the connectomes closed-form experiment (~10s)
	LG_CCF_QUICK=1 \
		$(UV) run python scripts/closedform/run_lg_connectomes_closedform.py

lg-anova-connectomes-robust:  ## Dyadic-robust Wald ANOVA on sigma across animal connectomes (reproducible)
	$(UV) run python scripts/experiments/run_lg_connectomes_anova_robust.py

lg-anova-connectomes-robust-quick:  ## Smoke run of the connectomes robust ANOVA (~30s)
	LG_CAR_QUICK=1 \
		$(UV) run python scripts/experiments/run_lg_connectomes_anova_robust.py

lg-gic-twitch:  ## Rank LG vs ER/WS/BA by GIC on Twitch country networks (6 graphs, ~2-5 min)
	LG_TWITCH_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_lg_twitch_gic.py

lg-gic-twitch-quick:  ## Smoke run on smaller Twitch country graphs (~30-60s)
	LG_TWITCH_QUICK=1 \
	LG_TWITCH_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_lg_twitch_gic.py

lg-gic-twitch-closedform:  ## Closed-form vs grid baselines + fair LG on Twitch BFS subgraphs (reproducible, ~2-4 min)
	$(UV) run python scripts/closedform/run_lg_twitch_closedform.py

lg-gic-twitch-closedform-quick:  ## Smoke run of the Twitch closed-form experiment (~15s)
	LG_TWCF_QUICK=1 \
		$(UV) run python scripts/closedform/run_lg_twitch_closedform.py

lg-gic-twitter:  ## Rank LG vs ER/WS/BA by GIC on Twitter SNAP ego nets (973 graphs, ~3-5 min)
	LG_TWITTER_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_lg_twitter_gic.py

lg-gic-twitter-quick:  ## Smoke run on small Twitter ego nets (~15-30s)
	LG_TWITTER_QUICK=1 \
	LG_TWITTER_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_lg_twitter_gic.py

lg-gic-twitter-closedform:  ## Closed-form vs grid baselines + fair LG on Twitter ego nets (reproducible, ~1-2 min)
	$(UV) run python scripts/closedform/run_lg_twitter_closedform.py

lg-gic-twitter-closedform-quick:  ## Smoke run of the Twitter closed-form experiment (~10s)
	LG_TCF_QUICK=1 \
		$(UV) run python scripts/closedform/run_lg_twitter_closedform.py

lg-gic-gplus:  ## Rank LG vs ER/WS/BA by GIC on Google+ ego nets (~3-5 min)
	LG_GPLUS_USE_CACHE=1 \
		$(UV) run python scripts/gic/run_lg_gplus_gic.py

lg-gic-gplus-quick:  ## Smoke run on small gplus subset (~30s)
	LG_GPLUS_QUICK=1 \
	LG_GPLUS_USE_CACHE=0 \
		$(UV) run python scripts/gic/run_lg_gplus_gic.py

lg-gic-gplus-closedform:  ## Closed-form vs grid baselines + fair LG on gplus (reproducible, ~30s)
	$(UV) run python scripts/closedform/run_lg_gplus_closedform.py

lg-gic-gplus-closedform-quick:  ## Smoke run of the closed-form baseline experiment (~5s)
	LG_CF_QUICK=1 \
		$(UV) run python scripts/closedform/run_lg_gplus_closedform.py

lg-roc-paper-smoke:  ## Fast probe of ROC curve shape (~30s, n_eff=200, exps=20)
	LG_EXPERIMENT_MODE=PAPER_ROC_SMOKE \
	LG_ROC_USE_CACHE=0 \
	LG_ROC_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_lg_roc_experiments.py

lg-roc-paper:  ## Reproduce paper Fig 3 + Fig 4: ANOVA ROC curves (~5 min)
	LG_EXPERIMENT_MODE=PAPER_ROC \
	LG_ROC_USE_CACHE=1 \
	LG_ROC_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_lg_roc_experiments.py

lg-sigma-convergence:  ## Reproduce paper Fig 2: σ̂ → σ as n grows (~10-15 min)
	LG_SIGMA_USE_CACHE=1 \
	LG_SIGMA_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_lg_sigma_convergence.py

lg-sigma-convergence-quick:  ## Smoke (n∈{20,50,100}, n_reps=2, ~30 sec)
	LG_SIGMA_QUICK=1 \
	LG_SIGMA_USE_CACHE=0 \
	LG_SIGMA_JOBS=$(JOBS) \
		$(UV) run python scripts/experiments/run_lg_sigma_convergence.py

lg-anova-twitch-robust:  ## Twitch sigma ANOVA with dyadic-cluster-robust SE (Wald, full graphs, ~3-5 min)
	$(UV) run python scripts/experiments/run_lg_twitch_anova_robust.py

lg-anova-twitch-robust-quick:  ## Smoke of the robust Twitch ANOVA (2 small regions, ~5s)
	LG_TWA_QUICK=1 \
		$(UV) run python scripts/experiments/run_lg_twitch_anova_robust.py

tlg-anova-twitch-robust:  ## TLG Twitch ANOVA on BOTH sigma and alpha (2-param dyadic-robust SE; omnibus + Bonferroni)
	$(UV) run python scripts/experiments/run_tlg_twitch_anova_robust.py

tlg-anova-twitch-robust-quick:  ## Smoke of the TLG Twitch ANOVA (2 small regions)
	LG_TTA_QUICK=1 \
		$(UV) run python scripts/experiments/run_tlg_twitch_anova_robust.py

tlg-anova-twitch-robust-validate:  ## Validate the 2-param dyadic-robust SE vs Monte-Carlo
	LG_TTA_VALIDATE=1 \
		$(UV) run python scripts/experiments/run_tlg_twitch_anova_robust.py

tlg-anova-connectomes-robust:  ## TLG ANOVA on sigma AND alpha across the 18 animal connectomes (2-param dyadic-robust SE; omnibus + Bonferroni, 153 pairs)
	$(UV) run python scripts/experiments/run_tlg_connectomes_anova_robust.py

tlg-anova-connectomes-robust-quick:  ## Smoke of the TLG connectomes ANOVA (4 graphs)
	LG_TCA_QUICK=1 \
		$(UV) run python scripts/experiments/run_tlg_connectomes_anova_robust.py

tlg-zero-test:  ## TLG single-graph significance test (H0: sigma=0, alpha=0) on twitch + connectomes (Wald, dyadic-robust SE)
	$(UV) run python scripts/experiments/run_tlg_zero_test.py

tlg-zero-test-quick:  ## Smoke of the TLG zero test (2 twitch regions + 4 connectomes)
	LG_ZT_QUICK=1 \
		$(UV) run python scripts/experiments/run_tlg_zero_test.py

tlg-zero-test-validate:  ## Monte-Carlo calibration / power / ROC of the alpha=0 test (consistent temporal MLE)
	LG_ZT_VALIDATE=1 \
		$(UV) run python scripts/experiments/run_tlg_zero_test.py

kpm-sensitivity-citation:  ## KPM (moments/probes) sensitivity on the cit-HepTh citation network vs reference + exact
	$(UV) run python scripts/experiments/run_kpm_sensitivity_citation.py

kpm-sensitivity-citation-quick:  ## Smoke of the KPM sensitivity study (small grids)
	LG_KPM_QUICK=1 \
		$(UV) run python scripts/experiments/run_kpm_sensitivity_citation.py

rhesus-case-study-figure:  ## Rhesus cerebral-cortex case-study figure (thesis Fig. 3.11): 7 models + Original, ranked by KL
	$(UV) run python scripts/experiments/run_rhesus_case_study_figure.py

rhesus-convergence-figure:  ## LG spectral-fitting convergence (Edge/Spectrum/GIC vs iteration) for a d>=1 connectome
	$(UV) run python scripts/experiments/run_rhesus_convergence_figure.py

rhesus-spectrum-figure:  ## Normalized-Laplacian spectral density (hist+KDE) of the connectome vs each model best fit
	$(UV) run python scripts/experiments/run_rhesus_spectrum_figure.py

# ─────────────────────────────────────────────────────────────
#  Temporal Logit-Graph experiments (tlg- prefix)
# ─────────────────────────────────────────────────────────────

tlg-recovery:  ## TLG param recovery vs n for d=0,1,2 (sigma,alpha + 95% bands, reproducible)
	$(UV) run python scripts/experiments/run_tlg_recovery.py

tlg-recovery-quick:  ## Smoke run of the TLG recovery experiment (d={0,1}, small n, ~30s)
	LG_TLG_QUICK=1 \
		$(UV) run python scripts/experiments/run_tlg_recovery.py

tlg-param-behavior:  ## TLG (d,sigma,alpha) behavior sweep for d={0,1,2}, n={50,200,500}: centrality, clustering, scale-free / power-law metrics + per-d heatmaps (reproducible, cached)
	$(UV) run python scripts/experiments/run_tlg_param_behavior.py

tlg-param-behavior-quick:  ## Smoke run of the TLG param-behavior sweep (d={0,1}, n=50, tiny grid, ~30s)
	LG_TLGB_QUICK=1 \
		$(UV) run python scripts/experiments/run_tlg_param_behavior.py

tlg-identifiability:  ## Unified latent-TLG identifiability: recover sigma,alpha,gamma_c,gamma_f,lambda vs n (MLE, add+remove)
	$(UV) run python scripts/experiments/run_tlg_latent_identifiability.py

tlg-identifiability-quick:  ## Smoke run of the latent-TLG identifiability experiment (1 scenario, small n, ~5s)
	LG_TLI_QUICK=1 \
		$(UV) run python scripts/experiments/run_tlg_latent_identifiability.py

tlg-dimred-twitch:  ## Dim-red (PCA/t-SNE/UMAP per network) of per-family graph ensembles vs the real Twitch graphs in structural-feature space
	LG_DR_DATASET=twitch $(UV) run python scripts/dim_red/run_tlg_dimred.py

tlg-dimred-connectome:  ## Same dim-red experiment on the animal connectomes dataset
	LG_DR_DATASET=connectome $(UV) run python scripts/dim_red/run_tlg_dimred.py

tlg-roc:  ## TLG ROC: group-difference tests on sigma AND alpha (effect-size + sample-size; single-graph Wald SE)
	$(UV) run python scripts/experiments/run_tlg_roc_experiments.py

tlg-roc-quick:  ## Smoke run of the TLG ROC experiment (d={0}, small grids, ~30s)
	LG_TLGROC_QUICK=1 \
		$(UV) run python scripts/experiments/run_tlg_roc_experiments.py

tlg-aic-d:  ## TLG AIC d-recovery: does AIC pick the true degree-feature depth d? (accuracy vs n + confusion)
	$(UV) run python scripts/experiments/run_tlg_aic_d_recovery.py

tlg-aic-d-quick:  ## Smoke run of the TLG AIC d-recovery experiment (~30s)
	LG_TLGAIC_QUICK=1 \
		$(UV) run python scripts/experiments/run_tlg_aic_d_recovery.py

tlg-twitch-gic:  ## Fit Twitch nets with the TLG (d by AIC, sigma/alpha by logistic regression, GIC by edge-gated growth) vs closed-form ER/BA/WS/KR/GRG/SBM
	$(UV) run python scripts/closedform/run_tlg_twitch_gic.py

tlg-twitch-gic-all:  ## Same, all six Twitch country networks (PTBR..DE)
	LG_TLGT_ALL=1 \
		$(UV) run python scripts/closedform/run_tlg_twitch_gic.py

tlg-twitch-gic-quick:  ## Smoke run of the TLG Twitch GIC experiment (PTBR, tiny)
	LG_TLGT_QUICK=1 \
		$(UV) run python scripts/closedform/run_tlg_twitch_gic.py

tlg-twitch-latent-gic:  ## Latent-TLG GIC sweep over ALL Twitch networks (cached/parallel) vs ER/BA/WS/KR/GRG/SBM
	$(UV) run python scripts/closedform/run_tlg_twitch_latent_gic.py

tlg-twitter-latent-gic:  ## Latent-TLG GIC sweep over all Twitter ego networks (50<n<1000; cached/parallel)
	$(UV) run python scripts/closedform/run_tlg_twitter_latent_gic.py

tlg-facebook-latent-gic:  ## Latent-TLG GIC sweep over all Facebook ego networks (cached/parallel)
	$(UV) run python scripts/closedform/run_tlg_facebook_latent_gic.py

tlg-arxiv-latent-gic:  ## Latent-TLG GIC on the arXiv HEP-Th citation network (big BFS subgraph)
	$(UV) run python scripts/closedform/run_tlg_arxiv_latent_gic.py

tlg-gplus-latent-gic:  ## Latent-TLG GIC sweep over all Google+ ego networks (50<n<1000; cached/parallel)
	$(UV) run python scripts/closedform/run_tlg_gplus_latent_gic.py

tlg-connectome-latent-gic:  ## Latent-TLG GIC sweep over all animal connectomes (cached/parallel)
	$(UV) run python scripts/closedform/run_tlg_connectome_latent_gic.py

tlg-human-latent-gic:  ## Latent-TLG GIC sweep over all OASIS3 human connectomes (cached/parallel)
	$(UV) run python scripts/closedform/run_tlg_human_latent_gic.py

tlg-all-latent-gic:  ## Latent-TLG GIC sweep over ALL datasets in one global pool + overall cross-dataset KL ranking
	$(UV) run python scripts/closedform/run_tlg_all_latent_gic.py

tlg-all-latent-gic-quick:  ## Smoke run of the latent-TLG GIC all-datasets sweep (tiny)
	LG_TLM_QUICK=1 \
		$(UV) run python scripts/closedform/run_tlg_all_latent_gic.py

tlg-convergence-diagnostics:  ## TLG (add+remove) convergence: chains from different initial densities mix to the same stationary distribution
	$(UV) run python scripts/diagnostics/run_tlg_convergence_diagnostics.py

tlg-convergence-diagnostics-quick:  ## Smoke run of the TLG convergence diagnostics (~30s)
	LG_TLGC_QUICK=1 \
		$(UV) run python scripts/diagnostics/run_tlg_convergence_diagnostics.py

tlg-esd-stop-eval:  ## Evaluate ESD-KL convergence stopping (until_convergence): stop-step + bias vs n, tol sensitivity
	$(UV) run python scripts/diagnostics/run_tlg_esd_stop_eval.py

tlg-esd-stop-eval-quick:  ## Smoke run of the ESD-stop evaluation (~30s)
	LG_ESS_QUICK=1 \
		$(UV) run python scripts/diagnostics/run_tlg_esd_stop_eval.py

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
