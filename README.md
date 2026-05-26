# Logit Graph

A probabilistic logit-based graph model and utilities for fitting, simulating, and comparing random graph models to real-world networks. The package provides a scikit-learn-like API to fit the Logit Graph (LG) model and benchmark it against classic random graph models (ER, WS, BA, optionally GRG), using spectral distances and a Graph Information Criterion (GIC).

- **PyPI**: `logit-graph`
- **Python**: >=3.9
- **License**: MIT

## Table of Contents

- [Installation](#installation)
- [Examples](#examples)
- [Quickstart](#quickstart)
- [Public API](#public-api)
- [Model Overview](#model-overview)
- [Project Structure](#project-structure)
- [Development](#development)
- [Citation](#citation)

## Installation

Install from PyPI:

```bash
pip install logit-graph
```

For local development within this repo (recommended — uses `uv`):

```bash
make install        # creates .venv and installs with viz/notebook/progress extras
make install-dev    # also installs pytest, ruff, mypy
make install-torch  # also installs optional PyTorch backend
```

Or manually:

```bash
pip install -e ".[viz,notebook,progress]"
```

For full research environment (notebooks, plotting), use `requirements.txt` or `environment.yml`.

## Examples

Two self-contained notebooks live under [`examples/`](examples/). They install [`logit-graph` from PyPI](https://pypi.org/project/logit-graph/) (>=0.1.3) and work without a repo checkout (except when you already have the package installed locally in editable mode).

| Notebook | What it shows |
|----------|---------------|
| [`pypi_estimate_d_sigma.ipynb`](examples/pypi_estimate_d_sigma.ipynb) | Simulate an LG graph with known `n=200`, `d`, and `σ`; recover `d̂` via AIC and `σ̂` via the Layer-2 offset logit |
| [`pypi_fit_real_network.ipynb`](examples/pypi_fit_real_network.ipynb) | Fit a **real** Facebook ego network (SNAP `686.edges`) where **LG wins** (lowest GIC vs ER / WS / BA) |

Run from the repo root (after `make install`):

```bash
jupyter notebook examples/pypi_estimate_d_sigma.ipynb
jupyter notebook examples/pypi_fit_real_network.ipynb
```

## Quickstart

The recommended workflow uses the **paper-consistent** Layer-2 offset logit (`feature_mode="incremental"`, `β=1`): pick `d̂` via AIC, estimate `σ̂`, then compare fitted models with spectral GIC.

### 1. Simulate a graph and recover `d̂`, `σ̂`

```python
import numpy as np
from logit_graph import simulate_graph, select_d_ensemble, estimate_sigma_from_graph

N, D_TRUE, SIGMA_TRUE = 200, 1, -4.0

adj, meta = simulate_graph(
    N, D_TRUE, sigma=SIGMA_TRUE, n_iter=30_000,
    feature_mode="incremental", target_density=0.10, seed=42, return_meta=True,
)

d_hat, aic_stats = select_d_ensemble(
    graphs=[adj],
    d_candidates=[0, 1, 2, 3],
    feature_mode="incremental",
)
sigma_hat = estimate_sigma_from_graph(adj, d_hat, feature_mode="incremental")

print(f"true  d={D_TRUE}, σ={SIGMA_TRUE:.3f}")
print(f"est   d̂={d_hat}, σ̂={sigma_hat:.3f}")
```

See [`examples/pypi_estimate_d_sigma.ipynb`](examples/pypi_estimate_d_sigma.ipynb) for the full notebook.

### 2. Fit a real network and compare LG vs ER / WS / BA

```python
import networkx as nx
from logit_graph import GraphModelComparator, estimate_sigma_from_graph, select_d_ensemble

G = nx.read_edgelist("686.edges", nodetype=int)   # SNAP Facebook ego net
G = nx.convert_node_labels_to_integers(nx.Graph(G))
adj = nx.to_numpy_array(G)

d_hat, _ = select_d_ensemble([adj], [0, 1, 2, 3], "incremental")
sigma_hat = estimate_sigma_from_graph(adj, d_hat, "incremental")

comparator = GraphModelComparator(
    d_list=[d_hat],                    # LG only at the AIC-selected d̂
    lg_params={
        "max_iterations": 5000,
        "patience": 500,
        "edge_delta": None,
        "min_gic_threshold": 5,
        "er_p": 0.05,
        "check_interval": 50,
    },
    other_model_n_runs=2,
    dist_type="KL",
    verbose=False,
    other_models=["ER", "WS", "BA"],
    other_model_grid_points=5,
    random_state=0,                    # reproducible GIC (requires logit-graph >= 0.1.3)
).compare(original_graph=G, graph_filepath="facebook_686")

print(comparator.summary_df.sort_values("gic_value"))
print(f"d̂={d_hat}, σ̂={sigma_hat:+.4f}")
```

On ego network **686**, LG typically achieves the **lowest GIC**. See [`examples/pypi_fit_real_network.ipynb`](examples/pypi_fit_real_network.ipynb) (downloads the graph from SNAP if needed, or uses `examples/data/686.edges` when present).

### 3. Direct spectral fit with `LogitGraphFitter`

For a fixed `d`, `LogitGraphFitter` runs a GIC-guided edge-swap search to produce a fitted graph whose spectrum matches the original:

```python
import networkx as nx
from logit_graph import LogitGraphFitter

G = nx.karate_club_graph()

fitter = LogitGraphFitter(
    d=1, n_iteration=5000, patience=500, dist_type="KL", verbose=True,
).fit(G)

print(f"GIC={fitter.metadata['gic_value']:.4f}, σ={fitter.metadata['sigma']:.4f}")
```

This is the MCMC-style spectral matcher. For paper-consistent model selection, prefer `select_d_ensemble` + `GraphModelComparator` as in step 2.

## Public API

All symbols below are importable from `logit_graph`. The **paper-consistent** path for estimation is `simulate_graph` → `select_d_ensemble` → `estimate_sigma_from_graph` → `GraphModelComparator`.

### `simulate_graph`

Generate a random graph at fixed `(n, d, σ)`.

```python
from logit_graph import simulate_graph

adj = simulate_graph(
    n=200, d=1, sigma=-4.0, n_iter=30_000,
    feature_mode="incremental", target_density=0.10, seed=42,
)
# adj, meta = simulate_graph(..., return_meta=True)  → also returns σ, β, density, …
```

| Parameter | Description |
|-----------|-------------|
| `n`, `d`, `sigma` | Graph size, feature depth, logit intercept |
| `n_iter` | Gibbs iterations (`d≥1`) or ignored (`d=0`, direct ER) |
| `feature_mode` | `"incremental"` (default paper mode), `"bounded"`, or `"full"` |
| `target_density` | Used when calibrating `β` if `sigma` is omitted |
| `return_meta` | If `True`, return `(adj, meta)` with fitted `σ`, `β`, density |

### `select_d_ensemble`

Pick `d̂` by AIC over candidate depths using the Layer-2 offset logit.

```python
from logit_graph import select_d_ensemble

d_hat, aic_stats = select_d_ensemble(
    graphs=[adj],                      # list of adjacency matrices
    d_candidates=[0, 1, 2, 3],
    feature_mode="incremental",
    extra_penalty_per_d=0.0,           # add e.g. 3.0 to penalise larger d
)
# aic_stats[d] → {"aic", "ll", "sigma_hat", "n_obs", …}
```

### `estimate_sigma_from_graph`

Offset-logit estimate of `σ̂` at a fixed `d` (same estimator used inside the AIC table).

```python
from logit_graph import estimate_sigma_from_graph

sigma_hat = estimate_sigma_from_graph(adj, d=1, feature_mode="incremental")
```

### `GraphModelComparator`

Compare **LG** (at one or more `d` values) against baseline models using spectral GIC (lower = better).

```python
from logit_graph import GraphModelComparator

comparator = GraphModelComparator(
    d_list=[d_hat],                    # usually the AIC-selected d̂ only
    lg_params={                        # passed to LogitGraphFitter internally
        "max_iterations": 5000,
        "patience": 500,
        "edge_delta": None,
        "min_gic_threshold": 5,
        "er_p": 0.05,
        "check_interval": 50,
    },
    other_model_n_runs=2,
    dist_type="KL",                    # "KL", "L1", or "L2"
    verbose=False,
    other_models=["ER", "WS", "BA"],   # optionally include "GRG"
    other_model_grid_points=5,
    random_state=0,                    # seed LG Gibbs + baseline sampling (>= 0.1.3)
).compare(original_graph=G, graph_filepath="my_graph")

comparator.summary_df                      # per-model GIC and attributes
comparator.fitted_graphs_data["LG"]        # {"graph", "metadata", "attributes"}
```

When `d_list` has multiple entries, the comparator searches over `d` internally and keeps the best LG fit. For paper consistency, pass `d_list=[d_hat]` where `d_hat` comes from `select_d_ensemble`.

### `LogitGraphFitter`

Sklearn-style fitter: given a fixed `d`, estimate `σ` via offset logit and search for a graph minimising spectral GIC.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d` | `0` | Neighborhood depth for degree-sum features |
| `n_iteration` | `10000` | Max edge-swap / Gibbs iterations |
| `warm_up` | `500` | Burn-in before GIC tracking |
| `patience` | `2000` | Early-stop patience |
| `dist_type` | `"KL"` | Spectral distance: `"KL"`, `"L1"`, `"L2"` |
| `min_gic_threshold` | `5` | Min GIC drop to reset patience |
| `er_p` | `0.05` | ER probability for warm-start graph |
| `verbose` | `True` | Print progress |

After `fit(G)`: `fitter.fitted_graph`, `fitter.metadata` (`sigma`, `gic_value`, `best_iteration`, …).

### Other exports

| Symbol | Role |
|--------|------|
| `LogitGraphSimulation` | Lower-level multi-run LG simulation (used inside the comparator) |
| `LogitRegEstimator` | Layer-2 offset logit on pair features; returns AIC stats |
| `calculate_graph_attributes` | Density, clustering, diameter, assortativity, … |
| `recommended_iterations` | Suggested Gibbs length as a function of `n` |
| `build_pair_dataset`, `pair_feature`, `pair_feature_layer2` | Feature construction for custom pipelines |
| `GraphModel` | Core Gibbs / edge-swap engine |
| `AICSweepConfig`, `SigmaSweepConfig`, `PRESETS` | Experiment presets under `logit_graph.experiments` |

## Model Overview

The Logit Graph model defines edge probabilities using a logistic function of local degree-sum features:

```
P(edge i–j) = sigmoid(σ · (deg_d(i) + deg_d(j)))
```

where `deg_d(v)` is the sum of degrees in the `d`-hop neighborhood of vertex `v`, and `σ` is the fitted scale parameter.

**Fitting** uses an iterative edge-swap procedure guided by the spectral density of the normalized Laplacian. At each step, edges are proposed and accepted/rejected based on a GIC-like criterion comparing spectral histograms.

**GIC** (Graph Information Criterion) is defined as:

```
GIC = 2 · spectral_distance(original, fitted) + 2 · |θ|
```

where `|θ|` is the number of free parameters (1 for LG). This penalizes model complexity analogously to AIC.

**Supported spectral distances:**
- `KL` — KL divergence between normalized Laplacian spectral density histograms (default)
- `L1` — Manhattan distance
- `L2` — Euclidean distance

**Supported baseline models for comparison:**

| Model | Description |
|-------|-------------|
| `ER` | Erdős–Rényi (edge probability `p`) |
| `WS` | Watts–Strogatz small-world |
| `BA` | Barabási–Albert preferential attachment |
| `GRG` | Geometric Random Graph (random geometric) |

## Project Structure

```
logit-graph/
├── src/logit_graph/          # Package source
│   ├── __init__.py           # Public exports (see Public API section)
│   ├── simulation.py         # High-level fitter and comparator classes
│   ├── graph.py              # GraphModel: MCMC edge-swap engine (core LG generation)
│   ├── logit_estimator.py    # Parameter estimation via logistic regression (sklearn / statsmodels / torch)
│   ├── gic.py                # GraphInformationCriterion: spectral density + GIC formula
│   ├── model_selection.py    # Model selection utilities and grid search helpers
│   ├── param_estimator.py    # Low-level sigma/alpha/beta parameter estimators
│   ├── degrees_counts.py     # degree_vertex / get_sum_degrees helpers
│   └── utils.py              # Miscellaneous utilities
│
├── tests/                    # Pytest test suite
│   ├── conftest.py
│   ├── test_graph_model.py   # GraphModel unit tests
│   ├── test_logit_estimator.py
│   ├── test_gic.py
│   ├── test_degrees_counts.py
│   ├── test_graph_helpers.py
│   ├── test_bugfixes.py
│   └── test_param_and_model_selection_smoke.py
│
├── examples/                 # PyPI-friendly tutorials (simulated + real data)
│   ├── pypi_estimate_d_sigma.ipynb
│   ├── pypi_fit_real_network.ipynb
│   └── data/                 # Cached SNAP ego net (686.edges)
│
├── notebooks/                # Reproducible analysis notebooks
│   ├── base/                 # Core model validation and synthetic experiments
│   ├── anova/                # ANOVA-based graph comparison experiments
│   ├── connectomes_datasets/ # Brain connectome analysis
│   ├── human_connectomes/    # Human connectome experiments
│   ├── misc_datasets/        # Social networks: Facebook, Twitter, Reddit, Twitch, G+
│   ├── more_baselines/       # Additional baseline model comparisons
│   ├── dim_red/              # Dimensionality reduction experiments
│   ├── kde/                  # KDE-based density estimation experiments
│   ├── scale_free_tests/     # Scale-free network tests
│   ├── citation/             # Citation network experiments
│   └── playground/           # Exploratory / scratch notebooks
│
├── data/                     # Network datasets (not required for pip install)
│   ├── brain_graph/          # Brain connectivity data
│   ├── connectomes/          # Connectome datasets
│   ├── citation_networks/    # arXiv HEP-Th citation network
│   ├── facebook_large/       # Facebook SNAP dataset
│   ├── git_web_ml/           # GitHub ML social graph
│   ├── reddit_connected/     # Reddit connected-community graphs
│   ├── reddit_threads/       # Reddit thread graphs
│   ├── twitch/, twitch_gamers/ # Twitch social network datasets
│   ├── soc-flickr/, soc-orkut/, soc-youtube/, soc-academia/, soc-hamsterster/
│   └── misc/                 # Miscellaneous small graphs
│
├── images/                   # Generated figures used in the paper
├── runs/                     # Saved comparator outputs (.pkl)
├── scripts/                  # Helper scripts
├── pyproject.toml            # Package metadata and dependencies
├── requirements.txt          # Full research environment dependencies
├── environment.yml           # Conda environment spec
├── Makefile                  # Dev workflow (see below)
└── uv.lock                   # Locked dependency versions
```

### Key source files

| File | Responsibility |
|------|----------------|
| `simulation.py` | `LogitGraphFitter`, `LogitGraphSimulation`, `GraphModelComparator` — the main user-facing classes |
| `graph.py` | `GraphModel` — MCMC-style edge-swap engine; initialized from an ER graph, iteratively proposes edge changes driven by the logit probability |
| `logit_estimator.py` | Estimates σ (and optionally α, β) via logistic regression on degree-sum features; supports sklearn, statsmodels, and an optional PyTorch backend |
| `gic.py` | `GraphInformationCriterion` — computes normalized Laplacian spectral density and evaluates GIC for any supported model |
| `model_selection.py` | Grid search over `d` and baseline model parameters; aggregates results into a summary DataFrame |
| `param_estimator.py` | Low-level MLE routines for model parameters |
| `degrees_counts.py` | Fast `degree_vertex` and `get_sum_degrees` helpers used throughout |

## Development

All common tasks are available via `make`. Run `make` (or `make help`) to list them:

```
  .venv             Create virtual environment
  install           Install package in editable mode with all extras
  install-dev       Install dev / test dependencies
  install-torch     Install with optional PyTorch support
  lock              Regenerate uv.lock
  sync              Sync environment from lockfile
  test              Run test suite
  test-cov          Run tests with coverage report
  lint              Lint source code with ruff
  lint-fix          Auto-fix lint issues
  format            Format code with ruff
  typecheck         Run mypy type checking
  check             Run all checks (lint + types + tests)
  build             Build sdist and wheel
  publish           Upload to PyPI
  clean             Remove caches and build artifacts
  clean-all         Remove everything including .venv
```

### Running tests

```bash
make test
# or with coverage:
make test-cov
```

### Adding a new notebook

Place it in the appropriate subdirectory under `notebooks/`. The `make nb-citation` and `make nb-playground` targets show how to execute notebooks non-interactively via `nbconvert`.

## Troubleshooting

- Plotting and notebooks require optional dependencies in `requirements.txt`.
- If `igraph` or `pycairo` fail to install, install their system packages or wheels first.
- For very large graphs, lower `max_iterations`/`patience` or compare fewer models at once.
- The PyTorch backend in `logit_estimator.py` is optional. If `torch` is not installed, the sklearn/statsmodels backend is used automatically.

## Citation

If you use this package in academic work, please cite:

```text
Ottoni, M. (2025). Logit Graph: probabilistic logit-based graph modeling and selection.
GitHub repository. https://github.com/mbottoni/logit-graph
```

A formal citation entry will be added upon publication.
