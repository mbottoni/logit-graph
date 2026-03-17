# Logit Graph

A probabilistic logit-based graph model and utilities for fitting, simulating, and comparing random graph models to real-world networks. The package provides a scikit-learn-like API to fit the Logit Graph (LG) model and benchmark it against classic random graph models (ER, WS, BA, optionally GRG), using spectral distances and a Graph Information Criterion (GIC).

- **PyPI**: `logit-graph`
- **Python**: >=3.9
- **License**: MIT

## Table of Contents

- [Installation](#installation)
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

## Quickstart

Interactive tutorial: https://colab.research.google.com/drive/1-WlU12bxN2-84fLI7IpEXB6jkifcMuaY?usp=sharing

### Fit a Logit Graph to a network

```python
import networkx as nx
from logit_graph import LogitGraphFitter

G = nx.karate_club_graph()

fitter = LogitGraphFitter(d=2, n_iteration=2000, patience=500, dist_type='KL', verbose=True)
fitter = fitter.fit(G)

fitted_graph = fitter.fitted_graph
print(f"GIC: {fitter.metadata['gic_value']:.4f}, sigma: {fitter.metadata['sigma']:.4f}")
```

### Compare models (LG vs ER/WS/BA)

```python
import networkx as nx
from logit_graph import GraphModelComparator

G = nx.karate_club_graph()

comparator = GraphModelComparator(
    d_list=[0, 1, 2, 3],
    lg_params={
        'max_iterations': 2000,
        'patience': 500,
        'edge_delta': None,
        'min_gic_threshold': 5,
        'er_p': 0.05,
    },
    other_model_n_runs=2,
    dist_type='KL',
    verbose=True,
    other_models=["ER", "WS", "BA"],    # optionally include "GRG"
    other_model_grid_points=5
)

comparator = comparator.compare(original_graph=G, graph_filepath="karate_club")
print(comparator.summary_df)

lg_graph = comparator.fitted_graphs_data['LG']['graph']
lg_meta  = comparator.fitted_graphs_data['LG']['metadata']
```

## Public API

The package exposes four top-level names (all importable directly from `logit_graph`):

### `LogitGraphFitter`

Fits a single Logit Graph model to a `networkx.Graph`.

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `d` | `int` | — | Neighborhood depth for degree-sum features |
| `n_iteration` | `int` | — | Maximum number of MCMC-like edge-swap iterations |
| `warm_up` | `int` | — | Burn-in iterations before GIC tracking starts |
| `patience` | `int` | — | Early-stopping patience (iterations without GIC improvement) |
| `dist_type` | `str` | `'KL'` | Spectral distance type: `'KL'`, `'L1'`, or `'L2'` |
| `edge_delta` | `float\|None` | `None` | If set, stops when edge count is within this fraction of original |
| `min_gic_threshold` | `float` | `5` | Minimum GIC improvement to reset patience counter |
| `er_p` | `float` | `0.05` | ER probability for the initial warm-up graph |
| `verbose` | `bool` | `False` | Print iteration progress |

**Methods:**
- `fit(original_graph: nx.Graph) -> self`

**Attributes after `fit`:**
- `fitted_graph: nx.Graph` — the best-fit graph found
- `metadata: dict` — contains `sigma`, `gic_value`, `best_iteration`, `spectrum_diffs`, `edge_diffs`, and more

### `GraphModelComparator`

Compares Logit Graph against baseline random graph models (ER, WS, BA, GRG).

**Constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_list` | `list[int]` | Values of `d` to search over for LG |
| `lg_params` | `dict` | LG generation settings (`max_iterations`, `patience`, `edge_delta`, `min_gic_threshold`, `er_p`) |
| `other_model_n_runs` | `int` | Number of independent runs per baseline model |
| `other_model_params` | `list\|None` | Explicit parameter grid for baseline models; defaults are used if `None` |
| `dist_type` | `str` | Spectral distance type (`'KL'`, `'L1'`, `'L2'`) |
| `verbose` | `bool` | Print progress |
| `other_models` | `list[str]` | Subset of `["ER", "WS", "GRG", "BA"]` |
| `other_model_grid_points` | `int` | Grid resolution for baseline model parameter sweep |

**Methods:**
- `compare(original_graph: nx.Graph, graph_filepath: str) -> self`

**Attributes after `compare`:**
- `summary_df: pd.DataFrame` — per-model GIC, spectral distance, and graph attributes
- `fitted_graphs_data: dict[str, {graph, metadata, attributes}]` — fitted graphs and metadata keyed by model name

### `LogitGraphSimulation`

Lower-level class for running and aggregating multiple LG simulation runs. Used internally by `GraphModelComparator`; can be used directly for custom simulation loops.

### `calculate_graph_attributes`

```python
from logit_graph import calculate_graph_attributes
attrs = calculate_graph_attributes(G)  # returns dict
```

Computes basic network properties: density, clustering coefficient, average path length, diameter, assortativity, largest connected component size, and more.

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
│   ├── __init__.py           # Exports: LogitGraphFitter, GraphModelComparator,
│   │                         #          LogitGraphSimulation, calculate_graph_attributes
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
