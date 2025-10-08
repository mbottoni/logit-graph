# Logit Graph

A probabilistic logit-based graph model and utilities for fitting, simulating, and comparing random graph models to real-world networks. The package provides a simple, scikit-learn-like API to fit the Logit Graph model and benchmark it against classic random graph models (ER, WS, BA, optionally GRG), with metrics based on spectral distances and a Graph Information Criterion (GIC).

- **PyPI**: `logit-graph`
- **Python**: >=3.9
- **License**: MIT

## Installation

Install the published package from PyPI:

```bash
pip install logit-graph
```

Or, for local development within this repo:

```bash
pip install -e .
```

If you prefer using the full research environment (for notebooks, plotting, etc.), use the provided `requirements.txt` or `environment.yml`.

## Quickstart

### Fit a Logit Graph to a network

```python
import networkx as nx
from logit_graph import LogitGraphFitter

# Load or build your original graph (undirected)
G = nx.karate_club_graph()

# Configure and fit Logit Graph
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

# Grid of d for Logit Graph and generation settings
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
    other_models=["ER", "WS", "BA"],         # optionally include "GRG"
    other_model_grid_points=5
)

comparator = comparator.compare(original_graph=G, graph_filepath="karate_club")
print(comparator.summary_df)

# Access fitted graphs and metadata
lg_graph = comparator.fitted_graphs_data['LG']['graph']
lg_meta = comparator.fitted_graphs_data['LG']['metadata']
```

## Public API

The package exposes the following top-level entries:

- `logit_graph.LogitGraphFitter`
  - Fits a single Logit Graph to a given `networkx.Graph`.
  - Key init args: `d`, `n_iteration`, `warm_up`, `patience`, `dist_type`, `edge_delta`, `min_gic_threshold`, `verbose`, `er_p`.
  - Methods:
    - `fit(original_graph: nx.Graph) -> self`
  - Attributes after `fit`:
    - `fitted_graph: nx.Graph`
    - `metadata: dict` with `sigma`, `gic_value`, `best_iteration`, `spectrum_diffs`, `edge_diffs`, and more.

- `logit_graph.GraphModelComparator`
  - Compares Logit Graph against other random graph models.
  - Init args:
    - `d_list: list[int]`: values of `d` to try for LG
    - `lg_params: dict`: parameters forwarded to LG generation (e.g., `max_iterations`, `patience`, `edge_delta`, `min_gic_threshold`, `er_p`)
    - `other_model_n_runs: int`
    - `other_model_params: list|None` (optional). If omitted, sensible defaults are used per model.
    - `dist_type: str` (`'KL'`, etc.)
    - `verbose: bool`
    - `other_models: list[str]` (subset of `["ER", "WS", "GRG", "BA"]`)
    - `other_model_grid_points: int`
  - Methods:
    - `compare(original_graph: nx.Graph, graph_filepath: str) -> self`
  - After `compare`:
    - `summary_df: pandas.DataFrame` with per-model metrics and attributes
    - `fitted_graphs_data: dict[str, {graph: nx.Graph, metadata: dict, attributes: dict}]`

- `logit_graph.calculate_graph_attributes(graph: nx.Graph) -> dict`
  - Convenience function to compute basic properties (density, clustering, path length, diameter, assortativity, largest component size, etc.).

Notes:
- Internally, LG estimation uses logistic regression over local degree-based features and optimizes a spectral criterion (GIC) while generating edges.
- You can import lower-level utilities from submodules if needed (e.g., `logit_graph.graph`, `logit_graph.gic`, `logit_graph.logit_estimator`), but the high-level API above is recommended.

## Data and Notebooks

- Datasets used in experiments live under `data/` (many are compressed archives) and cover brain connectomes, social networks, Reddit threads, and more.
- Reproducible analysis and figures are in `notebooks/`, organized by dataset category.
- Generated images used in the paper are in `images/`.

These folders are not required for installing or using the pip package; they support reproducing the research and examples.

## Troubleshooting

- Some features (plotting, notebooks) require optional dependencies present in `requirements.txt`.
- If `igraph` or `pycairo` fail to install on your platform, install their system packages or wheels first, then `pip install logit-graph` again.
- For very large graphs, consider lowering `max_iterations`/`patience` or running comparisons with fewer models first.

## Development

- Build from source:
  ```bash
  python -m build
  ```
- Run tests and examples via the scripts and notebooks in `scripts/` and `notebooks/`.
- We welcome issues and PRs. See `project.urls` for links.

## Citation

If you use this package in academic work, please cite the project and link to the repository `https://github.com/maruanottoni/logit-graph`. A formal citation entry will be added upon publication.

```text
Ottoni, M. (2024). Logit Graph: probabilistic logit-based graph modeling and selection. GitHub repository. https://github.com/maruanottoni/logit-graph
```
