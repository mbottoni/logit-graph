# Logit Graph

**Fit, simulate, and compare logit graph models against ER / WS / BA using spectral GIC.**

`logit-graph` is a probabilistic, paper-consistent random-graph model with Layer-2
estimation and a scikit-learn-style API. It lets you:

| | |
|---|---|
| **Estimate** | Pick the neighborhood depth `d̂` by AIC and the intercept `σ̂` from a real graph |
| **Generate** | Sample graphs whose normalized-Laplacian spectrum matches the data (GIC-guided Gibbs) |
| **Compare** | Rank LG against Erdős–Rényi, Watts–Strogatz, and Barabási–Albert on the same spectral GIC |

## Installation

```bash
pip install "logit-graph>=0.1.3"
```

## Quickstart

The paper-consistent pipeline is
`simulate_graph` → `select_d_ensemble` → `estimate_sigma_from_graph` → `GraphModelComparator`:

```python
from logit_graph import simulate_graph, select_d_ensemble, estimate_sigma_from_graph

# Generate a graph at a known (n, d, sigma)
adj = simulate_graph(n=200, d=1, sigma=-4.0, n_iter=30_000,
                     feature_mode="incremental", target_density=0.10, seed=42)

# Recover the neighborhood depth by AIC, then the intercept
d_hat, aic_stats = select_d_ensemble(graphs=[adj], d_candidates=[0, 1, 2, 3])
sigma_hat = estimate_sigma_from_graph(adj, d=d_hat, feature_mode="incremental")
```

## Where to next

- **[API Guide](API.md)** — task-oriented walkthrough of the main entry points with examples.
- **[API Reference](reference.md)** — full reference generated from the source docstrings.
- **[Connectomes case study](connectomes.md)** — applying LG to real brain networks.

## Model in one line

Edge probability `P(i–j) = sigmoid(σ · (deg_d(i) + deg_d(j)))`; models are ranked by
`GIC = 2 · spectral_distance(original, fitted) + 2 · |θ|` (lower is better), where the
spectral distance defaults to the KL divergence between normalized-Laplacian spectral densities.
