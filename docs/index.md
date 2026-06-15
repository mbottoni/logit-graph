# Logit Graph

**Fit, simulate, and compare the Logistic Random Graph (LG) model against ER / WS / BA using spectral GIC.**

`logit-graph` is the Python package for the **Logistic Random Graph (LG) model** — a logistic
random-graph model with neighborhood degree effects — with maximum-likelihood estimation,
AIC-based model selection, and a scikit-learn-style API. It lets you:

| | |
|---|---|
| **Estimate** | Pick the neighborhood radius `d̂` by AIC and the parameters `σ̂` (baseline) and `α̂` (degree effect) from a real graph |
| **Generate** | Run the LG iterative generation algorithm so the normalized-Laplacian spectrum matches the data |
| **Compare** | Rank LG against Erdős–Rényi, Watts–Strogatz, and Barabási–Albert on the same spectral GIC |

!!! note "Reference"
    This package accompanies Ottoni, Takahashi & Fujita, *A Logistic Random Graph Model with
    Neighborhood Degree Effects* (IMA Journal of Complex Networks, 2026), and the corresponding
    master's thesis. Notation and model names below follow the paper.

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

# Recover the neighborhood radius d by AIC, then the baseline sigma
d_hat, aic_stats = select_d_ensemble(graphs=[adj], d_candidates=[0, 1, 2, 3])
sigma_hat = estimate_sigma_from_graph(adj, d=d_hat, feature_mode="incremental")
```

## Where to next

- **[API Guide](API.md)** — task-oriented walkthrough of the main entry points with examples.
- **[API Reference](reference.md)** — full reference generated from the source docstrings.
- **[Connectomes case study](connectomes.md)** — applying LG to real brain networks.

## The model (paper notation)

Edge probability via the logistic link (Eq. 3.1):

`p_ij = logistic(η_ij)`,  `η_ij = σ + α·[g(S_i) + g(S_j)]`,  `g(s) = log(1 + s)`

where `S_i = Σ_{l ∈ N_d(i)} k_l` is the **neighborhood degree sum** over the `d`-hop ball
`N_d(i) = {l : dist(i,l) ≤ d}`, **σ** is the baseline connection propensity (baseline log-odds,
typically negative), and **α ≥ 0** is the strength of the **neighborhood degree effect**.

The **radius `d`** controls how far the degree feature reaches: `d=0` uses each node's own degree
only (`N_0(i)={i}`, so `S_i = k_i`); `d=1` adds the neighbors' degrees; larger `d` aggregates over a
wider ball. `d` is selected from the data by AIC (`select_d_ensemble`). The iterative generation algorithm
draws each pair from its predetermined value on the lagged graph `G(t-1)` (Eq. 3.5), which is also
what makes maximum-likelihood estimation of `(σ, α)` consistent.

**Extension (§3.7).** Any pairwise features can be appended: `logit p_ij = σ + Σₖ θₖ·φₖ(i,j)`. The
*unified five-feature LG model* of the thesis adds coarse/fine community indicators (`γc`, `γf`) and a
latent-proximity feature (`λ`) to the degree term.

Models are ranked by `GIC = 2·KL + 2·|θ|` (lower is better), where `KL` is the Kullback–Leibler
divergence between the normalized-Laplacian spectral densities of the observed and generated graphs
and `|θ|` is the number of free parameters; the un-penalized `KL` is reported as the goodness-of-fit.
