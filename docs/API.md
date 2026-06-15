# API Reference

Detailed reference for the main `logit_graph` exports. For a quick start, see the [Home page](index.md);
for the full docstring-generated reference, see the [API Reference](reference.md).

The **paper-consistent** pipeline is:

`simulate_graph` → `select_d_ensemble` → `estimate_sigma_from_graph` → `GraphModelComparator`

---

## `simulate_graph`

Generate a random graph at fixed `(n, d, σ)`.

```python
from logit_graph import simulate_graph

adj = simulate_graph(
    n=200, d=1, sigma=-4.0, n_iter=30_000,
    feature_mode="incremental", target_density=0.10, seed=42,
)
# adj, meta = simulate_graph(..., return_meta=True)
```

| Parameter | Description |
|-----------|-------------|
| `n`, `d`, `sigma` | Graph size, feature depth, logit intercept |
| `n_iter` | Gibbs iterations (`d≥1`) or ignored (`d=0`, direct ER) |
| `feature_mode` | `"incremental"` (paper mode), `"bounded"`, or `"full"` |
| `target_density` | Used when calibrating `β` if `sigma` is omitted |
| `return_meta` | If `True`, return `(adj, meta)` with fitted `σ`, `β`, density |

---

## `select_d_ensemble`

Pick `d̂` by AIC over candidate depths using the Layer-2 offset logit.

```python
from logit_graph import select_d_ensemble

d_hat, aic_stats = select_d_ensemble(
    graphs=[adj],
    d_candidates=[0, 1, 2, 3],
    feature_mode="incremental",
    extra_penalty_per_d=0.0,
)
# aic_stats[d] → {"aic", "ll", "sigma_hat", "n_obs", …}
```

---

## `estimate_sigma_from_graph`

Offset-logit estimate of `σ̂` at a fixed `d`.

```python
from logit_graph import estimate_sigma_from_graph

sigma_hat = estimate_sigma_from_graph(adj, d=1, feature_mode="incremental")
```

---

## `GraphModelComparator`

Compare **LG** against baseline models using spectral GIC (**lower = better**).

```python
from logit_graph import GraphModelComparator

comparator = GraphModelComparator(
    d_list=[d_hat],
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
    random_state=0,  # >= 0.1.3
).compare(original_graph=G, graph_filepath="my_graph")

comparator.summary_df
comparator.fitted_graphs_data["LG"]
```

Pass `d_list=[d_hat]` where `d_hat` comes from `select_d_ensemble` for paper-consistent selection.

---

## `LogitGraphFitter`

Sklearn-style fitter at a **fixed** `d`: estimate `σ` and search for a graph minimising spectral GIC.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d` | `0` | Neighborhood depth for degree-sum features |
| `n_iteration` | `10000` | Max edge-swap / Gibbs iterations |
| `warm_up` | `500` | Burn-in before GIC tracking |
| `patience` | `2000` | Early-stop patience |
| `dist_type` | `"KL"` | `"KL"`, `"L1"`, or `"L2"` |
| `min_gic_threshold` | `5` | Min GIC drop to reset patience |
| `er_p` | `0.05` | ER probability for warm-start graph |
| `verbose` | `True` | Print progress |

After `fit(G)`: `fitter.fitted_graph`, `fitter.metadata`.

---

## Other exports

| Symbol | Role |
|--------|------|
| `LogitGraphSimulation` | Lower-level multi-run LG simulation |
| `LogitRegEstimator` | Layer-2 offset logit on pair features |
| `calculate_graph_attributes` | Density, clustering, diameter, assortativity |
| `recommended_iterations` | Suggested Gibbs length vs `n` |
| `build_pair_dataset`, `pair_feature`, `pair_feature_layer2` | Feature construction |
| `GraphModel` | Core Gibbs / edge-swap engine |
| `AICSweepConfig`, `SigmaSweepConfig`, `PRESETS` | Experiment presets (`logit_graph.experiments`) |

---

## Model summary

Logistic Random Graph (LG) model, Ottoni–Takahashi–Fujita (2026).

**Edge probability (Eq. 3.1):** `p_ij = logistic(σ + α·[g(S_i) + g(S_j)])`, with `g(s) = log(1 + s)`
and `S_i = Σ_{l ∈ N_d(i)} k_l` the neighborhood degree sum over the `d`-hop ball. `σ` is the baseline
connection propensity; `α ≥ 0` is the strength of the neighborhood degree effect.

**Extension (§3.7):** append pairwise features — `logit p_ij = σ + Σₖ θₖ·φₖ(i,j)`.

**GIC:** `GIC = 2 · KL(observed ‖ generated) + 2 · |θ|` (lower = better); `KL` is the spectral KL
divergence between normalized-Laplacian spectral densities and is the un-penalized goodness-of-fit.

**Baselines:** ER, WS, BA, optionally GRG / KR / SBM.

**Spectral distances:** KL (default), L1, L2.
