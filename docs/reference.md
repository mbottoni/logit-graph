# API Reference

Auto-generated from the source docstrings. For a task-oriented walkthrough with examples,
see the [API Guide](API.md).

All symbols below are importable directly from the top-level `logit_graph` package, e.g.
`from logit_graph import simulate_graph`.

## The model — Logistic Random Graph (LG)

There is a single model, the **Logistic Random Graph (LG)** of Ottoni, Takahashi & Fujita (2026);
the entries below are its computational forms and the general-pairwise-features extension (§3.7).
All score an edge with a logistic link and differ only in what enters the linear predictor `η_ij`.

| Form | Edge log-odds `η_ij` | Packaged as |
|------|----------------------|-------------|
| **Equilibrium sampler** | `σ + α·[g(S_i)+g(S_j)]` at stationarity (Gibbs) | `simulate_graph`, `GraphModel` |
| **Iterative generation** | same, with `S` evaluated on the lagged graph `G(t-1)` (Eq. 3.5) | [LG generation & estimation](#lg-iterative-generation-estimation) |
| **General pairwise features (§3.7)** | `σ + Σₖ θₖ·φₖ(i,j)` — degree term **plus** extra features | [Unified LG](#unified-lg-general-pairwise-features) |

- **`σ`** — baseline connection propensity (baseline log-odds; typically negative).
- **`α ≥ 0`** — strength of the neighborhood degree effect on `g(S_i)+g(S_j)`, `g(s)=log(1+s)`,
  `S_i = Σ_{l∈N_d(i)} k_l` the neighborhood degree sum over the `d`-hop ball.
- **`d`** — the neighborhood radius (hops). `d=0` uses each node's own degree only (`S_i = k_i`);
  `d=1` adds the neighbors' degrees; larger `d` reaches further. Selected from data by AIC.
- **`φₖ` (extra features)** — appended columns: exogenous / pair-local ones (e.g. block membership via
  `community_feature`) are directly estimable; globally-coupled ones (e.g. latent proximity via
  `latent_feature`, or the degree term itself) are made consistent by the **predetermined-predictor
  (temporal)** form that reads them off `G(t-1)`. The *unified five-feature LG model* of the thesis is the
  instance with degree + coarse/fine community (`γc`, `γf`) + latent proximity (`λ`).

## Paper-consistent sampler & estimator

::: logit_graph.simulate_graph
::: logit_graph.select_d_ensemble
::: logit_graph.estimate_sigma_from_graph

## High-level API (scikit-learn style)

::: logit_graph.LogitGraphFitter
::: logit_graph.LogitGraphSimulation
::: logit_graph.GraphModelComparator
::: logit_graph.estimate_sigma_only
::: logit_graph.estimate_sigma_many
::: logit_graph.calculate_graph_attributes

## Core model & estimator

::: logit_graph.GraphModel
::: logit_graph.LogitRegEstimator

## Features

::: logit_graph.build_pair_dataset
::: logit_graph.pair_feature
::: logit_graph.pair_feature_layer2
::: logit_graph.incremental_h
::: logit_graph.recommended_iterations

## LG: iterative generation & estimation

The LG iterative generation algorithm and its estimator: `logit Pr[A_ij(t)=1 | G(t-1)] = σ + α·D(t-1)`
(Eq. 3.5), with the degree term `D = g(S_i)+g(S_j)` read from the predetermined previous graph so the
pooled dyad design is an ordinary logistic regression and the MLE of `(σ, α)` is consistent.

::: logit_graph.grow_graph
::: logit_graph.GrowthResult
::: logit_graph.growth_design_from_snapshots
::: logit_graph.fit_growth_params
::: logit_graph.fit_growth_from_result

## Unified LG: general pairwise features

The §3.7 extension `logit p_ij = σ + Σₖ θₖ·φₖ(i,j)`: the degree term plus arbitrary appended pairwise
features. `grow_graph_multi` / `fit_multi_params` mirror the degree-only API but carry the extra features;
`community_feature` (block membership) and `latent_feature` (latent proximity) build the canonical
covariates of the thesis's unified five-feature LG model from an observed graph.

::: logit_graph.grow_graph_multi
::: logit_graph.MultiGrowthResult
::: logit_graph.multi_design_from_snapshots
::: logit_graph.fit_multi_params
::: logit_graph.community_feature
::: logit_graph.latent_feature

## Experiment configuration

::: logit_graph.AICSweepConfig
::: logit_graph.SigmaSweepConfig
