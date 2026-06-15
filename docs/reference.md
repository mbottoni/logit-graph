# API Reference

Auto-generated from the source docstrings. For a task-oriented walkthrough with examples,
see the [API Guide](API.md).

All symbols below are importable directly from the top-level `logit_graph` package, e.g.
`from logit_graph import simulate_graph`.

## The model family

Every model here scores edges with a logistic link; they differ in what enters the linear predictor.

| Model | Edge log-odds | Packaged as |
|-------|---------------|-------------|
| **Equilibrium LG** | `σ` + degree feature (Gibbs at stationarity) | `simulate_graph`, `GraphModel` |
| **Temporal LG (TLG)** | `σ + α·D(t-1)` — degree from the predetermined snapshot | [Temporal LG](#temporal-growth-logit-graph) |
| **Multi-feature TLG** | `σ + α·D(t-1) + Σ βₖ·Fₖ` — degree **plus** fixed exogenous dyad covariates | [Multi-feature TLG](#multi-feature-unified-temporal-logit-graph) |

- **`σ` (intercept)** — baseline edge log-odds (controls density).
- **`α` (degree slope)** — how a dyad's degree feature `D` shifts its edge probability (hubs / preferential attachment).
- **`Fₖ` (extra covariates)** — fixed, exogenous per-dyad features such as same-community indicators
  (`community_feature`) or latent-embedding proximity (`latent_feature`). Because every `Fₖ` and the
  lagged `D` are predetermined, the pooled dyad design is an ordinary logistic regression whose MLE
  recovers `(σ, α, β₁…βₖ)` consistently — the property that makes the multi-feature model identifiable.

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

## Temporal / growth Logit-Graph

The degree-only temporal model: `logit P[edge_ij at t] = σ + α·D(t-1)`, with `D` read from the
predetermined previous snapshot so the pooled dyad design is an ordinary logistic regression.

::: logit_graph.grow_graph
::: logit_graph.GrowthResult
::: logit_graph.growth_design_from_snapshots
::: logit_graph.fit_growth_params
::: logit_graph.fit_growth_from_result

## Multi-feature (unified) Temporal Logit-Graph

The TLG extended with fixed exogenous dyad covariates:
`logit P[edge_ij at t] = σ + α·D(t-1) + Σ βₖ·Fₖ`. `grow_graph_multi` / `fit_multi_params` mirror the
degree-only API but carry arbitrary extra features; `community_feature` and `latent_feature` build the
canonical same-community and latent-embedding covariates from an observed graph.

::: logit_graph.grow_graph_multi
::: logit_graph.MultiGrowthResult
::: logit_graph.multi_design_from_snapshots
::: logit_graph.fit_multi_params
::: logit_graph.community_feature
::: logit_graph.latent_feature

## Experiment configuration

::: logit_graph.AICSweepConfig
::: logit_graph.SigmaSweepConfig
