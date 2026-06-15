# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0] - 2026-06-15

### Added
- Temporal / growth **Logistic Random Graph (LG)** model: `grow_graph`, `fit_growth_params`,
  `fit_growth_from_result`, `growth_design_from_snapshots`, `GrowthResult` — generation and
  consistent leave-one-out (full conditional) estimation of (σ, α).
- **Unified (multi-feature) LG** — the general-pairwise-features extension (paper §3.7):
  `grow_graph_multi`, `fit_multi_params`, `multi_design_from_snapshots`, `MultiGrowthResult`,
  plus the `community_feature` (block membership) and `latent_feature` (latent proximity) builders.
- Dyadic-cluster-robust standard errors (`robust_se`) and a Stochastic Block Model baseline (`sbm`).
- MkDocs + Material documentation site with a mkdocstrings API reference, ready for Read the Docs.
- Tag-triggered PyPI release workflow via GitHub Actions trusted publishing (OIDC, no stored token).
- Platform batch notebooks with step logging and per-network fit reports (`notebooks/refactors/`).

### Changed
- Documentation and docstrings aligned with the paper/thesis: the model is the **Logistic Random
  Graph (LG)** model, Eq. 3.1 notation, and **leave-one-out (full conditional)** terminology
  (replacing the internal "Layer-2" jargon).
- Trimmed comments and docstrings across `src/` and `scripts/` to ≤3 lines.
- Fixed stale repository URLs in `pyproject.toml`.

## [0.1.4] - 2026-05-26

### Added
- Top-level [`examples/`](examples/) folder with PyPI-friendly tutorial notebooks.
- `random_state` on `GraphModelComparator` for reproducible GIC comparisons.
- Seeded baseline model sampling in `GraphModelSelection`.

### Changed
- README and project layout: examples moved from `notebooks/examples/` to `examples/`.

## [0.1.3] - 2026-05-25

### Added
- Reproducible model comparison via `GraphModelComparator(random_state=...)`.
- Example notebook for real-network GIC comparison (`pypi_fit_real_network`).

## [0.1.2] - 2026-05-24

### Added
- Paper-consistent public API: `simulate_graph`, `select_d_ensemble`, `estimate_sigma_from_graph`, `GraphModelComparator`.
- PyPI package `logit-graph`.

[Unreleased]: https://github.com/mbottoni/logit-graph/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/mbottoni/logit-graph/releases/tag/v0.1.4
[0.1.3]: https://github.com/mbottoni/logit-graph/releases/tag/v0.1.3
[0.1.2]: https://github.com/mbottoni/logit-graph/releases/tag/v0.1.2
