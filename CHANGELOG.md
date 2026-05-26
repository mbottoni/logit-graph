# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Platform batch notebooks with step logging and per-network fit reports (`notebooks/refactors/`).

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
