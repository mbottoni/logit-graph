#!/usr/bin/env python3
"""Replot convergence_sigma from saved CSV / per-cell npz caches (partial runs OK)."""
from __future__ import annotations

import os
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from logit_graph.experiments import PRESETS, plot_convergence_sigma
from logit_graph.experiments.sweeps import (
    load_sigma_sweep_df,
    sigma_sweep_csv_path,
    sigma_sweep_results_json_path,
)

OUT = Path(__file__).resolve().parents[2] / "images" / "correction_paper"
MODE = os.environ.get("LG_EXPERIMENT_MODE", "PAPER")
cfg = PRESETS[MODE]["sigma"]

df = load_sigma_sweep_df(cfg, OUT, use_cache=True)
expected = len(cfg.d_values) * len(cfg.sigma_values) * len(cfg.n_values)
print(f"Loaded {len(df)}/{expected} cells from {sigma_sweep_csv_path(OUT, cfg)}")

fig_path = OUT / "convergence_sigma.png"
plot_convergence_sigma(df, fig_path)
print(f"Saved {fig_path}")
print(f"Metadata: {sigma_sweep_results_json_path(OUT, cfg)}")
