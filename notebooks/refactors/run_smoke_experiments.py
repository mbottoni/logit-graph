#!/usr/bin/env python3
"""Run SMOKE-tier paper experiments and save figures."""
from __future__ import annotations

import os
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from logit_graph.experiments import (
    PRESETS,
    plot_aic_confusion,
    plot_convergence_sigma,
    run_aic_d_sweep,
    run_sigma_sweep,
)

OUT = Path(__file__).resolve().parents[2] / "images" / "correction_paper"
OUT.mkdir(parents=True, exist_ok=True)

MODE = os.environ.get("LG_EXPERIMENT_MODE", "INSIGHT")
cfg_sigma = PRESETS[MODE]["sigma"]
cfg_aic = PRESETS[MODE]["aic"]

print(f"Mode={MODE}, sigma n={cfg_sigma.n_values}, aic n={cfg_aic.n_sizes}")

df = run_sigma_sweep(cfg_sigma, OUT, use_cache=False)
plot_convergence_sigma(df, OUT / "convergence_sigma.png")
print(f"Saved {OUT / 'convergence_sigma.png'}")

_, conf = run_aic_d_sweep(cfg_aic, OUT, use_cache=False)
plot_aic_confusion(conf, cfg_aic.d_true_values, OUT / "aic_d_confusion_n_sweep.png")
print(f"Saved {OUT / 'aic_d_confusion_n_sweep.png'}")
