#!/usr/bin/env python3
"""Replot ROC figures from completed per-cell caches (partial runs OK)."""
from __future__ import annotations

import os
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import pandas as pd

from logit_graph.experiments import PRESETS, plot_roc_effect_size, plot_roc_sample_size
from logit_graph.experiments.sweeps import (
    _config_hash,
    _load_rows_from_cell_caches,
)

OUT = Path(__file__).resolve().parents[2] / "images" / "correction_paper"
MODE = os.environ.get("LG_EXPERIMENT_MODE", "PAPER")
cfg = PRESETS[MODE]["roc"]
if "LG_ROC_N_EXPERIMENTS" in os.environ:
    cfg.n_experiments = int(os.environ["LG_ROC_N_EXPERIMENTS"])

cfg_hash = _config_hash(cfg)
rows = _load_rows_from_cell_caches(cfg, OUT, cfg_hash)
if not rows:
    raise SystemExit("No completed ROC cell caches found.")

df = pd.DataFrame(rows)
effect = df[df.sweep == "effect"]
sample = df[df.sweep == "sample"]
print(f"Cells loaded: effect={effect.groupby(['d','sigma2']).ngroups}, "
      f"sample={sample.groupby(['d','n']).ngroups}")

if not effect.empty:
    plot_roc_effect_size(df, OUT / "roc_effect_size.png", sigma1=cfg.sigma1, n_fixed=cfg.n_effect)
    print(f"Saved {OUT / 'roc_effect_size.png'}")
if not sample.empty:
    plot_roc_sample_size(df, OUT / "roc_sample_size.png", sigma1=cfg.sigma1, sigma2=cfg.sigma2_fixed)
    print(f"Saved {OUT / 'roc_sample_size.png'}")
