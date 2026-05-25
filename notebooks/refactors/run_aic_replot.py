#!/usr/bin/env python3
"""Replot AIC confusion figure from saved sweep artifacts (partial runs OK)."""
from __future__ import annotations

import os
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from logit_graph.experiments import (
    PRESETS,
    load_aic_sweep_results,
    plot_aic_confusion,
    summarize_aic_insights,
)
from logit_graph.experiments.sweeps import (
    aic_results_json_path,
    aic_trials_path,
)

OUT = Path(__file__).resolve().parents[2] / "images" / "correction_paper"
MODE = os.environ.get("LG_EXPERIMENT_MODE", "PAPER")
FIG_OUT = Path(os.environ.get("LG_AIC_FIG_OUT", OUT / "aic_d_confusion_n_sweep.png"))
INSIGHTS_OUT = Path(os.environ.get("LG_AIC_INSIGHTS_OUT", OUT / "aic_d_insights.txt"))

cfg = PRESETS[MODE]["aic"]
trials_csv = Path(os.environ["LG_AIC_TRIALS_CSV"]) if "LG_AIC_TRIALS_CSV" in os.environ else None

if trials_csv is not None:
    import pandas as pd
    from logit_graph.experiments.sweeps import _confusion_from_df

    df = pd.read_csv(trials_csv)
    conf = _confusion_from_df(df, cfg)
    print(f"Loaded {len(df)} trials from {trials_csv}")
else:
    df, conf = load_aic_sweep_results(OUT, cfg)
    print(
        f"Loaded {len(df)} trials from {aic_trials_path(OUT, cfg)} "
        f"(expected {len(cfg.n_sizes) * len(cfg.d_true_values) * cfg.n_runs})",
    )
    json_path = aic_results_json_path(OUT, cfg)
    if json_path.is_file():
        print(f"Metadata: {json_path}")

plot_aic_confusion(conf, cfg.d_true_values, FIG_OUT)
print(f"Saved {FIG_OUT}")

summary = summarize_aic_insights(conf, cfg.d_true_values)
print(summary)
INSIGHTS_OUT.write_text(summary + "\n")
print(f"Saved {INSIGHTS_OUT}")
