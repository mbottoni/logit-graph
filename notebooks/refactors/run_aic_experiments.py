#!/usr/bin/env python3
"""Run fast AIC d-selection insight experiments (default INSIGHT tier)."""
from __future__ import annotations

import os
from pathlib import Path


def _default_jobs() -> int:
    return min(4, max(1, (os.cpu_count() or 2) - 1))


def main() -> None:
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    from logit_graph.experiments import (
        PRESETS,
        plot_aic_confusion,
        run_aic_d_sweep,
        summarize_aic_insights,
    )

    OUT = Path(__file__).resolve().parents[2] / "images" / "correction_paper"
    OUT.mkdir(parents=True, exist_ok=True)

    MODE = os.environ.get("LG_EXPERIMENT_MODE", "INSIGHT")
    USE_CACHE = os.environ.get("LG_AIC_USE_CACHE", "1") == "1"
    N_JOBS = int(os.environ.get("LG_AIC_JOBS", _default_jobs()))

    cfg = PRESETS[MODE]["aic"]
    print(
        f"Mode={MODE}, n={cfg.n_sizes}, runs={cfg.n_runs}, "
        f"M={cfg.m_ensemble}, iter_cap={cfg.iter_cap}, pen={cfg.aic_penalty_per_d}, "
        f"cache={USE_CACHE}, jobs={N_JOBS}",
    )

    _, conf = run_aic_d_sweep(cfg, OUT, use_cache=USE_CACHE, n_jobs=N_JOBS)
    plot_aic_confusion(conf, cfg.d_true_values, OUT / "aic_d_confusion_n_sweep.png")
    print(f"Saved {OUT / 'aic_d_confusion_n_sweep.png'}")

    summary = summarize_aic_insights(conf, cfg.d_true_values)
    print(summary)
    (OUT / "aic_d_insights.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
