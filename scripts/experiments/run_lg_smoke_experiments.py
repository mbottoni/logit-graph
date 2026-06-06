#!/usr/bin/env python3
"""Run SMOKE-tier paper experiments and save figures."""
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
        plot_convergence_sigma,
        run_aic_d_sweep,
        run_sigma_sweep,
    )

    OUT = Path(__file__).resolve().parents[2] / "images" / "correction_paper"
    OUT.mkdir(parents=True, exist_ok=True)

    MODE = os.environ.get("LG_EXPERIMENT_MODE", "INSIGHT")
    USE_CACHE = os.environ.get("LG_SMOKE_USE_CACHE", "1") == "1"
    N_JOBS = int(os.environ.get("LG_SMOKE_JOBS", _default_jobs()))

    cfg_sigma = PRESETS[MODE]["sigma"]
    cfg_aic = PRESETS[MODE]["aic"]

    print(
        f"Mode={MODE}, sigma n={cfg_sigma.n_values}, aic n={cfg_aic.n_sizes}, "
        f"cache={USE_CACHE}, jobs={N_JOBS}",
    )

    df = run_sigma_sweep(cfg_sigma, OUT, use_cache=USE_CACHE, n_jobs=N_JOBS)
    plot_convergence_sigma(df, OUT / "convergence_sigma.png")
    print(f"Saved {OUT / 'convergence_sigma.png'}")

    _, conf = run_aic_d_sweep(cfg_aic, OUT, use_cache=USE_CACHE, n_jobs=N_JOBS)
    plot_aic_confusion(conf, cfg_aic.d_true_values, OUT / "aic_d_confusion_n_sweep.png")
    print(f"Saved {OUT / 'aic_d_confusion_n_sweep.png'}")


if __name__ == "__main__":
    main()
