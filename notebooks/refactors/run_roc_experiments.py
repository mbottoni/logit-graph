#!/usr/bin/env python3
"""Run ANOVA ROC sweeps (paper fig:roc_effect / fig:roc_sample)."""
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
        plot_roc_effect_size,
        plot_roc_sample_size,
        run_roc_sweeps,
    )

    OUT = Path(__file__).resolve().parents[2] / "images" / "correction_paper"
    OUT.mkdir(parents=True, exist_ok=True)

    MODE = os.environ.get("LG_EXPERIMENT_MODE", "SMOKE")
    USE_CACHE = os.environ.get("LG_ROC_USE_CACHE", "1") == "1"
    N_JOBS = int(os.environ.get("LG_ROC_JOBS", _default_jobs()))

    cfg = PRESETS[MODE]["roc"]
    if "LG_ROC_N_EXPERIMENTS" in os.environ:
        cfg.n_experiments = int(os.environ["LG_ROC_N_EXPERIMENTS"])
    print(
        f"Mode={MODE}, n_effect={cfg.n_effect}, n_values={cfg.n_values}, "
        f"d={cfg.d_values}, reps={cfg.n_reps}, exps={cfg.n_experiments}, "
        f"iter_cap={cfg.iter_cap}, jobs={N_JOBS}",
    )

    effect_df, sample_df = run_roc_sweeps(cfg, OUT, use_cache=USE_CACHE, n_jobs=N_JOBS)
    combined = __import__("pandas").concat([effect_df, sample_df], ignore_index=True)

    plot_roc_effect_size(
        combined, OUT / "roc_effect_size.png",
        sigma1=cfg.sigma1, n_fixed=cfg.n_effect,
    )
    plot_roc_sample_size(
        combined, OUT / "roc_sample_size.png",
        sigma1=cfg.sigma1, sigma2=cfg.sigma2_fixed,
    )
    print(f"Saved {OUT / 'roc_effect_size.png'}")
    print(f"Saved {OUT / 'roc_sample_size.png'}")

    summary = (
        effect_df.groupby(["d", "sigma2"])["power_at_005"].first().reset_index()
        .rename(columns={"power_at_005": "reject_at_005"})
    )
    print("\nEffect-size sweep (reject rate at alpha=0.05):")
    print(summary.to_string(index=False))

    summary_n = (
        sample_df.groupby(["d", "n"])["power_at_005"].first().reset_index()
        .rename(columns={"power_at_005": "reject_at_005"})
    )
    print("\nSample-size sweep (reject rate at alpha=0.05):")
    print(summary_n.to_string(index=False))


if __name__ == "__main__":
    main()
