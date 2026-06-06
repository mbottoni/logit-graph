#!/usr/bin/env python3
"""Quick ROC at n=1000 for d=0 and d=1 only."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from logit_graph.experiments.presets import ROCSweepConfig
from logit_graph.experiments.sweeps import (
    plot_roc_effect_size,
    plot_roc_sample_size,
    run_roc_sweeps,
)


def main() -> None:
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    OUT = Path(__file__).resolve().parents[2] / "images" / "correction_paper" / "roc_quick_n1000_d01"
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = ROCSweepConfig(
        n_effect=1000,
        n_values=[1000],
        sigma2_values=[-1.0, -1.5, -2.0, -2.5],
        d_values=[0, 1],
        n_reps=5,
        n_experiments=30,
        iter_cap=None,
        adaptive_stopping=True,
        adaptive_check_interval=25_000,
        adaptive_patience=4,
        adaptive_cv_tol=0.015,
        adaptive_min_iter=50_000,
        seed_base=9100,
    )

    print(
        f"ROC n=1000 d={{0,1}}: n_effect={cfg.n_effect}, n_values={cfg.n_values}, "
        f"reps={cfg.n_reps}, exps={cfg.n_experiments}, adaptive={cfg.adaptive_stopping}",
        flush=True,
    )

    effect_df, sample_df = run_roc_sweeps(
        cfg, OUT, use_cache=False, n_jobs=1, cell_jobs=1,
    )
    combined = pd.concat([effect_df, sample_df], ignore_index=True)

    plot_roc_effect_size(
        combined, OUT / "roc_effect_size.png",
        sigma1=cfg.sigma1, n_fixed=cfg.n_effect,
    )
    plot_roc_sample_size(
        combined, OUT / "roc_sample_size.png",
        sigma1=cfg.sigma1, sigma2=cfg.sigma2_fixed,
    )

    print(f"\nSaved {OUT / 'roc_effect_size.png'}")
    print(f"Saved {OUT / 'roc_sample_size.png'}")
    print("\nEffect-size (reject @ alpha=0.05):")
    print(
        effect_df.groupby(["d", "sigma2"])["power_at_005"].first().reset_index().to_string(index=False)
    )
    print("\nSample-size (reject @ alpha=0.05):")
    print(
        sample_df.groupby(["d", "n"])["power_at_005"].first().reset_index().to_string(index=False)
    )


if __name__ == "__main__":
    main()
