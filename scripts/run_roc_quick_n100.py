#!/usr/bin/env python3
"""Quick ROC sanity check at n=100."""
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

    OUT = Path(__file__).resolve().parents[1] / "images" / "correction_paper" / "roc_quick_n100"
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = ROCSweepConfig(
        n_effect=100,
        n_values=[10, 100],
        sigma2_values=[-1.0, -1.5, -2.0, -2.5],
        d_values=[0, 1, 2],
        n_reps=5,
        n_experiments=30,
        iter_cap=15_000,
        seed_base=9000,
    )

    print(
        f"Quick ROC: n_effect={cfg.n_effect}, n_values={cfg.n_values}, "
        f"d={cfg.d_values}, reps={cfg.n_reps}, exps={cfg.n_experiments}, "
        f"iter_cap={cfg.iter_cap}",
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
