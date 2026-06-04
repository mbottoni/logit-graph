#!/usr/bin/env python3
"""Reproduce paper Figure 2: σ̂ convergence to true σ as n grows.

Runs the PAPER_SIGMA_CONVERGENCE sweep preset:
  σ ∈ {-2, -4, -6, -8}, d ∈ {0, 1, 2}, n ∈ {20, 50, 100, 300, 1000},
  n_reps=5, adaptive Gibbs stopping, iter_cap=300k (prevents the d=2
  cascade from biasing σ̂ at sparse-favored σ).

Produces a 3-panel figure (one per d) with σ̂ vs n on a log x-axis,
95% CI shaded bands, and dotted lines at the true σ values.

Env-var overrides:
  LG_SIGMA_JOBS         parallel job count          default 4
  LG_SIGMA_USE_CACHE    reuse cached CSV (0/1)      default 1
  LG_SIGMA_QUICK        smoke (n∈{20,50,100}, n_reps=2, iter_cap=20k)

  make sigma-convergence        full preset, ~10-15 min
  make sigma-convergence-quick  smoke, ~30 sec
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
_src = _repo_root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from logit_graph.experiments.presets import PRESETS  # noqa: E402
from logit_graph.experiments.sweeps import (  # noqa: E402
    plot_convergence_sigma,
    run_sigma_sweep,
)

OUT_DIR = _repo_root / "images" / "correction_paper"
FIG_PATH = OUT_DIR / "convergence_sigma.png"
FIG_PDF = OUT_DIR / "convergence_sigma.pdf"


def _default_jobs() -> int:
    try:
        return max(1, (os.cpu_count() or 4) - 1)
    except Exception:
        return 4


def main() -> None:
    cfg = PRESETS["PAPER_SIGMA_CONVERGENCE"]["sigma"]

    if os.environ.get("LG_SIGMA_QUICK", "0") == "1":
        # Apply smoke overrides
        cfg.n_values = [20, 50, 100]
        cfg.n_reps = 2
        cfg.iter_cap = 20_000

    n_jobs = int(os.environ.get("LG_SIGMA_JOBS", _default_jobs()))
    use_cache = os.environ.get("LG_SIGMA_USE_CACHE", "1") == "1"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"σ-convergence  n={cfg.n_values}  d={cfg.d_values}  σ={cfg.sigma_values}  "
        f"n_reps={cfg.n_reps}  iter_cap={cfg.iter_cap}  jobs={n_jobs}  cache={use_cache}"
    )

    df = run_sigma_sweep(cfg, OUT_DIR, use_cache=use_cache, n_jobs=n_jobs)
    print(f"\nCollected {len(df)} cells.")
    print(df[["n", "d", "sigma_true", "sigma_hat_mean", "ci_lo", "ci_hi", "density_mean"]]
          .sort_values(["d", "sigma_true", "n"])
          .to_string(index=False))

    plot_convergence_sigma(df, FIG_PATH)
    # Also save PDF for the paper
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_convergence_sigma(df, FIG_PDF)
    plt.close("all")
    print(f"\nSaved {FIG_PATH}")
    print(f"Saved {FIG_PDF}")


if __name__ == "__main__":
    main()
