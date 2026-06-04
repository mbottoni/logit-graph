#!/usr/bin/env python3
"""Compare serial vs rep-parallel ROC on one cell + optional mini sweep."""
from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from logit_graph.experiments.presets import ROCSweepConfig
from logit_graph.experiments.sweeps import (
    _roc_parallel_plan,
    collect_anova_pvalues,
    run_roc_sweeps,
)


def _bench_cell(
    *,
    n: int,
    d: int,
    n_reps: int,
    n_experiments: int,
    n_iter: int,
    n_jobs: int,
    rep_jobs: int | None,
) -> tuple[float, float]:
    common = dict(
        n=n,
        d=d,
        sigma1=-1.0,
        sigma2=-1.5,
        n_reps=n_reps,
        n_experiments=n_experiments,
        n_iter=n_iter,
        feature_mode_gen="incremental",
        feature_mode_est="incremental",
        target_density=0.10,
        signal=0.5,
        seed_base=99,
        n_jobs=n_jobs,
        rep_use_threads=True,
    )
    exp_jobs, inner_rep = _roc_parallel_plan(n, n_reps, n_jobs, rep_jobs)
    t0 = time.perf_counter()
    pvals = collect_anova_pvalues(**common, rep_jobs=inner_rep)
    return time.perf_counter() - t0, float(pvals.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini-sweep", action="store_true")
    args = parser.parse_args()

    n_jobs = int(os.environ.get("LG_ROC_JOBS", "4"))
    for n in (80, 200):
        exp_jobs, auto_rep = _roc_parallel_plan(n, 5, n_jobs, None)
        print(f"n={n}, jobs={n_jobs} -> exp_jobs={exp_jobs}, rep_jobs={auto_rep}")

    t_serial, _ = _bench_cell(
        n=80, d=1, n_reps=5, n_experiments=6, n_iter=8000,
        n_jobs=1, rep_jobs=1,
    )
    t_small, _ = _bench_cell(
        n=80, d=1, n_reps=5, n_experiments=6, n_iter=8000,
        n_jobs=n_jobs, rep_jobs=None,
    )
    print(f"\nSingle cell n=80: serial={t_serial:.1f}s jobs={n_jobs}={t_small:.1f}s")

    t_serial200, _ = _bench_cell(
        n=200, d=1, n_reps=5, n_experiments=4, n_iter=3000,
        n_jobs=1, rep_jobs=1,
    )
    t_large, _ = _bench_cell(
        n=200, d=1, n_reps=5, n_experiments=4, n_iter=3000,
        n_jobs=n_jobs, rep_jobs=None,
    )
    print(
        f"Single cell n=200: serial={t_serial200:.1f}s optimized={t_large:.1f}s "
        f"speedup={t_serial200 / t_large:.2f}x"
    )

    if args.mini_sweep:
        smoke = ROCSweepConfig(
            n_effect=50,
            sigma2_values=[-1.0, -1.5],
            n_values=[10, 80],
            d_values=[0, 1],
            n_reps=3,
            n_experiments=4,
            iter_cap=5000,
            seed_base=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            run_roc_sweeps(smoke, Path(tmp), use_cache=False, n_jobs=n_jobs)
            print(f"Mini SMOKE sweep: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
