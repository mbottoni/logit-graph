#!/usr/bin/env python3
"""Run fast sigma insight experiments (default INSIGHT tier)."""
from __future__ import annotations

import os
from pathlib import Path


def _default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def main() -> None:
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    from logit_graph.experiments import (
        PRESETS,
        flag_sigma_sweep_issues,
        plot_convergence_sigma,
        run_sigma_estimator_ablation,
        run_sigma_sweep,
        summarize_sigma_insights,
    )
    from logit_graph.experiments.sweeps import (
        sigma_sweep_csv_path,
        sigma_sweep_results_json_path,
    )

    OUT = Path(__file__).resolve().parents[2] / "images" / "correction_paper"
    OUT.mkdir(parents=True, exist_ok=True)

    MODE = os.environ.get("LG_EXPERIMENT_MODE", "INSIGHT")
    RUN_ABLATION = os.environ.get("LG_SIGMA_ABLATION", "0") == "1"
    USE_CACHE = os.environ.get("LG_SIGMA_USE_CACHE", "1") == "1"
    N_JOBS = int(os.environ.get("LG_SIGMA_JOBS", _default_jobs()))
    CELL_JOBS = os.environ.get("LG_SIGMA_CELL_JOBS")
    cell_jobs = int(CELL_JOBS) if CELL_JOBS is not None else None

    cfg = PRESETS[MODE]["sigma"]
    if "LG_SIGMA_ITER_CAP" in os.environ:
        cfg.iter_cap = int(os.environ["LG_SIGMA_ITER_CAP"])
    elif os.environ.get("LG_SIGMA_ITER_CAP", "").lower() == "none":
        cfg.iter_cap = None
    if os.environ.get("LG_SIGMA_ADAPTIVE", "").lower() in ("0", "false", "no"):
        cfg.adaptive_stopping = False
    print(
        f"Mode={MODE}, n={cfg.n_values}, reps={cfg.n_reps}, iter_cap={cfg.iter_cap}, "
        f"adaptive={cfg.adaptive_stopping}, cache={USE_CACHE}, jobs={N_JOBS}"
        + (f", cell_jobs={cell_jobs}" if cell_jobs is not None else ", cell_jobs=auto"),
    )

    df = run_sigma_sweep(
        cfg, OUT, use_cache=USE_CACHE, n_jobs=N_JOBS, cell_jobs=cell_jobs,
    )
    plot_convergence_sigma(df, OUT / "convergence_sigma.png")
    print(f"Saved {OUT / 'convergence_sigma.png'}")
    print(f"Data CSV: {sigma_sweep_csv_path(OUT, cfg)}")
    print(f"Metadata: {sigma_sweep_results_json_path(OUT, cfg)}")
    print("Replot later: python notebooks/refactors/run_sigma_replot.py")

    summary = summarize_sigma_insights(df)
    print(summary)
    (OUT / "convergence_sigma_insights.txt").write_text(summary + "\n")

    issues = flag_sigma_sweep_issues(df, target_density=cfg.target_density)
    if len(issues):
        issues_path = OUT / "convergence_sigma_issues.csv"
        issues.to_csv(issues_path, index=False)
        print(f"Flagged {len(issues)} cells -> {issues_path}")

    if RUN_ABLATION:
        ab = run_sigma_estimator_ablation(n=80, n_reps=2, n_iter=15_000)
        ab_path = OUT / "convergence_sigma_ablation_n80.csv"
        ab.to_csv(ab_path, index=False)
        summary_ab = (
            ab.groupby(["d", "sigma_true", "est_mode"])["sigma_error"]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary_ab.to_csv(OUT / "convergence_sigma_ablation_summary.csv", index=False)
        print(f"Saved ablation -> {ab_path}")


if __name__ == "__main__":
    main()
