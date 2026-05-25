"""Picklable worker entry points for parallel experiment sweeps."""
from __future__ import annotations

import os
from typing import Any


def _pin_blas_threads() -> None:
    for var in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def rep_parallel_workers(n_jobs: int, n_reps: int) -> int:
    """Inner rep-level parallelism that avoids oversubscribing CPUs."""
    cpus = os.cpu_count() or 4
    budget = max(1, cpus - 1)
    if n_jobs <= 1:
        return min(n_reps, budget)
    return max(1, min(n_reps, budget // n_jobs))


def _ensemble_parallel_workers(n_jobs: int, m_ensemble: int) -> int:
    return rep_parallel_workers(n_jobs, m_ensemble)


def _simulate_ensemble_member(payload: dict[str, Any], m: int) -> Any:
    from .sweeps import simulate_graph

    seed = (
        payload["seed_base"]
        + payload["n"] * 1000
        + payload["d_true"] * 100
        + payload["run"] * 10
        + m
    )
    return simulate_graph(
        payload["n"],
        payload["d_true"],
        sigma=payload["sigma_gen"],
        n_iter=payload["n_iter"],
        feature_mode=payload["feature_mode_gen"],
        target_density=payload["target_density"],
        signal=payload["signal"],
        seed=seed,
    )


def sigma_rep_job(payload: dict[str, Any]) -> dict[str, Any]:
    """One replicate: simulate graph and estimate sigma_hat."""
    _pin_blas_threads()
    from .sweeps import estimate_sigma_from_graph, simulate_graph

    d = payload["d"]
    mode_est = "incremental" if d == 0 else payload["feature_mode_est"]
    adj, meta = simulate_graph(
        payload["n"],
        d,
        sigma=payload["sigma_true"],
        n_iter=payload["n_iter"],
        feature_mode=payload["feature_mode_gen"],
        target_density=payload["target_density"],
        signal=payload["signal"],
        seed=payload["seed"],
        return_meta=True,
    )
    sh = estimate_sigma_from_graph(
        adj,
        d,
        feature_mode=mode_est,
        csr_rows=meta.get("csr_rows"),
    )
    return {
        "rep": payload["rep"],
        "sigma_hat": float(sh),
        "density": float(meta["density"]),
        "beta": float(meta["beta"]),
        "feature_mean": float(meta["feature_mean"]),
    }


def aic_run_job(payload: dict[str, Any]) -> dict[str, Any]:
    """One AIC selection run: ensemble graphs + d_hat."""
    _pin_blas_threads()
    from concurrent.futures import ThreadPoolExecutor

    from .sweeps import select_d_ensemble

    m = payload["m_ensemble"]
    n_jobs = int(payload.get("n_jobs", 1))
    ensemble_jobs = int(
        payload.get("ensemble_jobs", _ensemble_parallel_workers(n_jobs, m))
    )

    if ensemble_jobs <= 1 or m <= 1:
        graphs = [_simulate_ensemble_member(payload, idx) for idx in range(m)]
    else:
        with ThreadPoolExecutor(max_workers=ensemble_jobs) as pool:
            graphs = list(pool.map(
                lambda idx: _simulate_ensemble_member(payload, idx),
                range(m),
            ))

    hat_d, aic_stats = select_d_ensemble(
        graphs,
        d_candidates=payload["d_est_values"],
        feature_mode=payload["feature_mode_est"],
        extra_penalty_per_d=payload["aic_penalty_per_d"],
    )
    row: dict[str, Any] = {
        "n": payload["n"],
        "d_true": payload["d_true"],
        "run": payload["run"],
        "hat_d": hat_d,
    }
    for de in payload["d_est_values"]:
        row[f"aic_d{de}"] = aic_stats[de]["aic"]
    return row


def anova_experiment_job(payload: dict[str, Any]) -> float:
    """One ROC Monte Carlo replicate: two sigma_hat groups + ANOVA p-value."""
    _pin_blas_threads()
    from .sweeps import run_anova_pvalue

    return run_anova_pvalue(
        payload["n"],
        payload["d"],
        payload["sigma1"],
        payload["sigma2"],
        n_reps=payload["n_reps"],
        n_iter=payload["n_iter"],
        feature_mode_gen=payload["feature_mode_gen"],
        feature_mode_est=payload["feature_mode_est"],
        target_density=payload["target_density"],
        signal=payload["signal"],
        seed=payload["seed"],
        rep_jobs=payload.get("rep_jobs", 1),
        rep_use_threads=payload.get("rep_use_threads", True),
    )
