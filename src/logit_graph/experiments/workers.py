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
    sh = estimate_sigma_from_graph(adj, d, feature_mode=mode_est)
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
    from .sweeps import select_d_ensemble, simulate_graph

    graphs = []
    for m in range(payload["m_ensemble"]):
        seed = (
            payload["seed_base"]
            + payload["n"] * 1000
            + payload["d_true"] * 100
            + payload["run"] * 10
            + m
        )
        adj = simulate_graph(
            payload["n"],
            payload["d_true"],
            sigma=payload["sigma_gen"],
            n_iter=payload["n_iter"],
            feature_mode=payload["feature_mode_gen"],
            target_density=payload["target_density"],
            signal=payload["signal"],
            seed=seed,
        )
        graphs.append(adj)

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
