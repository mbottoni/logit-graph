"""Picklable worker entry points for parallel experiment sweeps."""
from __future__ import annotations

import os
from typing import Any

import numpy as np


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


def _simulate_ensemble_member(payload: dict[str, Any], m: int) -> dict[str, Any]:
    from .sweeps import simulate_graph

    seed = (
        payload["seed_base"]
        + payload["n"] * 1000
        + payload["d_true"] * 100
        + payload["run"] * 10
        + m
    )
    adj, meta = simulate_graph(
        payload["n"],
        payload["d_true"],
        sigma=payload["sigma_gen"],
        n_iter=payload["n_iter"],
        feature_mode=payload["feature_mode_gen"],
        target_density=payload["target_density"],
        signal=payload["signal"],
        seed=seed,
        return_meta=True,
    )
    return {"adj": adj, "csr_rows": meta.get("csr_rows")}


def sigma_rep_job(payload: dict[str, Any]) -> dict[str, Any]:
    """One replicate: simulate graph and estimate sigma_hat."""
    _pin_blas_threads()
    from .sweeps import estimate_sigma_from_graph, simulate_graph

    d = payload["d"]
    mode_est = "incremental" if d == 0 else payload["feature_mode_est"]
    need_adj = d == 0
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
        collect_feature_mean=False,
        materialize_adjacency=need_adj,
        feature_mode_est=mode_est,
        adaptive_stopping=payload.get("adaptive_stopping", False),
        adaptive_check_interval=payload.get("adaptive_check_interval", 20_000),
        adaptive_patience=payload.get("adaptive_patience", 3),
        adaptive_cv_tol=payload.get("adaptive_cv_tol", 0.02),
        adaptive_min_iter=payload.get("adaptive_min_iter", 20_000),
    )
    csr_rows = meta.get("csr_rows")
    sh = estimate_sigma_from_graph(
        adj,
        d,
        feature_mode=mode_est,
        csr_rows=csr_rows,
    )
    return {
        "rep": payload["rep"],
        "sigma_hat": float(sh),
        "density": float(meta["density"]),
        "beta": float(meta["beta"]),
        "feature_mean": 0.0,
        "n_iter_used": float(meta.get("n_iter_used", payload["n_iter"])),
    }


def sigma_cell_job(payload: dict[str, Any]) -> dict[str, Any]:
    """One sweep cell: all replicates, cache npz, return aggregated row."""
    _pin_blas_threads()
    from .sweeps import _aggregate_sigma_cell, _run_sigma_reps

    d = payload["d"]
    sigma_true = payload["sigma_true"]
    n = payload["n"]
    nit = payload["nit"]
    n_reps = payload["n_reps"]
    rep_jobs = int(payload.get("rep_jobs", 1))

    print(
        f"[sigma sweep {payload['cell_idx']}/{payload['total_cells']}] "
        f"d={d} sigma={sigma_true} n={n} iters={nit} reps={n_reps} "
        f"rep_jobs={rep_jobs}"
        + (" adaptive=1" if payload.get("adaptive_stopping") else ""),
        flush=True,
    )

    hats, densities, betas, features = _run_sigma_reps(
        d=d,
        sigma_true=sigma_true,
        n=n,
        n_reps=n_reps,
        nit=nit,
        feature_mode_gen=payload["feature_mode_gen"],
        feature_mode_est=payload["feature_mode_est"],
        target_density=payload["target_density"],
        signal=payload["signal"],
        seed_base=payload["seed_base"],
        n_jobs=rep_jobs,
        adaptive_stopping=payload.get("adaptive_stopping", False),
        adaptive_check_interval=payload.get("adaptive_check_interval", 20_000),
        adaptive_patience=payload.get("adaptive_patience", 3),
        adaptive_cv_tol=payload.get("adaptive_cv_tol", 0.02),
        adaptive_min_iter=payload.get("adaptive_min_iter", 20_000),
    )
    np.savez(
        payload["cell_path"],
        hats=np.asarray(hats, dtype=float),
        densities=np.asarray(densities, dtype=float),
        betas=np.asarray(betas, dtype=float),
        features=np.asarray(features, dtype=float),
    )
    return _aggregate_sigma_cell(
        hats, densities, betas, features,
        d=d, sigma_true=sigma_true, n=n, nit=nit, n_reps=n_reps,
    )


def aic_run_job(payload: dict[str, Any]) -> dict[str, Any]:
    """One AIC selection run: ensemble graphs + d_hat."""
    import time
    _pin_blas_threads()
    from concurrent.futures import ThreadPoolExecutor

    from .sweeps import select_d_ensemble

    t_start = time.time()
    m = payload["m_ensemble"]
    n_jobs = int(payload.get("n_jobs", 1))
    ensemble_jobs = int(
        payload.get("ensemble_jobs", _ensemble_parallel_workers(n_jobs, m))
    )

    if ensemble_jobs <= 1 or m <= 1:
        members = [_simulate_ensemble_member(payload, idx) for idx in range(m)]
    else:
        with ThreadPoolExecutor(max_workers=ensemble_jobs) as pool:
            members = list(pool.map(
                lambda idx: _simulate_ensemble_member(payload, idx),
                range(m),
            ))

    graphs = [mem["adj"] for mem in members]
    csr_rows_list = [mem.get("csr_rows") for mem in members]
    hat_d, aic_stats = select_d_ensemble(
        graphs,
        d_candidates=payload["d_est_values"],
        feature_mode=payload["feature_mode_est"],
        extra_penalty_per_d=payload["aic_penalty_per_d"],
        csr_rows_list=csr_rows_list,
    )

    # Diagnostics: mean density across ensemble + wall time.
    # Useful to detect phase transitions (density blowing up) and to ETA-track.
    n = int(payload["n"])
    densities = []
    for adj in graphs:
        if adj is not None:
            densities.append(float(np.asarray(adj).sum()) / (n * (n - 1)))
    mean_density = float(np.mean(densities)) if densities else 0.0
    elapsed = float(time.time() - t_start)

    row: dict[str, Any] = {
        "n": payload["n"],
        "d_true": payload["d_true"],
        "run": payload["run"],
        "hat_d": hat_d,
        "density": mean_density,
        "elapsed_s": elapsed,
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
        adaptive_stopping=payload.get("adaptive_stopping", False),
        adaptive_check_interval=payload.get("adaptive_check_interval", 20_000),
        adaptive_patience=payload.get("adaptive_patience", 3),
        adaptive_cv_tol=payload.get("adaptive_cv_tol", 0.02),
        adaptive_min_iter=payload.get("adaptive_min_iter", 20_000),
    )


def roc_cell_job(payload: dict[str, Any]) -> dict[str, Any]:
    """One ROC sweep cell: all ANOVA p-values, save npy, return metadata."""
    _pin_blas_threads()
    from pathlib import Path

    from .sweeps import collect_anova_pvalues

    cell_path = Path(payload["cell_path"])
    ckpt_path = cell_path.with_suffix(".partial.npy")
    adaptive = payload.get("adaptive_stopping", False)
    print(
        f"[{payload['label']}] d={payload['d']} n={payload['n']} "
        f"sigma2={payload['sigma2']} iters={payload['n_iter']} "
        f"exps={payload['n_experiments']} exp_jobs={payload['exp_jobs']} "
        f"rep_jobs={payload['rep_jobs']}"
        + (" adaptive=1" if adaptive else ""),
        flush=True,
    )
    pvals = collect_anova_pvalues(
        payload["n"],
        payload["d"],
        payload["sigma1"],
        payload["sigma2"],
        n_reps=payload["n_reps"],
        n_experiments=payload["n_experiments"],
        n_iter=payload["n_iter"],
        feature_mode_gen=payload["feature_mode_gen"],
        feature_mode_est=payload["feature_mode_est"],
        target_density=payload["target_density"],
        signal=payload["signal"],
        seed_base=payload["seed_base"],
        n_jobs=payload["exp_jobs"],
        rep_jobs=payload["rep_jobs"],
        rep_use_threads=payload.get("rep_use_threads", True),
        adaptive_stopping=adaptive,
        adaptive_check_interval=payload.get("adaptive_check_interval", 20_000),
        adaptive_patience=payload.get("adaptive_patience", 3),
        adaptive_cv_tol=payload.get("adaptive_cv_tol", 0.02),
        adaptive_min_iter=payload.get("adaptive_min_iter", 20_000),
        checkpoint_path=ckpt_path,
        checkpoint_every=1,
    )
    np.save(cell_path, pvals)
    if ckpt_path.is_file():
        ckpt_path.unlink()
    return {
        "cell_key": payload["cell_key"],
        "pvals": pvals,
    }
