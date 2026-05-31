"""Cached experiment sweep runners for sigma and AIC-d selection."""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import graph as graph_mod
from ..lg_features import FeatureMode, recommended_iterations
from ..logit_estimator import LogitRegEstimator
from .presets import AICSweepConfig, ROCSweepConfig, SigmaSweepConfig


def _config_hash(cfg: Any) -> str:
    blob = json.dumps(cfg.__dict__, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _sigma_cell_cache_path(
    out_dir: Path,
    cfg_hash: str,
    d: int,
    sigma_true: float,
    n: int,
) -> Path:
    tag = f"d{d}_s{sigma_true:.1f}".replace(".", "p").replace("-", "m")
    return out_dir / f"sigma_hats_{cfg_hash}_{tag}_n{n}.npz"


def _aggregate_sigma_cell(
    hats: list[float],
    densities: list[float],
    betas: list[float],
    features: list[float],
    *,
    d: int,
    sigma_true: float,
    n: int,
    nit: int,
    n_reps: int,
) -> dict[str, Any]:
    arr = np.asarray(hats, dtype=float)
    m = float(np.nanmean(arr))
    se = float(np.nanstd(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "d": d,
        "sigma_true": sigma_true,
        "n": n,
        "sigma_hat_mean": m,
        "sigma_hat_std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "ci_lo": m - 1.96 * se,
        "ci_hi": m + 1.96 * se,
        "n_iter": nit,
        "n_reps": n_reps,
        "density_mean": float(np.mean(densities)),
        "density_std": float(np.std(densities, ddof=1)) if len(densities) > 1 else 0.0,
        "beta_mean": float(np.mean(betas)),
        "feature_mean": float(np.mean(features)),
        "sigma_error": m - sigma_true,
    }


def _run_sigma_reps(
    *,
    d: int,
    sigma_true: float,
    n: int,
    n_reps: int,
    nit: int,
    feature_mode_gen: FeatureMode,
    feature_mode_est: FeatureMode,
    target_density: float,
    signal: float,
    seed_base: int,
    n_jobs: int,
    use_threads: bool = False,
    adaptive_stopping: bool = False,
    adaptive_check_interval: int = 20_000,
    adaptive_patience: int = 3,
    adaptive_cv_tol: float = 0.02,
    adaptive_min_iter: int = 20_000,
) -> tuple[list[float], list[float], list[float], list[float]]:
    from .workers import sigma_rep_job

    jobs = [
        {
            "d": d,
            "sigma_true": sigma_true,
            "n": n,
            "rep": rep,
            "n_iter": nit,
            "feature_mode_gen": feature_mode_gen,
            "feature_mode_est": feature_mode_est,
            "target_density": target_density,
            "signal": signal,
            "seed": seed_base + hash((d, sigma_true, n, rep)) % (2**31 - 1),
            "adaptive_stopping": adaptive_stopping,
            "adaptive_check_interval": adaptive_check_interval,
            "adaptive_patience": adaptive_patience,
            "adaptive_cv_tol": adaptive_cv_tol,
            "adaptive_min_iter": adaptive_min_iter,
        }
        for rep in range(n_reps)
    ]

    if n_jobs <= 1:
        results = [sigma_rep_job(j) for j in jobs]
    elif use_threads:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: list[dict[str, Any]] = [{}] * n_reps
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(sigma_rep_job, payload): payload["rep"]
                for payload in jobs
            }
            for fut in as_completed(futures):
                rep = futures[fut]
                results[rep] = fut.result()
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results = [{}] * n_reps
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(sigma_rep_job, payload): payload["rep"]
                for payload in jobs
            }
            for fut in as_completed(futures):
                rep = futures[fut]
                results[rep] = fut.result()

    hats = [r["sigma_hat"] for r in results]
    densities = [r["density"] for r in results]
    betas = [r["beta"] for r in results]
    features = [r["feature_mean"] for r in results]
    return hats, densities, betas, features


_LARGE_N_REP_PARALLEL_THRESHOLD = 200


def _roc_parallel_plan(
    n: int,
    n_reps: int,
    n_jobs: int,
    rep_jobs: Optional[int] = None,
) -> tuple[int, int]:
    """Choose experiment- vs rep-level parallelism (AIC-style CPU budget)."""
    from .workers import rep_parallel_workers

    if rep_jobs is not None:
        exp_jobs = 1 if n >= _LARGE_N_REP_PARALLEL_THRESHOLD else n_jobs
        return exp_jobs, rep_jobs

    if n >= _LARGE_N_REP_PARALLEL_THRESHOLD:
        return 1, rep_parallel_workers(1, n_reps)

    exp_jobs = n_jobs
    inner = 1 if exp_jobs > 1 else rep_parallel_workers(1, n_reps)
    return exp_jobs, inner


def _sigma_parallel_plan(
    n_reps: int,
    n_jobs: int,
    cell_jobs: Optional[int] = None,
) -> tuple[int, int]:
    """Split CPU budget between cell- and rep-level sigma sweep parallelism."""
    from .workers import rep_parallel_workers

    cpus = max(1, (os.cpu_count() or 2) - 1)
    budget = max(1, min(n_jobs, cpus))

    if cell_jobs is not None:
        outer = max(1, min(int(cell_jobs), budget))
        return outer, rep_parallel_workers(outer, n_reps)

    if budget <= 1:
        return 1, rep_parallel_workers(1, n_reps)

    if n_reps >= 8:
        return 1, min(n_reps, budget)

    outer = max(1, min(budget // 2, 8))
    inner = max(1, min(n_reps, budget // outer))
    return outer, inner


def _iter_count(
    n: int,
    cap: Optional[int],
    sigma_true: Optional[float] = None,
) -> int:
    base = recommended_iterations(n)
    if sigma_true is not None and sigma_true <= -6.0:
        base = int(1.5 * base)
    # Very sparse small graphs (n≈10, σ≪0) need extra Gibbs mixing.
    if sigma_true is not None:
        if n <= 50 and sigma_true <= -6.0:
            base = max(base, 250_000)
        elif n <= 50 and sigma_true <= -4.0:
            base = max(base, 150_000)
        elif n <= 100 and sigma_true <= -6.0:
            base = max(base, 120_000)
    return min(base, cap) if cap is not None else base


def _sigma_adaptive_enabled(
    cfg: SigmaSweepConfig,
    n: int,
    sigma_true: float,
) -> bool:
    """Adaptive early-stop only when chains are long; small sparse cells run full budget."""
    if not cfg.adaptive_stopping:
        return False
    if n <= 100 and sigma_true <= -4.0:
        return False
    return True


def _roc_adaptive_enabled(
    cfg: ROCSweepConfig,
    n: int,
    d: int,
) -> bool:
    """Adaptive early-stop for d>=1; skip d=0 (direct ER) and cheap small-n cells."""
    if not cfg.adaptive_stopping or d == 0:
        return False
    if n < 200:
        return False
    return True


def _roc_adaptive_kwargs(cfg: ROCSweepConfig, n: int, d: int) -> dict[str, Any]:
    return {
        "adaptive_stopping": _roc_adaptive_enabled(cfg, n, d),
        "adaptive_check_interval": cfg.adaptive_check_interval,
        "adaptive_patience": cfg.adaptive_patience,
        "adaptive_cv_tol": cfg.adaptive_cv_tol,
        "adaptive_min_iter": cfg.adaptive_min_iter,
    }


def _roc_intra_cell_plan(
    n: int,
    n_reps: int,
    n_jobs: int,
    rep_jobs: Optional[int],
    *,
    cell_parallel: bool,
) -> tuple[int, int]:
    """Experiment- vs rep-level plan inside one ROC cell."""
    from .workers import rep_parallel_workers

    if cell_parallel:
        inner = rep_jobs if rep_jobs is not None else rep_parallel_workers(1, n_reps)
        return 1, inner
    return _roc_parallel_plan(n, n_reps, n_jobs, rep_jobs)


def _roc_iter_count(
    cfg: ROCSweepConfig,
    n: int,
    sigma_true: Optional[float] = None,
) -> int:
    return _iter_count(n, cfg.iter_cap, sigma_true=sigma_true)


def _aic_iter_count(cfg: AICSweepConfig, n: int) -> int:
    return _iter_count(n, cfg.iter_cap)


def _sigma_iter_count(
    cfg: SigmaSweepConfig,
    n: int,
    sigma_true: Optional[float] = None,
    d: Optional[int] = None,
) -> int:
    """Resolve the iter cap for a single (n, d) cell.

    Precedence: ``cfg.iter_cap_by_d[d]`` (int or {n: int}) overrides the
    global ``cfg.iter_cap`` when ``d`` is supplied and present in the dict.
    """
    cap = cfg.iter_cap
    if d is not None and cfg.iter_cap_by_d and d in cfg.iter_cap_by_d:
        spec = cfg.iter_cap_by_d[d]
        if isinstance(spec, dict):
            if n in spec:
                cap = spec[n]
        else:
            cap = spec
    return _iter_count(n, cap, sigma_true=sigma_true)


def _graph_density(adj: np.ndarray) -> float:
    n = adj.shape[0]
    if n <= 1:
        return 0.0
    return float(np.sum(adj) / (n * (n - 1)))


def _mean_pair_feature(
    adj: np.ndarray,
    d: int,
    feature_mode: FeatureMode,
    seed: int,
) -> float:
    from ..lg_features import build_pair_dataset

    offsets, _ = build_pair_dataset(
        adj, d=d, mode=feature_mode, layer2=True, seed=seed,
    )
    return float(np.mean(offsets))


def _calibrate_beta(
    n: int,
    d: int,
    target_density: float,
    signal: float,
    feature_mode: FeatureMode,
    seed: int,
) -> tuple[float, float]:
    """Return (sigma, beta) for stable generation at target density."""
    from ..lg_features import build_pair_dataset

    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < target_density:
                adj[i, j] = adj[j, i] = 1.0

    if d == 0:
        sigma = math.log(target_density / (1.0 - target_density))
        return sigma, 0.0

    offsets, _ = build_pair_dataset(adj, d=d, mode=feature_mode, layer2=True, seed=seed)
    positive = offsets[offsets > 0]
    scale = max(0.01, float(np.mean(positive)) if len(positive) else float(np.mean(offsets)))
    beta = signal / scale
    sigma = math.log(target_density / (1.0 - target_density)) - signal
    return sigma, beta


def _calibrate_beta_given_sigma(
    n: int,
    d: int,
    sigma: float,
    target_density: float,
    signal: float,
    feature_mode: FeatureMode,
    seed: int,
    *,
    pilot_iters: int = 0,
) -> float:
    """Return beta when sigma is fixed on the paper grid.

    The estimator fixes the feature coefficient at 1 (offset regression), so
    generation must use the same weight for sigma_hat to be consistent.
    """
    del n, d, sigma, target_density, signal, feature_mode, seed, pilot_iters
    return 1.0


def _sample_er_at_sigma(
    n: int,
    sigma: float,
    seed: Optional[int],
) -> np.ndarray:
    """Direct ER sample at p = expit(sigma). Exact equilibrium of d=0 Gibbs with beta=0."""
    from scipy.special import expit

    p = float(expit(sigma))
    rng = np.random.default_rng(seed)
    upper = rng.random((n, n)) < p
    upper = np.triu(upper, k=1)
    adj = (upper | upper.T).astype(float)
    return adj


def _run_adaptive_gibbs(
    fg: Any,
    rng: np.random.Generator,
    max_iter: int,
    *,
    check_interval: int,
    patience: int,
    cv_tol: float,
    min_iter: int,
) -> tuple[list, int]:
    """Run Gibbs in blocks; stop when edge-count CV stabilizes.

    Uses ``fg.edge_count`` (cheap) instead of refitting :math:`\\hat\\sigma`
    on every checkpoint.
    """
    edge_history: list[int] = []
    done = 0
    while done < max_iter:
        chunk = min(check_interval, max_iter - done)
        fg.run_steps(chunk, rng)
        done += chunk
        if done < min_iter:
            continue
        edge_history.append(int(fg.edge_count))
        if len(edge_history) >= patience:
            recent = np.asarray(edge_history[-patience:], dtype=float)
            mean_e = float(np.mean(recent))
            cv = (
                float(np.std(recent, ddof=1) / mean_e)
                if mean_e > 0
                else 0.0
            )
            if cv <= cv_tol:
                break
    return fg.to_rows_list(), done


def simulate_graph(
    n: int,
    d: int,
    sigma: Optional[float] = None,
    *,
    beta: Optional[float] = None,
    n_iter: int,
    feature_mode: FeatureMode = "incremental",
    target_density: float = 0.10,
    signal: float = 0.5,
    seed: Optional[int] = None,
    return_meta: bool = False,
    collect_feature_mean: bool = False,
    materialize_adjacency: bool = True,
    feature_mode_est: Optional[FeatureMode] = None,
    adaptive_stopping: bool = False,
    adaptive_check_interval: int = 20_000,
    adaptive_patience: int = 3,
    adaptive_cv_tol: float = 0.02,
    adaptive_min_iter: int = 20_000,
) -> np.ndarray | tuple[np.ndarray, dict[str, float]]:
    """Generate a graph at fixed sigma.

    d=0 uses a direct ER sample at p = expit(sigma) (exact equilibrium with no
    degree feedback). d>=1 runs Layer-2 Gibbs with feature coefficient fixed
    at beta=1 (matching the paper offset estimator).
    """
    if d == 0:
        if sigma is None:
            sigma = math.log(target_density / (1.0 - target_density))
        adj = _sample_er_at_sigma(n, sigma, seed)
        if not return_meta:
            return adj
        meta = {
            "sigma": float(sigma),
            "beta": 0.0,
            "density": _graph_density(adj),
            "feature_mean": 0.0,
        }
        return adj, meta

    if seed is not None and beta is None:
        if sigma is not None:
            beta = _calibrate_beta_given_sigma(
                n, d, sigma, target_density, signal, feature_mode, seed,
            )
        else:
            sigma_cal, beta_cal = _calibrate_beta(
                n, d, target_density, signal, feature_mode, seed,
            )
            sigma = sigma_cal
            beta = beta_cal

    assert sigma is not None
    use_beta = beta if beta is not None else 1.0
    gen_mode: FeatureMode = feature_mode

    from scipy.special import expit as _expit
    # Warm-start near equilibrium: use expit(sigma) for very sparse cases,
    # but never below a minimum that lets d>=1 features (ball overlaps) be
    # populated enough to inform Gibbs updates.
    er_init = float(np.clip(_expit(sigma), 0.02, 0.5))
    n_iter_used = n_iter

    gm = graph_mod.GraphModel(
        n=n,
        d=d,
        sigma=sigma,
        alpha=1.0,
        beta=float(use_beta),
        er_p=er_init,
        layer2=True,
        feature_mode=gen_mode,
        seed=seed,
    )

    if adaptive_stopping and d >= 1:
        from ..lg_features_fast import FastGibbsGraph

        fg = FastGibbsGraph(
            n,
            d,
            sigma,
            er_p=er_init,
            rng=gm._rng,
            feature_mode=gen_mode,
            alpha=1.0,
            beta=float(use_beta),
            adj=gm.graph,
        )
        csr_rows, n_iter_used = _run_adaptive_gibbs(
            fg,
            gm._rng,
            n_iter,
            check_interval=adaptive_check_interval,
            patience=adaptive_patience,
            cv_tol=adaptive_cv_tol,
            min_iter=adaptive_min_iter,
        )
    else:
        gm.populate_edges_baseline(
            warm_up=0,
            max_iterations=n_iter,
            patience=10,
            check_interval=10**9,
            fast_mode=True,
        )
        csr_rows = getattr(gm, "_csr_rows", None)
    from ..lg_features_fast import adj_from_rows, density_from_rows

    if csr_rows is not None:
        adj = adj_from_rows(csr_rows, n) if materialize_adjacency else None
    else:
        adj = gm.graph.copy()

    if not return_meta:
        if adj is None:
            adj = adj_from_rows(csr_rows, n)
        return adj

    density = (
        density_from_rows(csr_rows, n) if csr_rows is not None else _graph_density(adj)
    )
    if collect_feature_mean:
        feat_seed = seed if seed is not None else 0
        if csr_rows is not None and d >= 1:
            from ..lg_features_fast import build_pair_dataset_from_rows

            offsets, _ = build_pair_dataset_from_rows(
                csr_rows, d, mode=gen_mode,
            )
            feat_mean = float(np.mean(offsets))
        else:
            feat_mean = _mean_pair_feature(adj, d, gen_mode, feat_seed)
    else:
        feat_mean = 0.0

    meta = {
        "sigma": float(sigma),
        "beta": float(use_beta),
        "density": density,
        "feature_mean": feat_mean,
        "n_iter_used": float(n_iter_used),
    }
    if csr_rows is not None:
        meta["csr_rows"] = csr_rows
    return adj, meta


def _ensemble_aic_stats(
    offsets: np.ndarray,
    labels: np.ndarray,
    d_est: int,
    extra_penalty: float,
) -> dict[str, float]:
    from ..offset_logit import aic_from_offset_fit, fit_offset_logit_fast

    sigma_hat, ll = fit_offset_logit_fast(offsets, labels)
    stats = aic_from_offset_fit(sigma_hat, ll, extra_penalty=extra_penalty)
    stats["d_est"] = float(d_est)
    stats["n_obs"] = float(len(labels))
    return stats


def _aic_ensemble(
    graphs: list[np.ndarray],
    d_est: int,
    feature_mode: FeatureMode,
    extra_penalty: float = 0.0,
    *,
    csr_rows_list: Optional[list[Optional[list]]] = None,
) -> dict[str, float]:
    from ..lg_features_fast import build_multi_d_pair_datasets_fast

    all_off: list[np.ndarray] = []
    all_lab: list[np.ndarray] = []
    for idx, g in enumerate(graphs):
        rows = None if csr_rows_list is None else csr_rows_list[idx]
        labels, offsets_by_d = build_multi_d_pair_datasets_fast(
            g, [d_est], mode=feature_mode, rows=rows,
        )
        all_lab.append(labels)
        all_off.append(offsets_by_d[d_est])
    offsets = np.concatenate(all_off)
    labels = np.concatenate(all_lab)
    return _ensemble_aic_stats(offsets, labels, d_est, extra_penalty)


def select_d_ensemble(
    graphs: list[np.ndarray],
    d_candidates: list[int],
    feature_mode: FeatureMode,
    extra_penalty_per_d: float = 0.0,
    *,
    csr_rows_list: Optional[list[Optional[list]]] = None,
) -> tuple[int, dict[int, dict[str, float]]]:
    from ..lg_features_fast import build_multi_d_pair_datasets_fast

    d_sorted = sorted(d_candidates)
    offsets_acc: dict[int, list[np.ndarray]] = {d: [] for d in d_candidates}
    labels_acc: list[np.ndarray] = []

    for idx, g in enumerate(graphs):
        rows = None if csr_rows_list is None else csr_rows_list[idx]
        labels, offsets_by_d = build_multi_d_pair_datasets_fast(
            g, d_sorted, mode=feature_mode, rows=rows,
        )
        labels_acc.append(labels)
        for d in d_candidates:
            offsets_acc[d].append(offsets_by_d[d])

    labels_all = np.concatenate(labels_acc)
    stats = {
        d: _ensemble_aic_stats(
            np.concatenate(offsets_acc[d]),
            labels_all,
            d,
            extra_penalty=extra_penalty_per_d * d,
        )
        for d in d_candidates
    }
    valid = {d: s for d, s in stats.items() if np.isfinite(s["aic"])}
    best = min(valid, key=lambda d: valid[d]["aic"]) if valid else d_candidates[0]
    return best, stats


def estimate_sigma_from_graph(
    adj: Optional[np.ndarray],
    d: int,
    feature_mode: FeatureMode = "incremental",
    beta: float = 1.0,
    *,
    csr_rows: Optional[list] = None,
) -> float:
    del beta
    if csr_rows is not None and d >= 1:
        from ..lg_features_fast import build_pair_dataset_from_rows
        from ..offset_logit import fit_offset_logit_fast

        offsets, labels = build_pair_dataset_from_rows(csr_rows, d, mode=feature_mode)
        sigma_hat, _ = fit_offset_logit_fast(offsets, labels)
        return float(sigma_hat)
    if adj is None:
        raise ValueError("adj required when csr_rows is not provided")
    est = LogitRegEstimator(adj, d=d, layer2=True, feature_mode=feature_mode)
    stats = est.compute_aic(d_est=d, feature_mode=feature_mode)
    return float(stats["sigma_hat"])


def run_anova_pvalue(
    n: int,
    d: int,
    sigma1: float,
    sigma2: float,
    *,
    n_reps: int,
    n_iter: int,
    feature_mode_gen: FeatureMode,
    feature_mode_est: FeatureMode,
    target_density: float,
    signal: float,
    seed: int,
    rep_jobs: int = 1,
    rep_use_threads: bool = True,
    adaptive_stopping: bool = False,
    adaptive_check_interval: int = 20_000,
    adaptive_patience: int = 3,
    adaptive_cv_tol: float = 0.02,
    adaptive_min_iter: int = 20_000,
) -> float:
    """One Monte Carlo replicate: two groups of sigma_hat, one-way ANOVA p-value."""
    from concurrent.futures import ThreadPoolExecutor
    from scipy import stats

    common = dict(
        d=d,
        n=n,
        nit=n_iter,
        feature_mode_gen=feature_mode_gen,
        feature_mode_est=feature_mode_est,
        target_density=target_density,
        signal=signal,
        n_reps=n_reps,
        n_jobs=rep_jobs,
        use_threads=rep_use_threads,
        adaptive_stopping=adaptive_stopping,
        adaptive_check_interval=adaptive_check_interval,
        adaptive_patience=adaptive_patience,
        adaptive_cv_tol=adaptive_cv_tol,
        adaptive_min_iter=adaptive_min_iter,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut1 = pool.submit(
            _run_sigma_reps, sigma_true=sigma1, seed_base=seed, **common,
        )
        fut2 = pool.submit(
            _run_sigma_reps, sigma_true=sigma2, seed_base=seed + 10_000, **common,
        )
        g1, _, _, _ = fut1.result()
        g2, _, _, _ = fut2.result()
    _, p_val = stats.f_oneway(g1, g2)
    return float(p_val)


def _anova_experiment_seed(
    n: int,
    d: int,
    sigma1: float,
    sigma2: float,
    exp: int,
    seed_base: int,
) -> int:
    return seed_base + hash((n, d, sigma1, sigma2, exp)) % (2**31 - 1)


def collect_anova_pvalues(
    n: int,
    d: int,
    sigma1: float,
    sigma2: float,
    *,
    n_reps: int,
    n_experiments: int,
    n_iter: int,
    feature_mode_gen: FeatureMode,
    feature_mode_est: FeatureMode,
    target_density: float,
    signal: float,
    seed_base: int,
    n_jobs: int = 1,
    rep_jobs: Optional[int] = None,
    rep_use_threads: bool = True,
    adaptive_stopping: bool = False,
    adaptive_check_interval: int = 20_000,
    adaptive_patience: int = 3,
    adaptive_cv_tol: float = 0.02,
    adaptive_min_iter: int = 20_000,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 1,
) -> np.ndarray:
    from .workers import anova_experiment_job

    exp_jobs, inner_rep_jobs = _roc_parallel_plan(n, n_reps, n_jobs, rep_jobs)
    if inner_rep_jobs > 1 or exp_jobs != n_jobs:
        print(
            f"    parallel plan: exp_jobs={exp_jobs} rep_jobs={inner_rep_jobs} "
            f"threads={rep_use_threads}",
            flush=True,
        )

    start = 0
    p_values = np.empty(n_experiments, dtype=float)
    if checkpoint_path is not None and checkpoint_path.is_file():
        saved = np.load(checkpoint_path)
        n_done = min(len(saved), n_experiments)
        p_values[:n_done] = saved[:n_done]
        start = n_done
        if start:
            print(f"    resume from {start}/{n_experiments}", flush=True)

    jobs = [
        {
            "n": n,
            "d": d,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "n_reps": n_reps,
            "n_iter": n_iter,
            "feature_mode_gen": feature_mode_gen,
            "feature_mode_est": feature_mode_est,
            "target_density": target_density,
            "signal": signal,
            "seed": _anova_experiment_seed(n, d, sigma1, sigma2, exp, seed_base),
            "rep_jobs": inner_rep_jobs,
            "rep_use_threads": rep_use_threads,
            "adaptive_stopping": adaptive_stopping,
            "adaptive_check_interval": adaptive_check_interval,
            "adaptive_patience": adaptive_patience,
            "adaptive_cv_tol": adaptive_cv_tol,
            "adaptive_min_iter": adaptive_min_iter,
        }
        for exp in range(start, n_experiments)
    ]

    if not jobs:
        return p_values

    if exp_jobs <= 1:
        for i, payload in enumerate(jobs, start=start):
            p_values[i] = anova_experiment_job(payload)
            done = i + 1
            if checkpoint_path is not None and done % checkpoint_every == 0:
                np.save(checkpoint_path, p_values[:done])
            if done % max(1, n_experiments // 10) == 0:
                print(f"    exp {done}/{n_experiments}", flush=True)
        if checkpoint_path is not None:
            np.save(checkpoint_path, p_values)
        return p_values

    from concurrent.futures import ProcessPoolExecutor, as_completed

    done = start
    with ProcessPoolExecutor(max_workers=exp_jobs) as pool:
        futures = {
            pool.submit(anova_experiment_job, payload): start + offset
            for offset, payload in enumerate(jobs)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            p_values[idx] = fut.result()
            done += 1
            if checkpoint_path is not None and done % checkpoint_every == 0:
                np.save(checkpoint_path, p_values[:done])
            if done % max(1, n_experiments // 10) == 0:
                print(f"    exp {done}/{n_experiments}", flush=True)
    if checkpoint_path is not None:
        np.save(checkpoint_path, p_values)
    return p_values


def compute_roc_curve(
    p_values: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)
    rates = np.array([float(np.mean(p_values < t)) for t in thresholds], dtype=float)
    return thresholds, rates


def _roc_cell_pvalues_path(
    out_dir: Path,
    cfg_hash: str,
    sweep: str,
    d: int,
    *,
    sigma2: Optional[float] = None,
    n: Optional[int] = None,
) -> Path:
    if sweep == "effect":
        tag = f"effect_d{d}_s2{sigma2:.1f}".replace(".", "p").replace("-", "m")
    else:
        tag = f"sample_d{d}_n{n}"
    return out_dir / f"roc_pvalues_{cfg_hash}_{tag}.npy"


def _roc_rows_from_pvalues(
    pvals: np.ndarray,
    *,
    sweep: str,
    d: int,
    sigma1: float,
    sigma2: float,
    n: int,
    n_reps: int,
    n_experiments: int,
) -> list[dict[str, Any]]:
    thresh, rates = compute_roc_curve(pvals)
    power = float(np.mean(pvals < 0.05))
    return [
        {
            "sweep": sweep,
            "d": d,
            "n": n,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "alpha": t,
            "rejection_rate": r,
            "power_at_005": power,
            "n_reps": n_reps,
            "n_experiments": n_experiments,
        }
        for t, r in zip(thresh, rates)
    ]


def _roc_all_cells_complete(cfg: ROCSweepConfig, out_dir: Path, cfg_hash: str) -> bool:
    for d in cfg.d_values:
        for sigma2 in cfg.sigma2_values:
            if not _roc_cell_pvalues_path(
                out_dir, cfg_hash, "effect", d, sigma2=sigma2,
            ).is_file():
                return False
    for d in cfg.d_values:
        for n in cfg.n_values:
            if not _roc_cell_pvalues_path(
                out_dir, cfg_hash, "sample", d, n=n,
            ).is_file():
                return False
    return True


def _load_rows_from_cell_caches(
    cfg: ROCSweepConfig,
    out_dir: Path,
    cfg_hash: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in cfg.d_values:
        for sigma2 in cfg.sigma2_values:
            path = _roc_cell_pvalues_path(out_dir, cfg_hash, "effect", d, sigma2=sigma2)
            if path.is_file():
                pvals = np.load(path)
                rows.extend(_roc_rows_from_pvalues(
                    pvals,
                    sweep="effect",
                    d=d,
                    sigma1=cfg.sigma1,
                    sigma2=sigma2,
                    n=cfg.n_effect,
                    n_reps=cfg.n_reps,
                    n_experiments=cfg.n_experiments,
                ))
    for d in cfg.d_values:
        for n in cfg.n_values:
            path = _roc_cell_pvalues_path(out_dir, cfg_hash, "sample", d, n=n)
            if path.is_file():
                pvals = np.load(path)
                rows.extend(_roc_rows_from_pvalues(
                    pvals,
                    sweep="sample",
                    d=d,
                    sigma1=cfg.sigma1,
                    sigma2=cfg.sigma2_fixed,
                    n=n,
                    n_reps=cfg.n_reps,
                    n_experiments=cfg.n_experiments,
                ))
    return rows


def _roc_cell_key(sweep: str, d: int, *, sigma2: Optional[float] = None, n: Optional[int] = None) -> str:
    if sweep == "effect":
        return f"effect:d{d}:s2{sigma2}"
    return f"sample:d{d}:n{n}"


def _run_roc_pending_cells(
    pending: list[dict[str, Any]],
    *,
    outer_jobs: int,
    results: dict[str, np.ndarray],
    cache_path: Path,
    rows: list[dict[str, Any]],
    cfg: ROCSweepConfig,
) -> None:
    from .workers import roc_cell_job

    if not pending:
        return

    def _merge_cell(payload: dict[str, Any], pvals: np.ndarray) -> None:
        key = payload["cell_key"]
        results[key] = pvals
        sweep = payload["sweep"]
        d = payload["d"]
        n = payload["n"]
        sigma2 = payload["sigma2"]
        rows[:] = [
            r for r in rows
            if not (
                (sweep == "sample" and r["sweep"] == "sample" and r["d"] == d and r["n"] == n)
                or (sweep == "effect" and r["sweep"] == "effect" and r["d"] == d and r["sigma2"] == sigma2)
            )
        ]
        rows.extend(_roc_rows_from_pvalues(
            pvals,
            sweep=sweep,
            d=d,
            sigma1=cfg.sigma1,
            sigma2=sigma2,
            n=n,
            n_reps=cfg.n_reps,
            n_experiments=cfg.n_experiments,
        ))
        pd.DataFrame(rows).to_csv(cache_path, index=False)

    if outer_jobs <= 1 or len(pending) <= 1:
        for payload in pending:
            out = roc_cell_job(payload)
            _merge_cell(payload, out["pvals"])
        return

    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(
        f"[roc sweep] cell parallel: {len(pending)} cells, cell_jobs={outer_jobs}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=outer_jobs) as pool:
        futures = {pool.submit(roc_cell_job, payload): payload for payload in pending}
        for fut in as_completed(futures):
            payload = futures[fut]
            out = fut.result()
            _merge_cell(payload, out["pvals"])


def run_roc_sweeps(
    cfg: ROCSweepConfig,
    out_dir: Path,
    *,
    use_cache: bool = True,
    n_jobs: int = 1,
    rep_jobs: Optional[int] = None,
    rep_use_threads: bool = True,
    cell_jobs: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run effect-size and sample-size ROC sweeps; return long-form DataFrames."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = _config_hash(cfg)
    cache_path = out_dir / f"roc_sweeps_{cfg_hash}.csv"

    if use_cache and cache_path.is_file() and _roc_all_cells_complete(cfg, out_dir, cfg_hash):
        df = pd.read_csv(cache_path)
        effect = df[df.sweep == "effect"].copy()
        sample = df[df.sweep == "sample"].copy()
        return effect, sample

    outer_jobs, _ = _sigma_parallel_plan(cfg.n_reps, n_jobs, cell_jobs)
    cell_parallel = outer_jobs > 1

    rows: list[dict[str, Any]] = _load_rows_from_cell_caches(cfg, out_dir, cfg_hash)
    pvals_by_key: dict[str, np.ndarray] = {}
    pending: list[dict[str, Any]] = []
    gen_mode: FeatureMode = cfg.feature_mode_gen
    est_mode: FeatureMode = cfg.feature_mode_est

    total_sample = len(cfg.d_values) * len(cfg.n_values)
    cell = 0
    for d in cfg.d_values:
        for n in sorted(cfg.n_values):
            cell += 1
            cell_path = _roc_cell_pvalues_path(out_dir, cfg_hash, "sample", d, n=n)
            key = _roc_cell_key("sample", d, n=n)
            nit = _roc_iter_count(cfg, n, sigma_true=cfg.sigma1)
            if use_cache and cell_path.is_file():
                print(
                    f"[roc sample {cell}/{total_sample}] cache hit d={d} n={n}",
                    flush=True,
                )
                pvals_by_key[key] = np.load(cell_path)
            else:
                exp_jobs, inner_rep = _roc_intra_cell_plan(
                    n, cfg.n_reps, n_jobs, rep_jobs, cell_parallel=cell_parallel,
                )
                pending.append({
                    "cell_key": key,
                    "label": f"roc sample {cell}/{total_sample}",
                    "sweep": "sample",
                    "d": d,
                    "n": n,
                    "sigma1": cfg.sigma1,
                    "sigma2": cfg.sigma2_fixed,
                    "n_reps": cfg.n_reps,
                    "n_experiments": cfg.n_experiments,
                    "n_iter": nit,
                    "feature_mode_gen": gen_mode,
                    "feature_mode_est": est_mode,
                    "target_density": cfg.target_density,
                    "signal": cfg.signal,
                    "seed_base": cfg.seed_base + 5000 + d * 1000 + n,
                    "cell_path": str(cell_path),
                    "exp_jobs": exp_jobs,
                    "rep_jobs": inner_rep,
                    "rep_use_threads": rep_use_threads,
                    **_roc_adaptive_kwargs(cfg, n, d),
                })

    total_effect = len(cfg.d_values) * len(cfg.sigma2_values)
    cell = 0
    for d in cfg.d_values:
        nit = _roc_iter_count(cfg, cfg.n_effect, sigma_true=cfg.sigma1)
        for sigma2 in cfg.sigma2_values:
            cell += 1
            cell_path = _roc_cell_pvalues_path(
                out_dir, cfg_hash, "effect", d, sigma2=sigma2,
            )
            key = _roc_cell_key("effect", d, sigma2=sigma2)
            if use_cache and cell_path.is_file():
                print(
                    f"[roc effect {cell}/{total_effect}] cache hit d={d} sigma2={sigma2}",
                    flush=True,
                )
                pvals_by_key[key] = np.load(cell_path)
            else:
                exp_jobs, inner_rep = _roc_intra_cell_plan(
                    cfg.n_effect, cfg.n_reps, n_jobs, rep_jobs,
                    cell_parallel=cell_parallel,
                )
                pending.append({
                    "cell_key": key,
                    "label": f"roc effect {cell}/{total_effect}",
                    "sweep": "effect",
                    "d": d,
                    "n": cfg.n_effect,
                    "sigma1": cfg.sigma1,
                    "sigma2": sigma2,
                    "n_reps": cfg.n_reps,
                    "n_experiments": cfg.n_experiments,
                    "n_iter": nit,
                    "feature_mode_gen": gen_mode,
                    "feature_mode_est": est_mode,
                    "target_density": cfg.target_density,
                    "signal": cfg.signal,
                    "seed_base": cfg.seed_base + d * 100 + int(10 * sigma2),
                    "cell_path": str(cell_path),
                    "exp_jobs": exp_jobs,
                    "rep_jobs": inner_rep,
                    "rep_use_threads": rep_use_threads,
                    **_roc_adaptive_kwargs(cfg, cfg.n_effect, d),
                })

    _run_roc_pending_cells(
        pending,
        outer_jobs=outer_jobs,
        results=pvals_by_key,
        cache_path=cache_path,
        rows=rows,
        cfg=cfg,
    )

    if not rows:
        for d in cfg.d_values:
            for sigma2 in cfg.sigma2_values:
                key = _roc_cell_key("effect", d, sigma2=sigma2)
                path = _roc_cell_pvalues_path(out_dir, cfg_hash, "effect", d, sigma2=sigma2)
                pvals = pvals_by_key.get(key)
                if pvals is None and path.is_file():
                    pvals = np.load(path)
                if pvals is not None:
                    rows.extend(_roc_rows_from_pvalues(
                        pvals, sweep="effect", d=d,
                        sigma1=cfg.sigma1, sigma2=sigma2, n=cfg.n_effect,
                        n_reps=cfg.n_reps, n_experiments=cfg.n_experiments,
                    ))
        for d in cfg.d_values:
            for n in cfg.n_values:
                key = _roc_cell_key("sample", d, n=n)
                path = _roc_cell_pvalues_path(out_dir, cfg_hash, "sample", d, n=n)
                pvals = pvals_by_key.get(key)
                if pvals is None and path.is_file():
                    pvals = np.load(path)
                if pvals is not None:
                    rows.extend(_roc_rows_from_pvalues(
                        pvals, sweep="sample", d=d,
                        sigma1=cfg.sigma1, sigma2=cfg.sigma2_fixed, n=n,
                        n_reps=cfg.n_reps, n_experiments=cfg.n_experiments,
                    ))

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    effect = df[df.sweep == "effect"].copy()
    sample = df[df.sweep == "sample"].copy()
    return effect, sample


def plot_roc_effect_size(
    df: pd.DataFrame,
    out_path: Path,
    *,
    sigma1: float = -1.0,
    n_fixed: int = 500,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    palette = {
        -1.0: ("#888888", "--"),
        -1.5: ("#0072B2", "-"),
        -2.0: ("#E69F00", "-."),
        -2.5: ("#CC79A7", ":"),
    }
    sub = df[df.sweep == "effect"]
    if sub.empty:
        return
    d_values = sorted(sub["d"].unique())
    sigma2_values = sorted(sub["sigma2"].unique())

    fig, axes = plt.subplots(1, len(d_values), figsize=(7.3 * len(d_values), 8.5), sharey=True)
    if len(d_values) == 1:
        axes = [axes]

    for ax, d in zip(axes, d_values):
        ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle=":", linewidth=1.4, zorder=1)
        for sigma2 in sigma2_values:
            curve = sub[(sub.d == d) & (sub.sigma2 == sigma2)].sort_values("alpha")
            color, ls = palette.get(float(sigma2), ("#333333", "-"))
            lw = 1.8 if sigma2 == sigma1 else 2.5
            ax.plot(curve.alpha, curve.rejection_rate, color=color, linestyle=ls, linewidth=lw)
        ax.set_xlabel(r"Significance level $\alpha$")
        ax.set_title(f"$d = {int(d)}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Rejection rate (power)")

    handles = []
    for sigma2 in sigma2_values:
        color, ls = palette.get(float(sigma2), ("#333333", "-"))
        lw = 1.8 if sigma2 == sigma1 else 2.5
        diff = abs(sigma2 - sigma1)
        lbl = (
            f"$\\sigma_2 = {sigma2}$  ($|\\Delta| = {diff:.1f}$)"
            if diff > 0
            else f"$\\sigma_2 = \\sigma_1 = {sigma2}$ (null)"
        )
        handles.append(Line2D([], [], color=color, linestyle=ls, linewidth=lw, label=lbl))
    handles.append(Line2D([], [], color="#AAAAAA", linestyle=":", linewidth=1.4,
                            label="Diagonal (no power)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True,
               bbox_to_anchor=(0.5, 0.02), fontsize=14)
    fig.suptitle(
        rf"ROC: effect size ($n = {n_fixed}$, $\sigma_1 = {sigma1}$)",
        fontsize=22, y=0.98,
    )
    fig.subplots_adjust(wspace=0.08, bottom=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_sample_size(
    df: pd.DataFrame,
    out_path: Path,
    *,
    sigma1: float = -1.0,
    sigma2: float = -1.5,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    palette = {
        10: ("#888888", "--"),
        100: ("#0072B2", "-"),
        500: ("#E69F00", "-."),
        1000: ("#009E73", ":"),
        2000: ("#CC79A7", (0, (4, 2, 1, 2))),
    }
    sub = df[df.sweep == "sample"]
    if sub.empty:
        return
    d_values = sorted(sub["d"].unique())
    n_values = sorted(sub["n"].unique())

    fig, axes = plt.subplots(1, len(d_values), figsize=(7.3 * len(d_values), 8.5), sharey=True)
    if len(d_values) == 1:
        axes = [axes]

    for ax, d in zip(axes, d_values):
        ax.plot([0, 1], [0, 1], color="#AAAAAA", linestyle=":", linewidth=1.4, zorder=1)
        for n in n_values:
            curve = sub[(sub.d == d) & (sub.n == n)].sort_values("alpha")
            color, ls = palette.get(int(n), ("#333333", "-"))
            ax.plot(curve.alpha, curve.rejection_rate, color=color, linestyle=ls, linewidth=2.5)
        ax.set_xlabel(r"Significance level $\alpha$")
        ax.set_title(f"$d = {int(d)}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Rejection rate (power)")

    handles = [
        Line2D([], [], color=palette.get(int(n), ("#333333", "-"))[0],
               linestyle=palette.get(int(n), ("#333333", "-"))[1],
               linewidth=2.5, label=f"$n = {int(n)}$")
        for n in n_values
    ]
    handles.append(Line2D([], [], color="#AAAAAA", linestyle=":", linewidth=1.4,
                            label="Diagonal (no power)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=True, bbox_to_anchor=(0.5, 0.01), fontsize=14)
    fig.suptitle(
        rf"ROC: sample size ($\sigma_1 = {sigma1}$, $\sigma_2 = {sigma2}$)",
        fontsize=22, y=0.98,
    )
    fig.subplots_adjust(wspace=0.08, bottom=0.20)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _sigma_adaptive_kwargs(
    cfg: SigmaSweepConfig,
    n: int,
    sigma_true: float,
) -> dict[str, Any]:
    enabled = _sigma_adaptive_enabled(cfg, n, sigma_true)
    return {
        "adaptive_stopping": enabled,
        "adaptive_check_interval": cfg.adaptive_check_interval,
        "adaptive_patience": cfg.adaptive_patience,
        "adaptive_cv_tol": cfg.adaptive_cv_tol,
        "adaptive_min_iter": cfg.adaptive_min_iter,
    }


def _load_sigma_cell_record(
    cell_path: Path,
    *,
    d: int,
    sigma_true: float,
    n: int,
    nit: int,
    n_reps: int,
) -> dict[str, Any]:
    data = np.load(cell_path)
    hats = data["hats"].tolist()
    densities = data["densities"].tolist()
    betas = data["betas"].tolist()
    features = data["features"].tolist()
    return _aggregate_sigma_cell(
        hats, densities, betas, features,
        d=d, sigma_true=sigma_true, n=n, nit=nit, n_reps=n_reps,
    )


def run_sigma_sweep(
    cfg: SigmaSweepConfig,
    out_dir: Path,
    *,
    use_cache: bool = True,
    n_jobs: int = 1,
    cell_jobs: Optional[int] = None,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = _config_hash(cfg)
    cache_path = sigma_sweep_csv_path(out_dir, cfg)
    expected_cells = len(cfg.d_values) * len(cfg.sigma_values) * len(cfg.n_values)

    if use_cache and cache_path.is_file():
        cached = pd.read_csv(cache_path)
        if len(cached) >= expected_cells:
            return cached

    outer_jobs, rep_jobs = _sigma_parallel_plan(cfg.n_reps, n_jobs, cell_jobs)

    pending: list[dict[str, Any]] = []
    records_by_idx: dict[int, dict[str, Any]] = {}
    total_cells = expected_cells
    cell_idx = 0

    for d in cfg.d_values:
        for sigma_true in cfg.sigma_values:
            for n in cfg.n_values:
                cell_idx += 1
                nit = _sigma_iter_count(cfg, n, sigma_true=sigma_true, d=d)
                cell_path = _sigma_cell_cache_path(out_dir, cfg_hash, d, sigma_true, n)

                if use_cache and cell_path.is_file():
                    print(
                        f"[sigma sweep {cell_idx}/{total_cells}] cache hit "
                        f"d={d} sigma={sigma_true} n={n}",
                        flush=True,
                    )
                    records_by_idx[cell_idx] = _load_sigma_cell_record(
                        cell_path,
                        d=d, sigma_true=sigma_true, n=n, nit=nit, n_reps=cfg.n_reps,
                    )
                else:
                    pending.append({
                        "cell_idx": cell_idx,
                        "total_cells": total_cells,
                        "d": d,
                        "sigma_true": sigma_true,
                        "n": n,
                        "nit": nit,
                        "n_reps": cfg.n_reps,
                        "cell_path": str(cell_path),
                        "seed_base": cfg.seed_base,
                        "feature_mode_gen": cfg.feature_mode_gen,
                        "feature_mode_est": cfg.feature_mode_est,
                        "target_density": cfg.target_density,
                        "signal": cfg.signal,
                        "rep_jobs": rep_jobs,
                        **_sigma_adaptive_kwargs(cfg, n, sigma_true),
                    })

    if pending:
        from .workers import sigma_cell_job

        if outer_jobs <= 1 or len(pending) <= 1:
            for payload in pending:
                records_by_idx[payload["cell_idx"]] = sigma_cell_job(payload)
                interim = [records_by_idx[i] for i in sorted(records_by_idx)]
                pd.DataFrame(interim).to_csv(cache_path, index=False)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            print(
                f"[sigma sweep] cell parallel: {len(pending)} cells, "
                f"cell_jobs={outer_jobs} rep_jobs={rep_jobs}",
                flush=True,
            )
            with ProcessPoolExecutor(max_workers=outer_jobs) as pool:
                futures = {
                    pool.submit(sigma_cell_job, payload): payload["cell_idx"]
                    for payload in pending
                }
                for fut in as_completed(futures):
                    idx = futures[fut]
                    records_by_idx[idx] = fut.result()
                    interim = [records_by_idx[i] for i in sorted(records_by_idx)]
                    pd.DataFrame(interim).to_csv(cache_path, index=False)

    records = [records_by_idx[i] for i in sorted(records_by_idx)]
    df = pd.DataFrame(records)
    save_sigma_sweep_artifacts(cfg, out_dir, df)
    return df


def run_sigma_estimator_ablation(
    n: int = 100,
    *,
    d_values: Optional[list[int]] = None,
    sigma_values: Optional[list[float]] = None,
    n_reps: int = 4,
    n_iter: Optional[int] = None,
    target_density: float = 0.10,
    signal: float = 0.5,
    seed_base: int = 5000,
) -> pd.DataFrame:
    """Compare sigma_hat under bounded vs incremental estimation on the same graphs."""
    if d_values is None:
        d_values = [0, 1, 2]
    if sigma_values is None:
        sigma_values = [-2.0, -4.0, -6.0, -8.0]
    est_modes: list[FeatureMode] = ["bounded", "incremental"]

    rows: list[dict[str, Any]] = []
    for d in d_values:
        for sigma_true in sigma_values:
            nit = n_iter if n_iter is not None else _iter_count(n, 80_000, sigma_true=sigma_true)
            for rep in range(n_reps):
                seed = seed_base + hash((d, sigma_true, n, rep)) % (2**31 - 1)
                adj, _ = simulate_graph(
                    n, d, sigma=sigma_true,
                    n_iter=nit,
                    feature_mode="incremental",
                    target_density=target_density,
                    signal=signal,
                    seed=seed,
                    return_meta=True,
                )
                for mode in est_modes:
                    sh = estimate_sigma_from_graph(adj, d, feature_mode=mode)
                    rows.append({
                        "n": n,
                        "d": d,
                        "sigma_true": sigma_true,
                        "rep": rep,
                        "est_mode": mode,
                        "sigma_hat": sh,
                        "sigma_error": sh - sigma_true,
                        "density": _graph_density(adj),
                    })
    return pd.DataFrame(rows)


def flag_sigma_sweep_issues(
    df: pd.DataFrame,
    *,
    error_threshold: float = 0.5,
    density_drift: float = 0.05,
    target_density: float = 0.10,
) -> pd.DataFrame:
    """Return rows where |sigma_error| or density drift exceeds thresholds."""
    issues = df[
        (df["sigma_error"].abs() > error_threshold)
        | (df["density_mean"].sub(target_density).abs() > density_drift)
    ].copy()
    return issues.sort_values(["d", "sigma_true", "n"])


def summarize_aic_insights(
    conf: dict[int, dict[int, dict[int, int]]],
    d_values: list[int],
) -> str:
    """Human-readable confusion summary for AIC d selection."""
    lines = ["AIC d-selection confusion (rows=d_true, cols=d_hat):"]
    for n, mat in sorted(conf.items()):
        lines.append(f"  n={n}")
        header = "      " + "  ".join(f"d̂={d:>2}" for d in d_values) + "   acc"
        lines.append(header)
        diag_total = 0
        diag_correct = 0
        for dt in d_values:
            row = mat[dt]
            total = sum(row.values())
            if total == 0:
                continue
            on_diag = row.get(dt, 0)
            diag_total += total
            diag_correct += on_diag
            cells = "  ".join(f"{row.get(de, 0):>4d}" for de in d_values)
            acc = 100.0 * on_diag / total
            lines.append(f"  d_true={dt}:  {cells}   {acc:3.0f}%")
        overall = 100.0 * diag_correct / max(1, diag_total)
        lines.append(f"  overall accuracy: {overall:.0f}%")
    return "\n".join(lines)


def summarize_sigma_insights(
    df: pd.DataFrame,
    *,
    good_threshold: float = 0.3,
    warn_threshold: float = 0.6,
) -> str:
    """Human-readable summary for fast insight runs."""
    lines = ["σ recovery summary (σ_hat − σ_true):"]
    n_max = int(df["n"].max())
    at_max = df[df["n"] == n_max]
    good = int((at_max["sigma_error"].abs() <= good_threshold).sum())
    total = len(at_max)
    lines.append(f"  At n={n_max}: {good}/{total} cells with |error| ≤ {good_threshold}")

    for d in sorted(df["d"].unique()):
        lines.append(f"  d={int(d)}:")
        sub = df[df["d"] == d].sort_values(["sigma_true", "n"])
        for sigma in sorted(sub["sigma_true"].unique()):
            ssub = sub[sub["sigma_true"] == sigma]
            parts = [
                f"n={int(r.n)} {r.sigma_error:+.2f}"
                for _, r in ssub.iterrows()
            ]
            err_at_max = float(ssub.loc[ssub["n"] == n_max, "sigma_error"].iloc[0])
            tag = "ok" if abs(err_at_max) <= good_threshold else (
                "warn" if abs(err_at_max) <= warn_threshold else "bad"
            )
            lines.append(f"    σ={int(sigma):>3} [{tag}]: " + ", ".join(parts))

    issues = flag_sigma_sweep_issues(df)
    if len(issues):
        lines.append(f"  Flagged cells: {len(issues)} (see convergence_sigma_issues.csv)")
    return "\n".join(lines)


def sigma_sweep_csv_path(out_dir: Path, cfg: SigmaSweepConfig) -> Path:
    """Aggregated mean/CI table for a sigma consistency sweep."""
    return Path(out_dir) / f"convergence_sigma_{_config_hash(cfg)}.csv"


def sigma_sweep_results_json_path(out_dir: Path, cfg: SigmaSweepConfig) -> Path:
    """Config metadata + output paths for offline replot."""
    return Path(out_dir) / f"convergence_sigma_{_config_hash(cfg)}_results.json"


def load_sigma_sweep_df(
    cfg: SigmaSweepConfig,
    out_dir: Path,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load aggregated sweep CSV, rebuilding from per-cell npz caches if needed."""
    out_dir = Path(out_dir)
    cfg_hash = _config_hash(cfg)
    cache_path = sigma_sweep_csv_path(out_dir, cfg)
    if use_cache and cache_path.is_file():
        return pd.read_csv(cache_path)

    records: list[dict[str, Any]] = []
    for d in cfg.d_values:
        for sigma_true in cfg.sigma_values:
            for n in cfg.n_values:
                cell_path = _sigma_cell_cache_path(out_dir, cfg_hash, d, sigma_true, n)
                if not cell_path.is_file():
                    continue
                nit = _sigma_iter_count(cfg, n, sigma_true=sigma_true, d=d)
                data = np.load(cell_path)
                records.append(_aggregate_sigma_cell(
                    data["hats"].tolist(),
                    data["densities"].tolist(),
                    data["betas"].tolist(),
                    data["features"].tolist(),
                    d=d,
                    sigma_true=sigma_true,
                    n=n,
                    nit=nit,
                    n_reps=cfg.n_reps,
                ))
    if not records:
        raise FileNotFoundError(
            f"No sigma sweep data for hash {cfg_hash} under {out_dir}",
        )
    return pd.DataFrame(records)


def save_sigma_sweep_artifacts(
    cfg: SigmaSweepConfig,
    out_dir: Path,
    df: pd.DataFrame,
) -> None:
    """Write CSV + metadata JSON for replotting without re-simulation."""
    out_dir = Path(out_dir)
    csv_path = sigma_sweep_csv_path(out_dir, cfg)
    df.to_csv(csv_path, index=False)
    df.to_csv(out_dir / "convergence_sigma.csv", index=False)
    meta = {
        "config_hash": _config_hash(cfg),
        "csv_path": str(csv_path),
        "n_cells": len(df),
        "n_cells_expected": (
            len(cfg.d_values) * len(cfg.sigma_values) * len(cfg.n_values)
        ),
        "sigma_values": cfg.sigma_values,
        "d_values": cfg.d_values,
        "n_values": cfg.n_values,
        "n_reps": cfg.n_reps,
        "iter_cap": cfg.iter_cap,
        "adaptive_stopping": cfg.adaptive_stopping,
        "adaptive_check_interval": cfg.adaptive_check_interval,
        "adaptive_patience": cfg.adaptive_patience,
        "adaptive_cv_tol": cfg.adaptive_cv_tol,
        "adaptive_min_iter": cfg.adaptive_min_iter,
    }
    sigma_sweep_results_json_path(out_dir, cfg).write_text(
        json.dumps(meta, indent=2) + "\n",
    )


def aic_trials_path(out_dir: Path, cfg: AICSweepConfig) -> Path:
    """Per-trial CSV (all AIC values + ``hat_d``) for replotting."""
    return Path(out_dir) / f"aic_d_sweep_{_config_hash(cfg)}.csv"


def aic_confusion_summary_path(out_dir: Path, cfg: AICSweepConfig) -> Path:
    """Per-(n, d_true) recovery summary derived from trials."""
    return Path(out_dir) / f"aic_d_confusion_n_sweep_{_config_hash(cfg)}.csv"


def aic_results_json_path(out_dir: Path, cfg: AICSweepConfig) -> Path:
    """Full confusion counts + config metadata for offline replot."""
    return Path(out_dir) / f"aic_d_confusion_n_sweep_{_config_hash(cfg)}_results.json"


def _aic_trial_key(row: dict[str, Any] | pd.Series) -> tuple[int, int, int]:
    return (int(row["n"]), int(row["d_true"]), int(row["run"]))


def _expected_aic_trials(cfg: AICSweepConfig) -> int:
    return len(cfg.n_sizes) * len(cfg.d_true_values) * cfg.n_runs


def save_aic_sweep_artifacts(
    df: pd.DataFrame,
    cfg: AICSweepConfig,
    out_dir: Path,
) -> dict[int, dict[int, dict[int, int]]]:
    """Write trials CSV, recovery summary, and confusion JSON."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conf = _confusion_from_df(df, cfg)

    trials_path = aic_trials_path(out_dir, cfg)
    df.to_csv(trials_path, index=False)

    summary_rows = []
    for n in cfg.n_sizes:
        for dt in cfg.d_true_values:
            total = sum(conf[n][dt].values())
            rec = conf[n][dt][dt] / max(1, total)
            summary_rows.append({
                "n": n, "d_true": dt, "recovery": rec, "total": total,
            })
    pd.DataFrame(summary_rows).to_csv(
        aic_confusion_summary_path(out_dir, cfg), index=False,
    )

    meta = {
        "config": cfg.__dict__,
        "n_trials": len(df),
        "n_expected": _expected_aic_trials(cfg),
        "confusion": {
            str(n): {str(dt): conf[n][dt] for dt in cfg.d_true_values}
            for n in cfg.n_sizes
        },
    }
    aic_results_json_path(out_dir, cfg).write_text(
        json.dumps(meta, indent=2, default=str),
    )
    return conf


def load_aic_sweep_results(
    out_dir: Path,
    cfg: AICSweepConfig,
) -> tuple[pd.DataFrame, dict[int, dict[int, dict[int, int]]]]:
    """Load saved per-trial CSV and rebuild the confusion dict."""
    path = aic_trials_path(out_dir, cfg)
    if not path.is_file():
        raise FileNotFoundError(f"No saved AIC trials at {path}")
    df = pd.read_csv(path)
    return df, _confusion_from_df(df, cfg)


def run_aic_d_sweep(
    cfg: AICSweepConfig,
    out_dir: Path,
    *,
    use_cache: bool = True,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, dict[int, dict[int, dict[int, int]]]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = aic_trials_path(out_dir, cfg)
    n_expected = _expected_aic_trials(cfg)

    if trials_path.is_file():
        df_existing = pd.read_csv(trials_path)
        if use_cache and len(df_existing) >= n_expected:
            conf = _confusion_from_df(df_existing, cfg)
            return df_existing, conf

    from .workers import aic_run_job

    ensemble_override = os.environ.get("LG_AIC_ENSEMBLE_JOBS")
    default_ensemble_jobs = (
        int(ensemble_override)
        if ensemble_override is not None
        else None
    )

    completed_keys: set[tuple[int, int, int]] = set()
    rows_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    if trials_path.is_file():
        for _, row in pd.read_csv(trials_path).iterrows():
            key = _aic_trial_key(row)
            completed_keys.add(key)
            rows_by_key[key] = row.to_dict()
        if completed_keys:
            print(
                f"  resume: {len(completed_keys)}/{n_expected} trials cached "
                f"({trials_path.name})",
                flush=True,
            )
            # Per-cell accuracy summary of cached trials.
            resume_tally: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
            for row in rows_by_key.values():
                n_val = int(row["n"]); dt_val = int(row["d_true"])
                hd_val = int(row.get("hat_d", -1))
                resume_tally[(n_val, dt_val)][0] += int(hd_val == dt_val)
                resume_tally[(n_val, dt_val)][1] += 1
            cur_n = None
            for (n_val, dt_val), (c, t) in sorted(resume_tally.items()):
                if cur_n != n_val:
                    if cur_n is not None:
                        print()
                    print(f"    cached n={n_val}:", end="")
                    cur_n = n_val
                acc = 100 * c / t if t else 0
                print(f"  d={dt_val}:{c}/{t}({acc:.0f}%)", end="")
            print(flush=True)

    jobs: list[dict[str, Any]] = []
    for n in cfg.n_sizes:
        nit_default = _aic_iter_count(cfg, n)
        # Per-n sigma: scale sigma with n so d=3 3-hop balls stay ~10-30% of n.
        sigma_for_n = (
            cfg.sigma_gen_per_n[n]
            if (cfg.sigma_gen_per_n and n in cfg.sigma_gen_per_n)
            else cfg.sigma_gen
        )
        for d_true in cfg.d_true_values:
            # Per-d iter cap: allows d=2/d=3 to use smaller absolute caps for
            # speed or to avoid metastable phase transitions. Each value is
            # either an int (flat cap for all n) or a dict {n: cap} for per-n
            # caps — useful when one d needs more sweeps at larger n.
            if cfg.iter_cap_by_d and d_true in cfg.iter_cap_by_d:
                cap_val = cfg.iter_cap_by_d[d_true]
                if isinstance(cap_val, dict):
                    cap = cap_val.get(n)
                    nit = _iter_count(n, cap) if cap is not None else nit_default
                else:
                    nit = _iter_count(n, cap_val)
            else:
                nit = nit_default
            # Per-(n, d_true) sigma cell override: takes precedence over the
            # per-n sigma. Used when one cell needs a different sigma to stay
            # identifiable (e.g., n=500 d=3 needs sigma=-4.8 instead of -4.3
            # to avoid 3-hop ball saturation).
            sigma_for_cell = sigma_for_n
            if cfg.sigma_gen_per_n_d and d_true in cfg.sigma_gen_per_n_d:
                cell_sigmas = cfg.sigma_gen_per_n_d[d_true]
                if isinstance(cell_sigmas, dict) and n in cell_sigmas:
                    sigma_for_cell = float(cell_sigmas[n])
            for run in range(cfg.n_runs):
                key = (n, d_true, run)
                if key in completed_keys:
                    continue
                # Per-(n, d_true) m_ensemble override for speed at expensive cells.
                m_for_cell = cfg.m_ensemble
                if cfg.m_ensemble_by_d and d_true in cfg.m_ensemble_by_d:
                    spec = cfg.m_ensemble_by_d[d_true]
                    if isinstance(spec, dict):
                        m_for_cell = int(spec.get(n, cfg.m_ensemble))
                    else:
                        m_for_cell = int(spec)
                job = {
                    "n": n,
                    "d_true": d_true,
                    "run": run,
                    "n_iter": nit,
                    "m_ensemble": m_for_cell,
                    "sigma_gen": sigma_for_cell,
                    "feature_mode_gen": cfg.feature_mode_gen,
                    "feature_mode_est": cfg.feature_mode_est,
                    "target_density": cfg.target_density,
                    "signal": cfg.signal,
                    "d_est_values": cfg.d_est_values,
                    "aic_penalty_per_d": cfg.aic_penalty_per_d,
                    "seed_base": cfg.seed_base,
                    "n_jobs": n_jobs,
                }
                if default_ensemble_jobs is not None:
                    job["ensemble_jobs"] = default_ensemble_jobs
                jobs.append(job)

    def _persist() -> None:
        ordered = sorted(
            rows_by_key.values(),
            key=lambda r: (int(r["n"]), int(r["d_true"]), int(r["run"])),
        )
        save_aic_sweep_artifacts(pd.DataFrame(ordered), cfg, out_dir)

    n_pending = len(jobs)
    if n_pending == 0:
        df = pd.DataFrame(sorted(
            rows_by_key.values(),
            key=lambda r: (int(r["n"]), int(r["d_true"]), int(r["run"])),
        ))
        conf = _confusion_from_df(df, cfg)
        return df, conf

    # Track per-(n,d_true) accuracy and timing for rich progress output and ETA.
    import time as _time
    cell_tally: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0])
    cell_times: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows_by_key.values():
        n_val = int(row["n"]); dt_val = int(row["d_true"])
        cell_tally[(n_val, dt_val)][0] += int(int(row.get("hat_d", -1)) == dt_val)
        cell_tally[(n_val, dt_val)][1] += 1

    def _eta_seconds(remaining_jobs: list[dict]) -> float:
        """Estimate remaining wall time using observed time per (n, d_true) cell."""
        if not cell_times:
            return float("nan")
        per_n_avg: dict[int, float] = {}
        for (nn, dd), ts in cell_times.items():
            per_n_avg.setdefault(nn, 0.0)
        # Compute per-cell average; fallback to per-n average if cell unseen.
        per_n_means: dict[int, list[float]] = defaultdict(list)
        for (nn, dd), ts in cell_times.items():
            if ts:
                per_n_means[nn].extend(ts)
        total = 0.0
        for j in remaining_jobs:
            key = (int(j["n"]), int(j["d_true"]))
            if key in cell_times and cell_times[key]:
                total += float(np.mean(cell_times[key]))
            elif int(j["n"]) in per_n_means and per_n_means[int(j["n"])]:
                total += float(np.mean(per_n_means[int(j["n"])]))
            else:
                # Last resort: use overall mean
                all_ts = [t for ts in cell_times.values() for t in ts]
                total += float(np.mean(all_ts)) if all_ts else 0.0
        return total / max(n_jobs, 1)

    def _log_row(row: dict, done: int) -> None:
        n_val = int(row["n"]); dt_val = int(row["d_true"])
        run_val = int(row["run"]); hd_val = int(row["hat_d"])
        ok = hd_val == dt_val
        cell_tally[(n_val, dt_val)][0] += int(ok)
        cell_tally[(n_val, dt_val)][1] += 1
        elapsed = float(row.get("elapsed_s", float("nan")))
        density = float(row.get("density", float("nan")))
        if not (elapsed != elapsed):  # not NaN
            cell_times[(n_val, dt_val)].append(elapsed)
        c, t = cell_tally[(n_val, dt_val)]
        cell_str = f"{c}/{t} ({100 * c / t:.0f}%)"
        remaining = [
            jj for jj in jobs
            if (int(jj["n"]), int(jj["d_true"]), int(jj["run"])) not in rows_by_key
        ]
        eta_s = _eta_seconds(remaining)
        eta_str = f"{eta_s / 60:.1f}min" if eta_s == eta_s else "?"
        icon = "✓" if ok else "✗"
        rho_str = f"{density:.3f}" if density == density else "?"
        elapsed_str = f"{elapsed:5.1f}s" if elapsed == elapsed else "    ?"
        print(
            f"  [{done:3d}/{n_expected}] n={n_val:>4d} d={dt_val} r={run_val:<2d}"
            f" -> d_hat={hd_val} {icon}  rho={rho_str}  t={elapsed_str}"
            f"  cell[n={n_val},d={dt_val}]={cell_str}  ETA~{eta_str}",
            flush=True,
        )

    done = len(completed_keys)
    if n_jobs <= 1:
        for idx, payload in enumerate(jobs):
            row = aic_run_job(payload)
            rows_by_key[_aic_trial_key(row)] = row
            done += 1
            _persist()
            _log_row(row, done)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(aic_run_job, payload): payload
                for payload in jobs
            }
            for fut in as_completed(futures):
                row = fut.result()
                rows_by_key[_aic_trial_key(row)] = row
                done += 1
                _persist()
                _log_row(row, done)

    df = pd.DataFrame(sorted(
        rows_by_key.values(),
        key=lambda r: (int(r["n"]), int(r["d_true"]), int(r["run"])),
    ))
    conf = save_aic_sweep_artifacts(df, cfg, out_dir)
    return df, conf


def _confusion_from_df(
    df: pd.DataFrame,
    cfg: AICSweepConfig,
) -> dict[int, dict[int, dict[int, int]]]:
    conf: dict[int, dict[int, dict[int, int]]] = {
        n: {dt: {de: 0 for de in cfg.d_est_values} for dt in cfg.d_true_values}
        for n in cfg.n_sizes
    }
    for _, row in df.iterrows():
        n = int(row["n"])
        dt = int(row["d_true"])
        hd = int(row["hat_d"])
        conf[n][dt][hd] += 1
    return conf


def plot_convergence_sigma(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = {
        -2.0: ("#0072B2", "o"),
        -4.0: ("#E69F00", "s"),
        -6.0: ("#009E73", "^"),
        -8.0: ("#D55E00", "D"),
    }
    d_values = sorted(df["d"].unique())
    fig, axes = plt.subplots(1, len(d_values), figsize=(5 * len(d_values), 5), sharey=True)
    if len(d_values) == 1:
        axes = [axes]
    sigma_values = sorted(df["sigma_true"].unique())
    for ax, d in zip(axes, d_values):
        for sigma in sigma_values:
            sub = df[(df.d == d) & (df.sigma_true == sigma)].sort_values("n")
            color, marker = palette.get(float(sigma), ("#333", "o"))
            ax.plot(sub.n, sub.sigma_hat_mean, marker=marker, color=color, label=f"$\\sigma={int(sigma)}$")
            ax.fill_between(sub.n, sub.ci_lo, sub.ci_hi, color=color, alpha=0.15)
            ax.axhline(sigma, color=color, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("$n$")
        ax.set_title(f"$d={int(d)}$")
        ax.grid(alpha=0.3)
        # Clip y-axis to a paper-style range. Degenerate small-n + very-negative-σ
        # cells produce σ̂ ≈ -30 (zero-edge graphs → logit clamp). Those values
        # would stretch the axis and squash the convergence story. We keep the
        # data; we just don't render those outliers.
        y_min = min(sigma_values) - 2.0  # leave room below the most-negative σ
        ax.set_ylim(y_min, 2.0)
    axes[0].set_ylabel(r"$\hat{\sigma}$")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(
        r"Convergence of $\hat{\sigma}$ to the true parameter as $n$ increases",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aic_confusion(
    conf: dict[int, dict[int, dict[int, int]]],
    d_values: list[int],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_sizes = sorted(conf.keys())
    nd = len(d_values)
    ncols = len(n_sizes)
    # constrained_layout coordinates colorbar placement; +0.8" leaves room
    # so the colorbar sits to the right of the last panel rather than over it.
    fig, axes = plt.subplots(
        1, ncols, figsize=(5.5 * ncols + 0.8, 5.2), constrained_layout=True,
    )
    if ncols == 1:
        axes = [axes]

    im_ref = None
    for col, (ax, n) in enumerate(zip(axes, n_sizes)):
        mat = np.zeros((nd, nd))
        for i, dt in enumerate(d_values):
            total = sum(conf[n][dt].values())
            for j, de in enumerate(d_values):
                mat[i, j] = conf[n][dt][de] / max(1, total)

        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        im_ref = im

        for i in range(nd):
            for j in range(nd):
                v = mat[i, j]
                ax.text(
                    j, i, f"{v*100:.0f}%",
                    ha="center", va="center", fontsize=13,
                    color="white" if v > 0.55 else "black",
                    fontweight="bold" if i == j else "normal",
                )

        ax.set_xticks(range(nd))
        ax.set_yticks(range(nd))
        ax.set_xticklabels([str(d) for d in d_values], fontsize=12)
        ax.set_yticklabels([str(d) for d in d_values], fontsize=12)
        ax.set_xlabel(r"Selected $\hat{d} = \mathrm{arg\,min}\,\mathrm{AIC}(d_{\mathrm{est}})$",
                      fontsize=11)
        if col == 0:
            ax.set_ylabel(r"True $d_{\mathrm{true}}$", fontsize=12)

        overall = 100.0 * sum(
            conf[n][dt].get(dt, 0)
            for dt in d_values
        ) / max(1, sum(
            sum(conf[n][dt].values()) for dt in d_values
        ))
        ax.set_title(
            rf"$n = {n}$  (overall accuracy $= {overall:.0f}\%$)",
            fontsize=13, pad=8,
        )

    # Single colorbar on the far right; constrained_layout reserves the slot.
    if im_ref is not None:
        cbar = fig.colorbar(im_ref, ax=axes, shrink=0.85, pad=0.02)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label(r"$\Pr(\hat{d} = d_{\mathrm{est}} \mid d_{\mathrm{true}})$",
                       fontsize=10, rotation=270, labelpad=16)

    fig.suptitle(
        "AIC-based selection of $d$: accuracy improves with graph size $n$",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
