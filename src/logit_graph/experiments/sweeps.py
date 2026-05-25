"""Cached experiment sweep runners for sigma and AIC-d selection."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .. import graph as graph_mod
from ..lg_features import FeatureMode, recommended_iterations
from ..logit_estimator import LogitRegEstimator
from .presets import AICSweepConfig, SigmaSweepConfig


def _config_hash(cfg: Any) -> str:
    blob = json.dumps(cfg.__dict__, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _iter_count(
    n: int,
    cap: Optional[int],
    sigma_true: Optional[float] = None,
) -> int:
    base = recommended_iterations(n)
    if sigma_true is not None and sigma_true <= -6.0:
        base = int(1.5 * base)
    return min(base, cap) if cap is not None else base


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

    gm = graph_mod.GraphModel(
        n=n,
        d=d,
        sigma=sigma,
        alpha=1.0,
        beta=float(use_beta),
        er_p=er_init,
        layer2=True,
        feature_mode=gen_mode,
    )
    if seed is not None:
        gm._rng = np.random.default_rng(seed)

    gm.populate_edges_baseline(
        warm_up=0,
        max_iterations=n_iter,
        patience=10,
        check_interval=10**9,
        fast_mode=True,
    )
    adj = gm.graph.copy()
    if not return_meta:
        return adj

    feat_seed = seed if seed is not None else 0
    meta = {
        "sigma": float(sigma),
        "beta": float(use_beta),
        "density": _graph_density(adj),
        "feature_mean": _mean_pair_feature(adj, d, gen_mode, feat_seed),
    }
    return adj, meta


def _aic_ensemble(
    graphs: list[np.ndarray],
    d_est: int,
    feature_mode: FeatureMode,
    extra_penalty: float = 0.0,
) -> dict[str, float]:
    from ..lg_features import build_pair_dataset
    import statsmodels.api as sm
    import warnings

    all_off: list[np.ndarray] = []
    all_lab: list[np.ndarray] = []
    for g in graphs:
        off, lab = build_pair_dataset(g, d=d_est, mode=feature_mode, layer2=True)
        all_off.append(off)
        all_lab.append(lab)
    offsets = np.concatenate(all_off)
    labels = np.concatenate(all_lab)
    y = labels.astype(int)
    x = np.ones((len(y), 1), dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for method in ("bfgs", "newton", "lbfgs"):
            try:
                res = sm.Logit(y, x, offset=offsets).fit(method=method, disp=False, maxiter=200)
                if np.isfinite(res.llf):
                    ll = float(res.llf)
                    return {
                        "aic": -2.0 * ll + 2.0 + extra_penalty,
                        "ll": ll,
                        "k": 1.0,
                        "sigma_hat": float(res.params[0]),
                        "d_est": float(d_est),
                        "n_obs": float(len(y)),
                    }
            except Exception:
                continue
    return {"aic": float("nan"), "ll": float("nan"), "k": 1.0, "sigma_hat": float("nan"),
            "d_est": float(d_est), "n_obs": float(len(y))}


def select_d_ensemble(
    graphs: list[np.ndarray],
    d_candidates: list[int],
    feature_mode: FeatureMode,
    extra_penalty_per_d: float = 0.0,
) -> tuple[int, dict[int, dict[str, float]]]:
    stats = {
        d: _aic_ensemble(
            graphs, d, feature_mode,
            extra_penalty=extra_penalty_per_d * d,
        )
        for d in d_candidates
    }
    valid = {d: s for d, s in stats.items() if np.isfinite(s["aic"])}
    best = min(valid, key=lambda d: valid[d]["aic"]) if valid else d_candidates[0]
    return best, stats


def estimate_sigma_from_graph(
    adj: np.ndarray,
    d: int,
    feature_mode: FeatureMode = "incremental",
    beta: float = 1.0,
) -> float:
    est = LogitRegEstimator(adj, d=d, layer2=True, feature_mode=feature_mode)
    stats = est.compute_aic(d_est=d, feature_mode=feature_mode)
    return float(stats["sigma_hat"])


def run_sigma_sweep(
    cfg: SigmaSweepConfig,
    out_dir: Path,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"convergence_sigma_{_config_hash(cfg)}.csv"

    if use_cache and cache_path.is_file():
        return pd.read_csv(cache_path)

    records: list[dict[str, Any]] = []
    total_cells = len(cfg.d_values) * len(cfg.sigma_values) * len(cfg.n_values)
    cell_idx = 0
    for d in cfg.d_values:
        for sigma_true in cfg.sigma_values:
            for n in cfg.n_values:
                cell_idx += 1
                nit = _iter_count(n, cfg.iter_cap, sigma_true=sigma_true)
                print(
                    f"[sigma sweep {cell_idx}/{total_cells}] d={d} sigma={sigma_true} n={n} "
                    f"iters={nit} reps={cfg.n_reps}",
                    flush=True,
                )
                hats: list[float] = []
                densities: list[float] = []
                betas: list[float] = []
                features: list[float] = []
                for rep in range(cfg.n_reps):
                    seed = cfg.seed_base + hash((d, sigma_true, n, rep)) % (2**31 - 1)
                    adj, meta = simulate_graph(
                        n, d, sigma=sigma_true,
                        n_iter=nit,
                        feature_mode=cfg.feature_mode_gen,
                        target_density=cfg.target_density,
                        signal=cfg.signal,
                        seed=seed,
                        return_meta=True,
                    )
                    mode_est: FeatureMode = (
                        "incremental" if d == 0 else cfg.feature_mode_est
                    )
                    sh = estimate_sigma_from_graph(adj, d, feature_mode=mode_est)
                    hats.append(sh)
                    densities.append(meta["density"])
                    betas.append(meta["beta"])
                    features.append(meta["feature_mean"])

                arr = np.asarray(hats, dtype=float)
                m = float(np.nanmean(arr))
                se = float(np.nanstd(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
                records.append({
                    "d": d,
                    "sigma_true": sigma_true,
                    "n": n,
                    "sigma_hat_mean": m,
                    "sigma_hat_std": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "ci_lo": m - 1.96 * se,
                    "ci_hi": m + 1.96 * se,
                    "n_iter": nit,
                    "n_reps": cfg.n_reps,
                    "density_mean": float(np.mean(densities)),
                    "density_std": float(np.std(densities, ddof=1)) if len(densities) > 1 else 0.0,
                    "beta_mean": float(np.mean(betas)),
                    "feature_mean": float(np.mean(features)),
                    "sigma_error": m - sigma_true,
                })
                pd.DataFrame(records).to_csv(cache_path, index=False)

    df = pd.DataFrame(records)
    df.to_csv(cache_path, index=False)
    df.to_csv(out_dir / "convergence_sigma.csv", index=False)
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


def run_aic_d_sweep(
    cfg: AICSweepConfig,
    out_dir: Path,
    *,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict[int, dict[int, dict[int, int]]]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"aic_d_sweep_{_config_hash(cfg)}.csv"

    if use_cache and cache_path.is_file():
        df = pd.read_csv(cache_path)
        conf = _confusion_from_df(df, cfg)
        return df, conf

    rows: list[dict[str, Any]] = []
    conf: dict[int, dict[int, dict[int, int]]] = {
        n: {dt: {de: 0 for de in cfg.d_est_values} for dt in cfg.d_true_values}
        for n in cfg.n_sizes
    }

    for n in cfg.n_sizes:
        nit = _iter_count(n, cfg.iter_cap)
        for d_true in cfg.d_true_values:
            for run in range(cfg.n_runs):
                graphs = []
                for m in range(cfg.m_ensemble):
                    seed = cfg.seed_base + n * 1000 + d_true * 100 + run * 10 + m
                    adj = simulate_graph(
                        n, d_true, sigma=cfg.sigma_gen,
                        n_iter=nit,
                        feature_mode=cfg.feature_mode_gen,
                        target_density=cfg.target_density,
                        signal=cfg.signal,
                        seed=seed,
                    )
                    graphs.append(adj)

                hat_d, aic_stats = select_d_ensemble(
                    graphs,
                    d_candidates=cfg.d_est_values,
                    feature_mode=cfg.feature_mode_est,
                    extra_penalty_per_d=cfg.aic_penalty_per_d,
                )
                conf[n][d_true][hat_d] += 1
                rows.append({
                    "n": n,
                    "d_true": d_true,
                    "run": run,
                    "hat_d": hat_d,
                    **{f"aic_d{de}": aic_stats[de]["aic"] for de in cfg.d_est_values},
                })

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)

    summary_rows = []
    for n in cfg.n_sizes:
        for dt in cfg.d_true_values:
            total = sum(conf[n][dt].values())
            rec = conf[n][dt][dt] / max(1, total)
            summary_rows.append({
                "n": n, "d_true": dt, "recovery": rec, "total": total,
            })
    pd.DataFrame(summary_rows).to_csv(out_dir / "aic_d_confusion_n_sweep.csv", index=False)

    meta = {
        "config": cfg.__dict__,
        "confusion": {
            str(n): {str(dt): conf[n][dt] for dt in cfg.d_true_values}
            for n in cfg.n_sizes
        },
    }
    (out_dir / "aic_d_confusion_n_sweep_results.json").write_text(
        json.dumps(meta, indent=2, default=str),
    )
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
    for ax, d in zip(axes, d_values):
        for sigma in sorted(df["sigma_true"].unique()):
            sub = df[(df.d == d) & (df.sigma_true == sigma)].sort_values("n")
            color, marker = palette.get(float(sigma), ("#333", "o"))
            ax.plot(sub.n, sub.sigma_hat_mean, marker=marker, color=color, label=f"$\\sigma={int(sigma)}$")
            ax.fill_between(sub.n, sub.ci_lo, sub.ci_hi, color=color, alpha=0.15)
            ax.axhline(sigma, color=color, ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("$n$")
        ax.set_title(f"$d={int(d)}$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\hat{\sigma}$")
    axes[0].legend(fontsize=8)
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
    fig, axes = plt.subplots(1, len(n_sizes), figsize=(5 * len(n_sizes), 5))
    if len(n_sizes) == 1:
        axes = [axes]
    for ax, n in zip(axes, n_sizes):
        mat = np.zeros((len(d_values), len(d_values)))
        for i, dt in enumerate(d_values):
            total = sum(conf[n][dt].values())
            for j, de in enumerate(d_values):
                mat[i, j] = conf[n][dt][de] / max(1, total)
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(d_values)):
            for j in range(len(d_values)):
                v = mat[i, j]
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(d_values)))
        ax.set_yticks(range(len(d_values)))
        ax.set_xticklabels([str(d) for d in d_values])
        ax.set_yticklabels([str(d) for d in d_values])
        acc = np.trace(mat) / len(d_values)
        ax.set_title(f"$n={n}$, acc={acc*100:.0f}%")
        ax.set_xlabel(r"$\hat d$")
        ax.set_ylabel(r"$d_{\mathrm{true}}$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
