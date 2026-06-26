#!/usr/bin/env python3
"""Temporal Logit-Graph (TLG) parameter-behavior characterization: simulate growth graphs over a
grid of (sigma, alpha) for n in {50, 200, 500} and register structural descriptors — centrality,
clustering/connectivity, degree heterogeneity, and power-law / scale-free fits — so the influence
of (sigma, alpha) on graph behavior can be read off heatmaps. `make tlg-param-behavior`.

Model: the equilibrium TLG (``grow_graph`` with the default ``allow_removal=True`` ergodic chain),
degree-feature depth d=1 (at d=0 the feature is 0 and alpha has no effect). Set
``LG_TLGB_ALLOW_REMOVAL=0`` for the add-only growth (preferential-attachment) regime instead.
Reproducible (seeded) and cached: per-n raw metrics CSVs keyed by a config hash."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from logit_graph.temporal import grow_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Config (all env-overridable, matching the other tlg- experiments)
# ---------------------------------------------------------------------------

def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


def _floats(env, default):
    raw = os.environ.get(env)
    return [float(x) for x in raw.split(",")] if raw else default


def _ints(env, default):
    raw = os.environ.get(env)
    return [int(x) for x in raw.split(",")] if raw else default


QUICK = os.environ.get("LG_TLGB_QUICK", "0") == "1"
SEED = _int("LG_TLGB_SEED", 20260626)
NREPS = _int("LG_TLGB_NREPS", 2 if QUICK else 5)
NSTEPS = _int("LG_TLGB_NSTEPS", 6 if QUICK else 20)
D_FIX = _int("LG_TLGB_D", 1)               # d=0 => degree feature is 0 => alpha inert
P0 = float(os.environ.get("LG_TLGB_P0", "0.02"))
ALLOW_REMOVAL = os.environ.get("LG_TLGB_ALLOW_REMOVAL", "1") == "1"
USE_CACHE = os.environ.get("LG_TLGB_USE_CACHE", "1") == "1"

SIGMAS = _floats("LG_TLGB_SIGMAS", [-3.0, -2.0] if QUICK else [-4.0, -3.0, -2.0])
ALPHAS = _floats("LG_TLGB_ALPHAS", [0.0, 0.3] if QUICK else [0.0, 0.15, 0.30, 0.45])
NS = _ints("LG_TLGB_NS", [50] if QUICK else [50, 200, 500])

OUT_DIR = _here / "runs" / "tlg_param_behavior"

# Metrics shown as heatmaps (key -> display label). The CSV keeps every metric;
# this is just the curated panel for the param-behavior figure.
HEATMAP_METRICS = [
    ("density", "density"),
    ("mean_degree", "mean degree"),
    ("deg_gini", "degree Gini"),
    ("assortativity", "degree assortativity"),
    ("avg_clustering", "avg clustering"),
    ("lcc_frac", "largest-comp. frac."),
    ("betweenness_max", "max betweenness"),
    ("pl_alpha", "power-law exponent"),
    ("frac_scale_free", "scale-free fraction"),
]


def _cache_key(n: int) -> str:
    payload = (SIGMAS, ALPHAS, D_FIX, NREPS, NSTEPS, ALLOW_REMOVAL, P0, SEED, n)
    return hashlib.md5(repr(payload).encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0=equal, ->1=concentrated)."""
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, x.size + 1)
    return float((2.0 * np.sum(idx * x) / (x.size * x.sum())) - (x.size + 1.0) / x.size)


def _powerlaw_fit(degs: np.ndarray) -> dict:
    """Fit a discrete power law to the (positive) degree sequence and compare it to an
    exponential. Returns the exponent, x_min, KS distance, the (normalized) loglikelihood
    ratio R vs exponential with its p-value, and an is_scale_free flag (R>0, p<0.1, 2<=a<=3.5)."""
    out = dict(pl_alpha=np.nan, pl_xmin=np.nan, pl_ks=np.nan,
               pl_R_exp=np.nan, pl_p_exp=np.nan, is_scale_free=0.0)
    d = np.asarray(degs, dtype=float)
    d = d[d > 0]
    if d.size < 10 or np.unique(d).size < 3:
        return out
    try:
        import powerlaw  # local dependency; imported lazily to keep import cost off the hot path
        with warnings.catch_warnings(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            fit = powerlaw.Fit(d, discrete=True, verbose=False)
            a = float(fit.alpha)
            xmin = float(fit.xmin)
            ks = float(fit.power_law.D)
            R, p = fit.distribution_compare("power_law", "exponential",
                                            normalized_ratio=True)
    except Exception:
        return out
    out.update(pl_alpha=a, pl_xmin=xmin, pl_ks=ks, pl_R_exp=float(R), pl_p_exp=float(p))
    out["is_scale_free"] = float(np.isfinite(a) and R > 0 and p < 0.1 and 2.0 <= a <= 3.5)
    return out


def graph_metrics(adj: np.ndarray) -> dict:
    """Register the full structural-descriptor set for one graph (adjacency, 0/1, symmetric)."""
    n = adj.shape[0]
    A = (np.asarray(adj) > 0).astype(int)
    np.fill_diagonal(A, 0)
    G = nx.from_numpy_array(A)
    degs = np.array([deg for _, deg in G.degree()], dtype=float)
    mean_deg = float(degs.mean()) if degs.size else 0.0

    m = dict(
        n_edges=int(G.number_of_edges()),
        density=float(nx.density(G)),
        mean_degree=mean_deg,
        max_degree=float(degs.max()) if degs.size else 0.0,
        deg_gini=_gini(degs),
        deg_cv=float(degs.std() / mean_deg) if mean_deg > 0 else 0.0,
    )
    try:
        m["assortativity"] = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        m["assortativity"] = np.nan

    m["avg_clustering"] = float(nx.average_clustering(G))
    m["transitivity"] = float(nx.transitivity(G))

    comps = list(nx.connected_components(G))
    lcc = max(comps, key=len) if comps else set()
    m["n_components"] = len(comps)
    m["lcc_frac"] = len(lcc) / n if n else 0.0
    H = G.subgraph(lcc)
    if H.number_of_nodes() > 1:
        m["avg_path_len"] = float(nx.average_shortest_path_length(H))
        m["diameter"] = float(nx.diameter(H))
    else:
        m["avg_path_len"] = np.nan
        m["diameter"] = np.nan

    bc = np.fromiter(nx.betweenness_centrality(G, normalized=True).values(), dtype=float)
    m["betweenness_mean"] = float(bc.mean()) if bc.size else 0.0
    m["betweenness_max"] = float(bc.max()) if bc.size else 0.0
    cc = np.fromiter(nx.closeness_centrality(G).values(), dtype=float)
    m["closeness_mean"] = float(cc.mean()) if cc.size else 0.0
    try:
        ev = np.fromiter(nx.eigenvector_centrality_numpy(G).values(), dtype=float)
        m["eigenvector_max"] = float(np.abs(ev).max())
    except Exception:
        m["eigenvector_max"] = np.nan
    try:
        m["spectral_radius"] = float(np.max(np.linalg.eigvalsh(A.astype(float))))
    except Exception:
        m["spectral_radius"] = np.nan

    m.update(_powerlaw_fit(degs))
    return m


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_n(n: int):
    """Run the (sigma, alpha) grid at one n. Returns (long-form per-rep metrics DataFrame,
    {cell_key: rep-0 degree sequence} for the CCDF panel)."""
    rows = []
    deg_seqs = {}
    for si, sigma in enumerate(SIGMAS):
        for ai, alpha in enumerate(ALPHAS):
            t0 = time.perf_counter()
            for rep in range(NREPS):
                seed = SEED + 100003 * si + 1009 * ai + 13 * rep + n
                res = grow_graph(n, d=D_FIX, sigma=sigma, alpha=alpha,
                                 n_steps=NSTEPS, seed=seed,
                                 allow_removal=ALLOW_REMOVAL,
                                 record_design=False, store_snapshots=False)
                met = graph_metrics(res.adj)
                rows.append(dict(n=n, sigma=sigma, alpha=alpha, rep=rep, seed=seed, **met))
                if rep == 0:
                    A = (np.asarray(res.adj) > 0).astype(int)
                    np.fill_diagonal(A, 0)
                    deg_seqs[f"s{si}_a{ai}"] = A.sum(axis=1).astype(int)
            dt = time.perf_counter() - t0
            print(f"    n={n:4d} sigma={sigma:+.2f} alpha={alpha:.2f}: "
                  f"{NREPS} reps in {dt:5.1f}s")
            sys.stdout.flush()
    return pd.DataFrame(rows), deg_seqs


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean + std of every numeric metric over reps, per (n, sigma, alpha)."""
    metric_cols = [c for c in raw.columns
                   if c not in ("n", "sigma", "alpha", "rep", "seed")]
    g = raw.groupby(["n", "sigma", "alpha"], as_index=False)
    summ = g[metric_cols].mean()
    std = g[metric_cols].std().rename(columns={c: f"{c}_std" for c in metric_cols})
    return pd.concat([summ, std[[c for c in std.columns if c.endswith("_std")]]], axis=1)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _grid(summary: pd.DataFrame, n: int, metric: str) -> np.ndarray:
    """metric mean as a (len(SIGMAS), len(ALPHAS)) grid (rows top=high sigma)."""
    M = np.full((len(SIGMAS), len(ALPHAS)), np.nan)
    sub = summary[summary["n"] == n]
    for si, s in enumerate(SIGMAS):
        for ai, a in enumerate(ALPHAS):
            cell = sub[(np.isclose(sub["sigma"], s)) & (np.isclose(sub["alpha"], a))]
            if len(cell):
                M[si, ai] = float(cell[metric].iloc[0])
    return M


def plot_heatmaps(summary: pd.DataFrame, out_path: Path):
    """metrics (rows) x n (cols) grid of heatmaps over the (alpha, sigma) plane."""
    nrows, ncols = len(HEATMAP_METRICS), len(NS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.7 * nrows),
                             squeeze=False)
    for ri, (key, label) in enumerate(HEATMAP_METRICS):
        for ci, n in enumerate(NS):
            ax = axes[ri][ci]
            M = _grid(summary, n, key)
            im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis")
            ax.set_xticks(range(len(ALPHAS)))
            ax.set_xticklabels([f"{a:g}" for a in ALPHAS], fontsize=8)
            ax.set_yticks(range(len(SIGMAS)))
            ax.set_yticklabels([f"{s:g}" for s in SIGMAS], fontsize=8)
            for si in range(len(SIGMAS)):
                for ai in range(len(ALPHAS)):
                    v = M[si, ai]
                    if np.isfinite(v):
                        ax.text(ai, si, f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color="w")
            if ri == 0:
                ax.set_title(f"n = {n}", fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{label}\nσ", fontsize=9)
            if ri == nrows - 1:
                ax.set_xlabel("α (degree slope)", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    regime = "add+remove (equilibrium)" if ALLOW_REMOVAL else "add-only (growth)"
    fig.suptitle(f"TLG structural behavior vs (σ, α)   [d={D_FIX}, {NSTEPS} steps, "
                 f"{regime}, mean of {NREPS} reps]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_scale_free_vs_alpha(summary: pd.DataFrame, out_path: Path):
    """Power-law exponent and scale-free fraction vs alpha (one line per sigma), per n."""
    fig, axes = plt.subplots(2, len(NS), figsize=(4.2 * len(NS), 7),
                             sharex=True, squeeze=False)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(SIGMAS)))
    for ci, n in enumerate(NS):
        sub = summary[summary["n"] == n]
        for si, s in enumerate(SIGMAS):
            cell = sub[np.isclose(sub["sigma"], s)].sort_values("alpha")
            axes[0][ci].plot(cell["alpha"], cell["pl_alpha"], marker="o",
                             color=colors[si], label=f"σ={s:g}")
            axes[1][ci].plot(cell["alpha"], cell["frac_scale_free"], marker="s",
                             color=colors[si], label=f"σ={s:g}")
        axes[0][ci].set_title(f"n = {n}")
        axes[1][ci].set_xlabel("α (degree slope)")
        for r in (0, 1):
            axes[r][ci].grid(alpha=0.25)
    axes[0][0].set_ylabel("power-law exponent")
    axes[1][0].set_ylabel("scale-free fraction")
    axes[0][-1].legend(fontsize=8, frameon=False)
    fig.suptitle("Scale-free behavior vs α  (lines = σ)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_degree_ccdf(deg_seqs: dict, n: int, out_path: Path):
    """Log-log CCDF of the degree distribution for each (sigma, alpha) cell at one n."""
    fig, axes = plt.subplots(len(SIGMAS), len(ALPHAS),
                             figsize=(3.0 * len(ALPHAS), 2.6 * len(SIGMAS)),
                             squeeze=False)
    for si, s in enumerate(SIGMAS):
        for ai, a in enumerate(ALPHAS):
            ax = axes[si][ai]
            seq = deg_seqs.get(f"s{si}_a{ai}")
            if seq is not None:
                d = np.sort(np.asarray(seq, dtype=float))
                d = d[d > 0]
                if d.size:
                    ccdf = 1.0 - np.arange(d.size) / d.size
                    ax.loglog(d, ccdf, marker=".", ls="none", ms=4)
            ax.set_title(f"σ={s:g}, α={a:g}", fontsize=9)
            ax.grid(alpha=0.25, which="both")
            if si == len(SIGMAS) - 1:
                ax.set_xlabel("degree", fontsize=8)
            if ai == 0:
                ax.set_ylabel("CCDF", fontsize=8)
    fig.suptitle(f"Degree distribution CCDF (log-log) at n = {n}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    regime = "add+remove" if ALLOW_REMOVAL else "add-only"
    print(f"TLG param-behavior  seed={SEED}  quick={QUICK}  d={D_FIX}  regime={regime}\n"
          f"  n={NS}  sigma={SIGMAS}  alpha={ALPHAS}  reps={NREPS}  steps={NSTEPS}")

    all_raw = []
    deg_seqs_by_n = {}
    for n in NS:
        key = _cache_key(n)
        raw_path = OUT_DIR / f"metrics_raw_n{n}_{key}.csv"
        seq_path = OUT_DIR / f"degree_seqs_n{n}_{key}.npz"
        print(f"\n[n={n}]")
        if USE_CACHE and raw_path.exists():
            print(f"  using cached {raw_path.name}")
            raw = pd.read_csv(raw_path)
            if seq_path.exists():
                npz = np.load(seq_path)
                deg_seqs_by_n[n] = {k: npz[k] for k in npz.files}
        else:
            raw, deg_seqs = simulate_n(n)
            raw.to_csv(raw_path, index=False)
            np.savez(seq_path, **{k: np.asarray(v) for k, v in deg_seqs.items()})
            deg_seqs_by_n[n] = deg_seqs
        all_raw.append(raw)

    raw_all = pd.concat(all_raw, ignore_index=True)
    summary = aggregate(raw_all)
    # Mean of the per-rep 0/1 is_scale_free flag = fraction of reps that are scale-free.
    summary["frac_scale_free"] = summary["is_scale_free"]
    summary.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    raw_all.to_csv(OUT_DIR / "metrics_raw_all.csv", index=False)

    plot_heatmaps(summary, OUT_DIR / "param_behavior_heatmaps.png")
    plot_scale_free_vs_alpha(summary, OUT_DIR / "scale_free_vs_alpha.png")
    n_big = NS[-1]
    if deg_seqs_by_n.get(n_big):
        plot_degree_ccdf(deg_seqs_by_n[n_big], n_big, OUT_DIR / "degree_ccdf.png")

    (OUT_DIR / "config.json").write_text(json.dumps(
        dict(seed=SEED, d=D_FIX, ns=NS, sigmas=SIGMAS, alphas=ALPHAS,
             n_reps=NREPS, n_steps=NSTEPS, allow_removal=ALLOW_REMOVAL, p0=P0),
        indent=2))

    print("\n" + "=" * 72)
    print("Behavior summary (mean over reps):")
    for n in NS:
        print(f"\n n = {n}")
        sub = summary[summary["n"] == n].sort_values(["sigma", "alpha"])
        for r in sub.itertuples():
            sf = "  <scale-free>" if getattr(r, "frac_scale_free", 0) >= 0.5 else ""
            print(f"  σ={r.sigma:+.2f} α={r.alpha:.2f}: dens={r.density:.3f} "
                  f"<k>={r.mean_degree:5.1f} gini={r.deg_gini:.2f} "
                  f"clust={r.avg_clustering:.2f} assort={r.assortativity:+.2f} "
                  f"pl_a={r.pl_alpha:.2f}{sf}")
    print(f"\nWrote {OUT_DIR}/ : metrics_summary.csv, metrics_raw_all.csv, "
          f"param_behavior_heatmaps.png, scale_free_vs_alpha.png, degree_ccdf.png")


if __name__ == "__main__":
    main()
