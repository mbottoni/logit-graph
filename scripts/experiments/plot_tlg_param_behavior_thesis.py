#!/usr/bin/env python3
"""Curated, publication-quality figures from the cached TLG parameter-behavior sweep.

Reads the aggregated `metrics_summary.csv` written by run_tlg_param_behavior.py (does NOT rerun
the experiment) and emits two clean figures for the thesis:

  tlg_param_behavior_heatmap.png : 2x2 heatmaps of four descriptors (density, mean degree,
                                   degree-CV, clustering) over the (sigma, alpha) plane at a
                                   fixed (n, d) -- "what the parameters do".
  tlg_param_nd_grid.png          : 3x3 density phase-diagram grid, radius d down the rows and
                                   size n across the columns -- "how the response scales with d, n".

Usage:
    python scripts/experiments/plot_tlg_param_behavior_thesis.py [--n 500] [--d 1] [--out DIR]
The defaults reproduce the thesis figures. Output defaults to the sweep's runs directory.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_here = Path(__file__).resolve().parent
DEFAULT_CSV = _here / "runs" / "tlg_param_behavior" / "metrics_summary.csv"
DEFAULT_OUT = _here / "runs" / "tlg_param_behavior"

# (column, display label, colour scale, colormap) for the 2x2 response figure.
PANELS = [
    ("density", "Density", None, "viridis"),
    ("mean_degree", "Mean degree", "log", "viridis"),
    ("deg_cv", "Degree heterogeneity (CV)", None, "magma"),
    ("avg_clustering", "Average clustering", None, "cividis"),
]


def _grid(df, sig, alp, metric):
    """(sigma x alpha) matrix for one metric, sigma descending down the rows."""
    return df.pivot(index="sigma", columns="alpha", values=metric).reindex(
        index=sig, columns=alp).values


def plot_response(s, n, d, out_path):
    """2x2 heatmaps of four descriptors over the (sigma, alpha) plane at fixed (n, d)."""
    sl = s[(s.n == n) & (s.d == d)]
    sig = sorted(sl.sigma.unique(), reverse=True)
    alp = sorted(sl.alpha.unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4))
    for ax, (key, label, scale, cmap) in zip(axes.flat, PANELS):
        M = _grid(sl, sig, alp, key)
        norm = LogNorm(vmin=max(M.min(), 1e-3), vmax=M.max()) if scale == "log" else None
        im = ax.imshow(M, aspect="auto", cmap=cmap, norm=norm, origin="upper")
        ax.set_xticks(range(len(alp))); ax.set_xticklabels([f"{a:g}" for a in alp], fontsize=9)
        ax.set_yticks(range(len(sig))); ax.set_yticklabels([f"{v:g}" for v in sig], fontsize=9)
        ax.set_xlabel(r"neighborhood-degree effect  $\alpha$", fontsize=10)
        ax.set_ylabel(r"baseline  $\sigma$", fontsize=10)
        ax.set_title(label, fontsize=12, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle(r"Structural response of the LG model to $(\sigma,\alpha)$   "
                 f"($n={n}$, $d={d}$, mean of 5 replicates)", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_nd_grid(s, metric, out_path):
    """3x3 phase-diagram grid of `metric` over (sigma, alpha), rows = d, cols = n."""
    ns = sorted(s.n.unique())
    ds = sorted(s.d.unique())
    sig = sorted(s.sigma.unique(), reverse=True)
    alp = sorted(s.alpha.unique())
    vmax = float(s[metric].max())
    fig, axes = plt.subplots(len(ds), len(ns), figsize=(11, 9), squeeze=False)
    im = None
    for ri, d in enumerate(ds):
        for ci, n in enumerate(ns):
            ax = axes[ri][ci]
            M = _grid(s[(s.n == n) & (s.d == d)], sig, alp, metric)
            im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, origin="upper")
            if ri == 0:
                ax.set_title(f"$n = {n}$", fontsize=12, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"$d = {d}$\n\n" + r"baseline $\sigma$", fontsize=11)
                ax.set_yticks(range(len(sig))); ax.set_yticklabels([f"{v:g}" for v in sig], fontsize=8)
            else:
                ax.set_yticks([])
            if ri == len(ds) - 1:
                ax.set_xlabel(r"$\alpha$", fontsize=11)
                ax.set_xticks(range(len(alp))); ax.set_xticklabels([f"{a:g}" for a in alp], fontsize=7)
            else:
                ax.set_xticks([])
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02); cb.set_label(metric, fontsize=11)
    fig.suptitle(r"Density phase diagram across neighborhood radius $d$ (rows) and size $n$ (columns)",
                 fontsize=13, y=0.97)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="cached metrics_summary.csv")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--n", type=int, default=500, help="size for the response figure")
    ap.add_argument("--d", type=int, default=1, help="radius for the response figure")
    ap.add_argument("--metric", default="density", help="metric for the d/n grid figure")
    args = ap.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"cached summary not found: {args.csv}\n"
                         f"Run `make tlg-param-behavior` first to produce it.")
    s = pd.read_csv(args.csv)
    args.out.mkdir(parents=True, exist_ok=True)

    p1 = args.out / "tlg_param_behavior_heatmap.png"
    p2 = args.out / "tlg_param_nd_grid.png"
    plot_response(s, args.n, args.d, p1)
    plot_nd_grid(s, args.metric, p2)
    print(f"Wrote {p1}\nWrote {p2}")


if __name__ == "__main__":
    main()
