#!/usr/bin/env python3
"""Temporal Logit-Graph (TLG) AIC d-recovery experiment.

The TLG analog of the equilibrium AIC d-selection experiment. There, AIC over
candidate neighborhood depths d is known to **collapse to d_hat=0** — the offset
model (only sigma free) cannot tell d>=1 apart from d=0. The temporal model fits a
free degree coefficient alpha *at depth d*, so the candidate designs differ in fit
and AIC can actually identify the true d.

Mechanism (one replicate):
  * grow a TLG graph at the true depth d_true (storing snapshots);
  * for each candidate d in D_GRID, rebuild the at-risk logistic design from the
    SAME snapshots at that d, fit (sigma_hat, alpha_hat), read off AIC;
  * d_hat = argmin_d AIC.  (k=2 for every candidate, so AIC ranking == fit ranking.)
Repeating over replicates and n gives the recovery accuracy P(d_hat = d_true) and the
(d_true, d_hat) confusion matrix.

Output under runs/tlg_aic_d/ (gitignored):
  - aic_<dtrue>_n<n>.npy   cached per-cell AIC matrices (reps x len(D_GRID))
  - aic_d_long.csv         tidy accuracy / confusion rows
  - aic_accuracy.png       recovery accuracy vs n (one line per d_true)
  - aic_confusion.png      (d_true x d_hat) confusion matrices, one per n

Env knobs (all optional):
  LG_TLGAIC_QUICK (0)      1 -> small grids
  LG_TLGAIC_SEED (7000)    LG_TLGAIC_JOBS (cpu-2)
  LG_TLGAIC_NREPS (20)     replicates per (d_true, n) cell
  LG_TLGAIC_NSTEPS (5)     growth steps per generated graph
  LG_TLGAIC_SIGMA (-2.0)   LG_TLGAIC_ALPHA (0.05)
  LG_TLGAIC_DTRUE (0,1,2)  true depths to test
  LG_TLGAIC_DGRID (0,1,2)  candidate depths for AIC
  LG_TLGAIC_NS (50,250,500)
  LG_TLGAIC_USE_CACHE (1)

  make tlg-aic-d           full run
  make tlg-aic-d-quick     smoke
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from logit_graph.temporal import (  # noqa: E402
    grow_graph, growth_design_from_snapshots, fit_growth_params,
)

OUT_DIR = _here / "runs" / "tlg_aic_d"


def _int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _float(name, default):
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _ints(name, default):
    raw = os.environ.get(name)
    return [int(x) for x in raw.split(",")] if raw else default


QUICK = os.environ.get("LG_TLGAIC_QUICK", "0") == "1"
SEED = _int("LG_TLGAIC_SEED", 7000)
NREPS = _int("LG_TLGAIC_NREPS", 5 if QUICK else 20)
NSTEPS = _int("LG_TLGAIC_NSTEPS", 5)
SIGMA = _float("LG_TLGAIC_SIGMA", -2.0)
ALPHA = _float("LG_TLGAIC_ALPHA", 0.05)
DTRUE = _ints("LG_TLGAIC_DTRUE", [0, 1, 2])
DGRID = _ints("LG_TLGAIC_DGRID", [0, 1, 2])
NS = _ints("LG_TLGAIC_NS", [50, 250] if QUICK else [50, 250, 500])
JOBS = _int("LG_TLGAIC_JOBS", max(1, (os.cpu_count() or 4) - 2))
USE_CACHE = os.environ.get("LG_TLGAIC_USE_CACHE", "1") == "1"


def aic_vector(job):
    """One replicate -> AIC for each candidate d (argmin gives d_hat)."""
    n, d_true, seed = job["n"], job["d_true"], job["seed"]
    res = grow_graph(n, d=d_true, sigma=SIGMA, alpha=ALPHA, n_steps=NSTEPS,
                     seed=seed, store_snapshots=True)
    out = []
    for de in DGRID:
        X, y = growth_design_from_snapshots(res.snapshots, d=de)
        out.append(fit_growth_params(X, y)["aic"])
    return np.asarray(out, dtype=float)


def collect_cell(d_true, n):
    """AIC matrix (NREPS x len(DGRID)) for one (d_true, n) cell (cached)."""
    path = OUT_DIR / f"aic_d{d_true}_n{n}.npy"
    if USE_CACHE and path.is_file():
        cached = np.load(path)
        if cached.shape == (NREPS, len(DGRID)):
            return cached
    jobs = [{"n": n, "d_true": d_true, "seed": SEED + 1000 * d_true + 7 * r + n}
            for r in range(NREPS)]
    if JOBS <= 1:
        mat = np.vstack([aic_vector(j) for j in jobs])
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=JOBS) as pool:
            mat = np.vstack(list(pool.map(aic_vector, jobs, chunksize=1)))
    np.save(path, mat)
    return mat


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"TLG AIC d-recovery  quick={QUICK} seed={SEED} jobs={JOBS} nreps={NREPS} "
          f"nsteps={NSTEPS}\n  sigma={SIGMA} alpha={ALPHA}  d_true={DTRUE} "
          f"d_grid={DGRID}  n={NS}")
    rows = []
    for d_true in DTRUE:
        for n in NS:
            mat = collect_cell(d_true, n)
            d_hat = np.array([DGRID[i] for i in mat.argmin(axis=1)])
            acc = float(np.mean(d_hat == d_true))
            for de in DGRID:
                rows.append({"d_true": d_true, "n": n, "d_hat": de,
                             "frac": float(np.mean(d_hat == de)),
                             "accuracy": acc, "n_reps": NREPS})
            print(f"  d_true={d_true} n={n:4d}: accuracy={acc:.2f}  "
                  f"d_hat dist={[round(float(np.mean(d_hat==de)),2) for de in DGRID]}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "aic_d_long.csv", index=False)
    _plot_accuracy(df, OUT_DIR / "aic_accuracy.png")
    _plot_confusion(df, OUT_DIR / "aic_confusion.png")
    print(f"\nWrote {OUT_DIR}/ (aic_d_long.csv, aic_accuracy.png, aic_confusion.png)")


# Colorblind-safe (Okabe-Ito) + distinct markers.
CB_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
CB_MARK = ["o", "s", "^", "D", "v", "P"]


def _plot_accuracy(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    acc = df.drop_duplicates(["d_true", "n"])[["d_true", "n", "accuracy"]]
    d_trues = sorted(acc["d_true"].unique())
    ns = sorted(acc["n"].unique())
    for i, dt in enumerate(d_trues):
        sub = acc[acc["d_true"] == dt].sort_values("n")
        ax.plot(sub["n"], sub["accuracy"], color=CB_COLORS[i % len(CB_COLORS)],
                marker=CB_MARK[i % len(CB_MARK)], ms=7, lw=2.0,
                markeredgecolor="white", markeredgewidth=0.6,
                label=f"$d_{{\\mathrm{{true}}}}={dt}$")
    chance = 1.0 / len(DGRID)
    ax.axhline(chance, color="#888888", ls=":", lw=1.2,
               label=f"chance (1/{len(DGRID)})")
    ax.set_xscale("log"); ax.set_xticks(ns)
    ax.set_xticklabels([str(v) for v in ns])
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("n (nodes)")
    ax.set_ylabel(r"AIC recovery accuracy  $P(\hat{d}=d_{\mathrm{true}})$")
    ax.set_title(f"TLG AIC d-recovery (σ={SIGMA:g}, α={ALPHA:g}; "
                 f"candidates d∈{{{','.join(map(str, DGRID))}}})")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_confusion(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ns = sorted(df["n"].unique())
    d_trues = sorted(df["d_true"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(4.0 * len(ns), 4.0),
                             squeeze=False)
    im = None
    for ci, n in enumerate(ns):
        ax = axes[0][ci]
        M = np.zeros((len(d_trues), len(DGRID)))
        for ri, dt in enumerate(d_trues):
            for cj, de in enumerate(DGRID):
                v = df[(df.n == n) & (df.d_true == dt) & (df.d_hat == de)]["frac"]
                M[ri, cj] = float(v.iloc[0]) if len(v) else 0.0
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        # overall accuracy = mean of the diagonal (true == est) entries
        diag = [M[ri, DGRID.index(dt)] for ri, dt in enumerate(d_trues)
                if dt in DGRID]
        acc = 100.0 * np.mean(diag) if diag else 0.0
        ax.set_title(f"$n = {n}$  (overall accuracy $= {acc:.0f}\\%$)", fontsize=12)
        ax.set_xticks(range(len(DGRID))); ax.set_xticklabels(DGRID)
        ax.set_yticks(range(len(d_trues))); ax.set_yticklabels(d_trues)
        ax.set_xlabel(r"Selected $\hat{d} = \arg\min\,\mathrm{AIC}(d_{\mathrm{est}})$")
        if ci == 0:
            ax.set_ylabel(r"True $d_{\mathrm{true}}$")
        for ri in range(len(d_trues)):
            for cj in range(len(DGRID)):
                ax.text(cj, ri, f"{100 * M[ri, cj]:.0f}%", ha="center", va="center",
                        color="white" if M[ri, cj] > 0.5 else "#333333",
                        fontsize=10, fontweight="bold" if M[ri, cj] > 0.5 else "normal")
        # box the diagonal (correct-recovery) cells
        for ri, dt in enumerate(d_trues):
            if dt in DGRID:
                cj = DGRID.index(dt)
                ax.add_patch(Rectangle((cj - 0.5, ri - 0.5), 1, 1, fill=False,
                                       edgecolor="black", lw=2.0, zorder=5))
        for sp in ("top", "right", "left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)

    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.02,
                 label=r"$\mathbb{P}(\hat{d} = d_{\mathrm{est}} \mid d_{\mathrm{true}})$")
    fig.suptitle(r"AIC-based selection of $d$: accuracy improves with graph size $n$",
                 fontsize=14, y=1.02)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
