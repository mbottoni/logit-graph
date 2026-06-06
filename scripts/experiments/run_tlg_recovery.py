#!/usr/bin/env python3
"""Temporal Logit-Graph (TLG) parameter-recovery experiment.

Generates growth graphs with KNOWN ground truth (sigma, alpha) for d in {0,1,2}
and n in {10,20,50,75,100,150,500,1000,1500}, estimates (sigma, alpha) by logistic regression on
the at-risk dyads, and shows the estimates converge to the truth as n grows.
Several (sigma, alpha) scenarios are swept and overlaid in ONE figure.

Output under runs/tlg_recovery/ (gitignored):
  - <scenario>/recovery_raw.csv   per-replicate estimates (cache + custom re-plots)
  - <scenario>/recovery.csv       tidy per-scenario summary
  - <scenario>/results.json       per-scenario config + summary
  - recovery_all.csv              combined summary over all scenarios
  - recovery.png                  ONE figure: rows = (sigma, alpha), cols = d.
                                  Each (sigma,alpha) scenario is one COLOR, the same
                                  across every subplot; solid line = mean estimate,
                                  shaded = 95% interval, dashed = that scenario's true
                                  value; x = n (log).

Re-plotting: per-scenario results are cached. Re-running with LG_TLG_USE_CACHE=1
(default) reloads the caches and only regenerates recovery.png, so styling tweaks
are cheap. Force a fresh run with LG_TLG_USE_CACHE=0.

Env knobs (all optional):
  LG_TLG_SEED (12345)     LG_TLG_QUICK (0 -> full; 1 -> 2 scenarios, d={0,1}, small n)
  LG_TLG_NREPS (12)       replicates per (d, n) cell
  LG_TLG_NSTEPS (4)       growth steps per generated graph
  LG_TLG_SIGMAS (-2,-3,-4,-5,-6)  true intercepts; PAIRED index-wise with ALPHAS
  LG_TLG_ALPHAS (0.04,0.06,0.08,0.10,0.12)  true degree coeffs; one distinct pair each
  LG_TLG_USE_CACHE (1)    1 -> reload + replot if cache exists; 0 -> always simulate

  make tlg-recovery         full run (scenario grid)
  make tlg-recovery-quick   smoke (small grid, d={0,1}, n={10,50})
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from logit_graph.temporal import grow_graph, fit_growth_from_result  # noqa: E402


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


def _floats(env, default):
    raw = os.environ.get(env)
    return [float(x) for x in raw.split(",")] if raw else default


QUICK = os.environ.get("LG_TLG_QUICK", "0") == "1"
SEED = _int("LG_TLG_SEED", 12345)
NREPS = _int("LG_TLG_NREPS", 3 if QUICK else 12)
NSTEPS = _int("LG_TLG_NSTEPS", 3 if QUICK else 4)
# Scenarios are DISTINCT (sigma, alpha) PAIRS (zipped index-wise) — every sigma and
# every alpha is unique across scenarios, so no two scenarios share a value/dashed line.
SIGMAS = _floats("LG_TLG_SIGMAS",
                 [-2.0, -4.0] if QUICK else [-2.0, -3.0, -4.0, -5.0, -6.0])
ALPHAS = _floats("LG_TLG_ALPHAS",
                 [0.05, 0.10] if QUICK else [0.04, 0.06, 0.08, 0.10, 0.12])
USE_CACHE = os.environ.get("LG_TLG_USE_CACHE", "1") == "1"

DS = [0, 1] if QUICK else [0, 1, 2]
NS = [10, 50] if QUICK else [10, 20, 50, 75, 100, 150, 500, 1000, 1500]

PARAMS = ("sigma", "alpha")
LABEL = {"sigma": r"$\hat{\sigma}$", "alpha": r"$\hat{\alpha}$ (degree)"}
OUT_DIR = _here / "runs" / "tlg_recovery"


def _scenario_dir(sigma, alpha):
    tag = f"s{sigma:g}_a{alpha:g}".replace("-", "m").replace(".", "p")
    return OUT_DIR / tag


# ---------------------------------------------------------------------------
# Simulation + aggregation
# ---------------------------------------------------------------------------

def simulate(sigma, alpha) -> pd.DataFrame:
    rows = []
    for d in DS:
        for n in NS:
            t0 = time.perf_counter()
            for rep in range(NREPS):
                seed = SEED + 1000 * d + 7 * rep + n
                res = grow_graph(n, d=d, sigma=sigma, alpha=alpha,
                                 n_steps=NSTEPS, seed=seed, store_snapshots=False)
                out = fit_growth_from_result(res)
                rows.append(dict(d=d, n=n, rep=rep,
                                 sigma=out["sigma"], alpha=out["alpha"],
                                 se_sigma=out["se_sigma"], se_alpha=out["se_alpha"],
                                 density=float(res.adj.sum() / (n * (n - 1)))))
            print(f"    d={d} n={n:5d}: {NREPS} reps in {time.perf_counter()-t0:5.1f}s")
            sys.stdout.flush()
    return pd.DataFrame(rows)


def aggregate(raw, sigma, alpha) -> pd.DataFrame:
    true = {"sigma": sigma, "alpha": alpha}
    out = []
    for d in DS:
        for n in NS:
            cell = raw[(raw["d"] == d) & (raw["n"] == n)]
            for p in PARAMS:
                vals = cell[p].to_numpy()
                vals = vals[np.isfinite(vals)]
                mean = float(np.mean(vals)) if len(vals) else float("nan")
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                out.append(dict(d=d, n=n, param=p, true=true[p], mean=mean, std=std,
                                ci_lo=mean - 1.96 * std, ci_hi=mean + 1.96 * std,
                                n_reps=int(len(vals)),
                                density_mean=float(cell["density"].mean())))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Combined plot: all scenarios in one figure (color = scenario)
# ---------------------------------------------------------------------------

def plot_combined_recovery(combined, scenarios, out_path):
    # Colorblind-safe (Okabe-Ito) colors + distinct markers, so scenarios are
    # distinguishable by BOTH hue and shape (accessible for color-vision deficiency).
    cb_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
                 "#000000", "#56B4E9"]
    cb_markers = ["o", "s", "^", "D", "v", "P", "X"]
    colors = {sc: cb_colors[i % len(cb_colors)] for i, sc in enumerate(scenarios)}
    marks = {sc: cb_markers[i % len(cb_markers)] for i, sc in enumerate(scenarios)}
    sig_vals = [s for s, _ in scenarios]
    alp_vals = [a for _, a in scenarios]
    ylim = {"sigma": (min(sig_vals) - 2.0, max(sig_vals) + 2.0),
            "alpha": (min(alp_vals) - 0.15, max(alp_vals) + 0.15)}

    fig, axes = plt.subplots(2, len(DS), figsize=(4.8 * len(DS), 8),
                             sharex=True, squeeze=False)
    for (sigma, alpha) in scenarios:
        c = colors[(sigma, alpha)]
        mk = marks[(sigma, alpha)]
        true = {"sigma": sigma, "alpha": alpha}
        sc = combined[(combined["true_sigma"] == sigma) &
                      (combined["true_alpha"] == alpha)]
        for ri, p in enumerate(PARAMS):
            for ci, d in enumerate(DS):
                ax = axes[ri][ci]
                sub = sc[(sc["param"] == p) & (sc["d"] == d)].sort_values("n")
                x = sub["n"].to_numpy()
                ax.fill_between(x, sub["ci_lo"], sub["ci_hi"], color=c,
                                alpha=0.08, zorder=1)
                ax.plot(x, sub["mean"], marker=mk, ms=5, color=c, lw=1.6, zorder=3)
                ax.axhline(true[p], color=c, ls="--", lw=0.9, alpha=0.6, zorder=2)

    for ri, p in enumerate(PARAMS):
        for ci, d in enumerate(DS):
            ax = axes[ri][ci]
            ax.set_xscale("log"); ax.set_xticks(NS)
            ax.set_xticklabels([str(v) for v in NS], rotation=45, fontsize=8)
            ax.set_ylim(*ylim[p]); ax.grid(alpha=0.25)
            if ri == 0:
                ax.set_title(f"d = {d}")
            if ci == 0:
                ax.set_ylabel(LABEL[p])
            if ri == len(PARAMS) - 1:
                ax.set_xlabel("n (nodes)")

    handles = [Line2D([0], [0], color=colors[sc], marker=marks[sc], lw=1.6,
                      label=f"σ={sc[0]:g}, α={sc[1]:g}") for sc in scenarios]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(scenarios), 5), fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("TLG parameter recovery — estimates → true value as n grows "
                 "(colorblind-safe: color+marker = (σ,α) scenario; dashed = truth)")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = list(zip(SIGMAS, ALPHAS))  # distinct (sigma, alpha) pairs
    print(f"TLG recovery  seed={SEED}  quick={QUICK}  d={DS}  n={NS}  reps={NREPS}  "
          f"steps={NSTEPS}\n  scenarios (sigma,alpha)={scenarios}")
    all_summaries = []
    for sigma, alpha in scenarios:
        sdir = _scenario_dir(sigma, alpha)
        sdir.mkdir(parents=True, exist_ok=True)
        raw_path = sdir / "recovery_raw.csv"
        print(f"\n[scenario] sigma={sigma:g} alpha={alpha:g}  -> {sdir.name}")
        if USE_CACHE and raw_path.exists():
            print(f"  using cached {raw_path.name}")
            raw = pd.read_csv(raw_path)
        else:
            raw = simulate(sigma, alpha)
            raw.to_csv(raw_path, index=False)
        summary = aggregate(raw, sigma, alpha)
        summary.to_csv(sdir / "recovery.csv", index=False)
        (sdir / "results.json").write_text(json.dumps(
            {"true": {"sigma": sigma, "alpha": alpha}, "ds": DS, "ns": NS,
             "n_reps": NREPS, "n_steps": NSTEPS, "seed": SEED,
             "summary": summary.to_dict(orient="records")}, indent=2, default=float))
        summary.insert(0, "true_alpha", alpha)
        summary.insert(0, "true_sigma", sigma)
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(OUT_DIR / "recovery_all.csv", index=False)
    plot_combined_recovery(combined, scenarios, OUT_DIR / "recovery.png")

    print("\n" + "=" * 70)
    for sigma, alpha in scenarios:
        sub = combined[(combined["true_sigma"] == sigma) &
                       (combined["true_alpha"] == alpha)]
        big = sub[sub["n"] == NS[-1]]
        msg = "  ".join(f"d={int(r.d)} {r.param}={r.mean:+.3f}±{1.96*r.std:.3f}"
                        for r in big.itertuples())
        print(f"[σ={sigma:g}, α={alpha:g}] at n={NS[-1]}: {msg}")
    print(f"\nWrote {OUT_DIR}/recovery.png + per-scenario CSVs + recovery_all.csv")


if __name__ == "__main__":
    main()
