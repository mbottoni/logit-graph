#!/usr/bin/env python3
"""Identifiability / parameter-recovery experiment for the UNIFIED latent TLG.

This is the recovery experiment of ``run_tlg_recovery.py`` extended from the degree-only
TLG to the unified model that wins the multi-dataset GIC comparison
(``scripts/closedform/run_tlg_multidataset_gic.py``):

    logit P[edge_ij at t] = sigma + alpha * D_ij(t-1)        (degree, predetermined)
                                  + gamma_c * Bc_ij           (coarse-community indicator)
                                  + gamma_f * Bf_ij           (fine-community indicator)
                                  + lambda  * L_ij            (latent-embedding proximity)

The whole point of the winning model is that EVERY feature is identifiable: the degree
feature is read from the *previous* snapshot (predetermined), and the community indicators
and latent feature are FIXED exogenous covariates. So in the add+remove Bernoulli model
(every dyad resampled each step from the lagged probability) each draw is, conditional on
the past, an independent Bernoulli and the pooled dyad design is an ordinary logistic
regression whose MLE recovers (sigma, alpha, gamma_c, gamma_f, lambda) consistently — no
degeneracy (there is NO endogenous clustering term, which is exactly what would make an
ERGM-style feature non-identifiable).

This experiment makes that claim falsifiable: generate add+remove temporal graphs with a
KNOWN parameter vector and synthetic but fixed exogenous covariates (two random community
partitions + a random latent embedding, standing in for the Louvain partitions and the
adjacency spectral embedding used on real graphs), estimate all five coefficients by pooled
MLE, and show the estimates converge to truth while their spread shrinks ~1/sqrt(n).

Output under runs/tlg_latent_identifiability/ (gitignored):
  - <scenario>/recovery_raw.csv   per-replicate estimates (cache + custom re-plots)
  - <scenario>/recovery.csv       tidy per-scenario summary (mean/std/CI per (n, param))
  - <scenario>/results.json       per-scenario config + summary
  - recovery_all.csv              combined summary over all scenarios
  - identifiability.png           one figure: a panel per parameter (estimate -> truth as
                                  n grows; color+marker = scenario; dashed = truth) plus a
                                  log-log consistency panel (std vs n with a 1/sqrt(n) ref).

Re-plotting: per-scenario results are cached. Re-running with LG_TLI_USE_CACHE=1 (default)
reloads the caches and only regenerates the figure. Force a fresh run with
LG_TLI_USE_CACHE=0.

Env knobs (all optional):
  LG_TLI_SEED (12345)     LG_TLI_QUICK (0 -> full; 1 -> 1 scenario, small n)
  LG_TLI_NREPS (12)       replicates per (scenario, n)
  LG_TLI_NSTEPS (5)       add+remove steps per generated graph
  LG_TLI_KC (4)           coarse-partition block count   LG_TLI_KF (16) fine-partition
  LG_TLI_KLAT (4)         latent embedding dimension
  LG_TLI_USE_CACHE (1)

  make tlg-identifiability        full run
  make tlg-identifiability-quick  smoke (1 scenario, small n)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from logit_graph.lg_features import build_pair_dataset  # noqa: E402


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


QUICK = os.environ.get("LG_TLI_QUICK", "0") == "1"
SEED = _int("LG_TLI_SEED", 12345)
NREPS = _int("LG_TLI_NREPS", 3 if QUICK else 12)
NSTEPS = _int("LG_TLI_NSTEPS", 3 if QUICK else 5)
KC = _int("LG_TLI_KC", 4)            # coarse-community block count
KF = _int("LG_TLI_KF", 16)           # fine-community block count
KLAT = _int("LG_TLI_KLAT", 4)        # latent-embedding dimension
USE_CACHE = os.environ.get("LG_TLI_USE_CACHE", "1") == "1"
DEG_D = 1

NS = [50, 100] if QUICK else [50, 100, 200, 400, 800]

# The five coefficients of the unified model, in design-column order.
PARAMS = ("sigma", "alpha", "gamma_c", "gamma_f", "lam")
LABEL = {"sigma": r"$\hat{\sigma}$ (intercept)",
         "alpha": r"$\hat{\alpha}$ (degree)",
         "gamma_c": r"$\hat{\gamma}_c$ (coarse comm.)",
         "gamma_f": r"$\hat{\gamma}_f$ (fine comm.)",
         "lam": r"$\hat{\lambda}$ (latent)"}

# Distinct ground-truth scenarios (one color/marker each). Coefficients are moderate:
# the exogenous covariates carry no endogenous feedback, so there is no ERGM-style
# degeneracy and a wide range recovers — these just keep densities reasonable.
SCENARIOS = (
    [dict(sigma=-2.5, alpha=0.05, gamma_c=1.0, gamma_f=1.5, lam=1.0)]
    if QUICK else
    [dict(sigma=-2.5, alpha=0.05, gamma_c=1.0, gamma_f=1.5, lam=1.0),
     dict(sigma=-3.0, alpha=0.08, gamma_c=0.8, gamma_f=2.0, lam=0.6),
     dict(sigma=-2.0, alpha=0.03, gamma_c=1.5, gamma_f=1.0, lam=1.4)]
)

OUT_DIR = _here / "runs" / "tlg_latent_identifiability"


def _scenario_tag(p):
    return ("s{sigma:g}_a{alpha:g}_gc{gamma_c:g}_gf{gamma_f:g}_l{lam:g}"
            .format(**p).replace("-", "m").replace(".", "p"))


# ---------------------------------------------------------------------------
# Fixed exogenous covariates (synthetic analogues of the real-graph features)
# ---------------------------------------------------------------------------

def make_covariates(n, seed):
    """Two same-community indicators (coarse/fine random partitions) and a latent
    proximity feature (random rank-KLAT embedding, dot product, standardized) over all
    upper-triangle pairs. All fixed for the whole trajectory -> exogenous covariates."""
    rng = np.random.default_rng(seed)
    rows, cols = np.triu_indices(n, k=1)
    coarse = rng.integers(0, KC, n)
    fine = rng.integers(0, KF, n)
    z = rng.normal(size=(n, KLAT))
    Bc = (coarse[rows] == coarse[cols]).astype(float)
    Bf = (fine[rows] == fine[cols]).astype(float)
    L = (z[rows] * z[cols]).sum(1)
    L = (L - L.mean()) / (L.std() + 1e-9)
    return rows, cols, Bc, Bf, L


# ---------------------------------------------------------------------------
# Generation (add+remove Bernoulli chain) + estimation (pooled MLE)
# ---------------------------------------------------------------------------

def generate(n, truth, covars, seed, p0=0.02):
    """Add+remove temporal graph: every step resample ALL dyads from the lagged
    probability expit(sigma + alpha*D(t-1) + gc*Bc + gf*Bf + lam*L). D is the degree
    feature of the previous snapshot (predetermined); Bc/Bf/L are fixed. Returns the
    pooled design X=[D, Bc, Bf, L] and outcomes y over all dyads x steps."""
    rows, cols, Bc, Bf, L = covars
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n))
    seed_mask = rng.random(rows.shape[0]) < p0
    adj[rows[seed_mask], cols[seed_mask]] = 1.0
    adj[cols[seed_mask], rows[seed_mask]] = 1.0
    Xs, ys = [], []
    for _ in range(NSTEPS):
        D, _lab = build_pair_dataset(adj, d=DEG_D, mode="bounded", layer2=True)
        D = np.asarray(D, dtype=np.float64)
        lo = (truth["sigma"] + truth["alpha"] * D + truth["gamma_c"] * Bc
              + truth["gamma_f"] * Bf + truth["lam"] * L)
        draw = rng.random(lo.shape[0]) < expit(lo)
        Xs.append(np.column_stack([D, Bc, Bf, L]))
        ys.append(draw.astype(np.int8))
        adj[:] = 0.0
        adj[rows[draw], cols[draw]] = 1.0
        adj[cols[draw], rows[draw]] = 1.0
    return np.vstack(Xs), np.concatenate(ys)


def fit_mle(X, y):
    """Pooled logistic-regression MLE on [1, D, Bc, Bf, L] (the exact MLE for the
    add+remove model). Solver fallbacks for robustness."""
    Xc = sm.add_constant(X, has_constant="add")
    res = None
    for method in ("newton", "bfgs", "lbfgs"):
        try:
            r = sm.Logit(y, Xc).fit(method=method, disp=0, maxiter=300)
            if np.isfinite(r.llf):
                res = r
                break
        except Exception:
            continue
    if res is None:
        res = sm.Logit(y, Xc).fit_regularized(method="l1", alpha=1e-4, disp=0)
    b = np.asarray(res.params, dtype=float)
    return dict(sigma=b[0], alpha=b[1], gamma_c=b[2], gamma_f=b[3], lam=b[4])


# ---------------------------------------------------------------------------
# Simulation + aggregation
# ---------------------------------------------------------------------------

def simulate(truth) -> pd.DataFrame:
    rows = []
    for n in NS:
        t0 = time.perf_counter()
        for rep in range(NREPS):
            seed = SEED + 7 * rep + n
            covars = make_covariates(n, seed=seed)
            X, y = generate(n, truth, covars, seed=seed)
            est = fit_mle(X, y)
            rows.append(dict(n=n, rep=rep, **est))
        print(f"    n={n:5d}: {NREPS} reps in {time.perf_counter()-t0:5.1f}s")
    return pd.DataFrame(rows)


def aggregate(raw, truth) -> pd.DataFrame:
    out = []
    for n in NS:
        cell = raw[raw["n"] == n]
        for p in PARAMS:
            vals = cell[p].to_numpy()
            vals = vals[np.isfinite(vals)]
            mean = float(np.mean(vals)) if len(vals) else float("nan")
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out.append(dict(n=n, param=p, true=float(truth[p]), mean=mean, std=std,
                            ci_lo=mean - 1.96 * std, ci_hi=mean + 1.96 * std,
                            n_reps=int(len(vals))))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Plot: a panel per parameter + a consistency (std vs n) panel
# ---------------------------------------------------------------------------

def plot_identifiability(combined, scenarios, out_path):
    cb_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00",
                 "#000000", "#56B4E9"]
    cb_markers = ["o", "s", "^", "D", "v", "P", "X"]
    tags = [_scenario_tag(s) for s in scenarios]
    colors = {t: cb_colors[i % len(cb_colors)] for i, t in enumerate(tags)}
    marks = {t: cb_markers[i % len(cb_markers)] for i, t in enumerate(tags)}

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    flat = axes.flatten()
    for pi, p in enumerate(PARAMS):
        ax = flat[pi]
        for s in scenarios:
            tag = _scenario_tag(s)
            sub = combined[(combined["scenario"] == tag) &
                           (combined["param"] == p)].sort_values("n")
            x = sub["n"].to_numpy()
            ax.fill_between(x, sub["ci_lo"], sub["ci_hi"], color=colors[tag],
                            alpha=0.08, zorder=1)
            ax.plot(x, sub["mean"], marker=marks[tag], ms=5, color=colors[tag],
                    lw=1.6, zorder=3)
            ax.axhline(s[p], color=colors[tag], ls="--", lw=0.9, alpha=0.6, zorder=2)
        ax.set_xscale("log"); ax.set_xticks(NS)
        ax.set_xticklabels([str(v) for v in NS], rotation=45, fontsize=8)
        ax.set_ylabel(LABEL[p]); ax.set_xlabel("n (nodes)"); ax.grid(alpha=0.25)
        ax.set_title(LABEL[p])

    # consistency panel: std(estimate) vs n (log-log), all params/scenarios, + 1/sqrt(n) ref
    ax = flat[5]
    for s in scenarios:
        tag = _scenario_tag(s)
        for p in PARAMS:
            sub = combined[(combined["scenario"] == tag) &
                           (combined["param"] == p)].sort_values("n")
            ax.plot(sub["n"], sub["std"], color=colors[tag], lw=0.8, alpha=0.5)
    n0 = NS[0]
    ref = np.array(NS, dtype=float)
    ax.plot(ref, (ref / n0) ** -0.5 * 0.3, color="k", ls=":", lw=1.5,
            label=r"$\propto 1/\sqrt{n}$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(NS)
    ax.set_xticklabels([str(v) for v in NS], rotation=45, fontsize=8)
    ax.set_xlabel("n (nodes)"); ax.set_ylabel("std of estimate")
    ax.set_title("consistency: estimate spread shrinks with n"); ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    handles = [Line2D([0], [0], color=colors[_scenario_tag(s)],
                      marker=marks[_scenario_tag(s)], lw=1.6,
                      label="σ={sigma:g}, α={alpha:g}, γc={gamma_c:g}, "
                            "γf={gamma_f:g}, λ={lam:g}".format(**s))
               for s in scenarios]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Unified LG identifiability — all five coefficients recover by "
                 "MLE as n grows (dashed = truth)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"TLG latent identifiability  seed={SEED}  quick={QUICK}  n={NS}  "
          f"reps={NREPS}  steps={NSTEPS}  kc={KC} kf={KF} klat={KLAT}\n"
          f"  scenarios={len(SCENARIOS)}")
    all_summaries = []
    for truth in SCENARIOS:
        tag = _scenario_tag(truth)
        sdir = OUT_DIR / tag
        sdir.mkdir(parents=True, exist_ok=True)
        raw_path = sdir / "recovery_raw.csv"
        print(f"\n[scenario] {truth}  -> {tag}")
        if USE_CACHE and raw_path.exists():
            print(f"  using cached {raw_path.name}")
            raw = pd.read_csv(raw_path)
        else:
            raw = simulate(truth)
            raw.to_csv(raw_path, index=False)
        summary = aggregate(raw, truth)
        summary.to_csv(sdir / "recovery.csv", index=False)
        (sdir / "results.json").write_text(json.dumps(
            {"true": truth, "ns": NS, "n_reps": NREPS, "n_steps": NSTEPS,
             "kc": KC, "kf": KF, "klat": KLAT, "seed": SEED,
             "summary": summary.to_dict(orient="records")}, indent=2, default=float))
        summary.insert(0, "scenario", tag)
        all_summaries.append(summary)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(OUT_DIR / "recovery_all.csv", index=False)
    plot_identifiability(combined, SCENARIOS, OUT_DIR / "identifiability.png")

    print("\n" + "=" * 70)
    for truth in SCENARIOS:
        tag = _scenario_tag(truth)
        sub = combined[(combined["scenario"] == tag) & (combined["n"] == NS[-1])]
        msg = "  ".join(f"{r.param}={r.mean:+.3f}±{1.96*r.std:.3f}(t{r.true:+.2f})"
                        for r in sub.itertuples())
        print(f"[{tag}] at n={NS[-1]}: {msg}")
    print(f"\nWrote {OUT_DIR}/identifiability.png + per-scenario CSVs + recovery_all.csv")


if __name__ == "__main__":
    main()
