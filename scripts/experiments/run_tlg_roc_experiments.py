#!/usr/bin/env python3
"""Temporal Logit-Graph (TLG) ROC experiments — group-difference tests on (sigma, alpha).

The TLG analog of the equilibrium ROC sweeps (run_lg_roc_experiments.py). There the
test detects whether two groups of graphs differ in sigma; here we test BOTH temporal
parameters — the intercept sigma AND the degree coefficient alpha — under the growth
model.

Test (one Monte-Carlo experiment). Two methods (env LG_TLGROC_METHOD):
  * "wald" (default): ONE growth graph per group; fit by logistic regression ->
    (theta_hat, se) for the tested parameter; two-sample Wald z-test using the MLE's
    own standard errors, z = (th1 - th2)/sqrt(se1^2 + se2^2), p = 2(1 - Phi(|z|)).
    No replicates needed — the logistic regression already gives the SE. (Verified
    well-calibrated: the null rejection rate tracks the significance level even with
    the design pooled across growth steps.)
  * "anova": ``n_reps`` graphs per group; one-way ANOVA (scipy f_oneway) on the
    tested parameter's estimates -> p-value (the empirical-variance analog, kept for
    comparison).
Repeating over ``n_experiments`` gives a p-value distribution; the "ROC" plots the
rejection rate P(p < level) vs the significance level (the p-value CDF), with the
null (no difference) sitting on the chance diagonal — the same presentation as the LG
figures.

Two variants per parameter (each paneled by d):
  * effect-size: fixed n = ``n_effect``, one curve per group-2 value (effect size),
    including the null;
  * sample-size: fixed effect, one curve per n.

Output under runs/tlg_roc/ (gitignored):
  - pvals_*.npy            cached per-cell p-value arrays (resumable, cheap re-plots)
  - roc_long.csv           tidy rejection-rate-vs-level rows
  - roc_effect.png         rows = {sigma-test, alpha-test}, cols = d
  - roc_sample.png         rows = {sigma-test, alpha-test}, cols = d

Env knobs (all optional):
  LG_TLGROC_QUICK (0)        1 -> d={0}, tiny grids/experiments
  LG_TLGROC_METHOD (wald)    "wald" (single-graph SE) or "anova" (n_reps replicates)
  LG_TLGROC_SEED (4000)      LG_TLGROC_JOBS (cpu-2)   process pool over experiments
  LG_TLGROC_NREPS (10)       graphs per group (anova method only)
  LG_TLGROC_NEXP (200)       experiments per cell
  LG_TLGROC_NSTEPS (5)       growth steps per generated graph
  LG_TLGROC_DS (0,1,2)       degree-feature hops (panel columns)
  LG_TLGROC_NEFFECT (60)     fixed n for the effect-size variant
  LG_TLGROC_NS (20,40,60,100,160)  n grid for the sample-size variant
  LG_TLGROC_USE_CACHE (1)    1 -> reload cached cells + replot

  make tlg-roc              full run
  make tlg-roc-quick        smoke
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import stats

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from logit_graph.temporal import grow_graph, fit_growth_from_result  # noqa: E402

OUT_DIR = _here / "runs" / "tlg_roc"

# Base (group-1) parameters, shared by both tests.
SIGMA1 = -2.0
ALPHA1 = 0.05


def _int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _ints(name, default):
    raw = os.environ.get(name)
    return [int(x) for x in raw.split(",")] if raw else default


QUICK = os.environ.get("LG_TLGROC_QUICK", "0") == "1"
METHOD = os.environ.get("LG_TLGROC_METHOD", "wald")  # "wald" | "anova"
SEED = _int("LG_TLGROC_SEED", 4000)
NREPS = _int("LG_TLGROC_NREPS", 4 if QUICK else 10)
NEXP = _int("LG_TLGROC_NEXP", 40 if QUICK else 200)
NSTEPS = _int("LG_TLGROC_NSTEPS", 5)
DS = _ints("LG_TLGROC_DS", [0] if QUICK else [0, 1, 2])
N_EFFECT = _int("LG_TLGROC_NEFFECT", 60)
NS = _ints("LG_TLGROC_NS", [20, 40] if QUICK else [20, 40, 60, 100, 160])
JOBS = _int("LG_TLGROC_JOBS", max(1, (os.cpu_count() or 4) - 2))
USE_CACHE = os.environ.get("LG_TLGROC_USE_CACHE", "1") == "1"

# Per-parameter effect grids (group-2 values; first entry is the null = base).
# Calibrated so curves span from the chance diagonal to high power at the defaults.
SIGMA2_EFFECT = [-2.0, -2.05, -2.10, -2.15, -2.25] if not QUICK else [-2.0, -2.15]
ALPHA2_EFFECT = [0.05, 0.06, 0.07, 0.08, 0.10] if not QUICK else [0.05, 0.08]
SIGMA2_SAMPLE = -2.10   # fixed effect for the sigma sample-size variant (|d|=0.10)
ALPHA2_SAMPLE = 0.07    # fixed effect for the alpha sample-size variant (|d|=0.02)

LEVELS = np.linspace(0.0, 1.0, 201)
PARAMS = ("sigma", "alpha")
PARAM_IDX = {"sigma": 0, "alpha": 1}


def _fit_full(n, d, sigma, alpha, seed):
    res = grow_graph(n, d=d, sigma=sigma, alpha=alpha, n_steps=NSTEPS, seed=seed,
                     store_snapshots=False)
    return fit_growth_from_result(res)


def _group(n, d, sigma, alpha, seed0):
    est = [(o["sigma"], o["alpha"])
           for o in (_fit_full(n, d, sigma, alpha, seed0 + 7 * r) for r in range(NREPS))]
    return np.asarray(est)  # (NREPS, 2): columns = (sigma_hat, alpha_hat)


def experiment_pvalue(job):
    """One Monte-Carlo experiment -> p-value for the tested parameter difference."""
    param = job["param"]
    n, d, seed = job["n"], job["d"], job["seed"]
    if job["method"] == "wald":
        # One graph per group; two-sample Wald z-test using the logistic-regression
        # SEs directly (no replicates).
        o1 = _fit_full(n, d, SIGMA1, ALPHA1, seed)
        o2 = _fit_full(n, d, job["sigma2"], job["alpha2"], seed + 500_000)
        diff = o1[param] - o2[param]
        denom = float(np.hypot(o1["se_" + param], o2["se_" + param]))
        if not np.isfinite(denom) or denom == 0.0:
            return 1.0
        z = diff / denom
        return float(2.0 * stats.norm.sf(abs(z)))
    # anova: n_reps replicates per group, one-way ANOVA on the estimates
    g1 = _group(n, d, SIGMA1, ALPHA1, seed)
    g2 = _group(n, d, job["sigma2"], job["alpha2"], seed + 500_000)
    _, p = stats.f_oneway(g1[:, PARAM_IDX[param]], g2[:, PARAM_IDX[param]])
    return float(p)


def _cell_seed(param, sweep, d, key):
    # Deterministic (no str hash), distinct per cell. key = sigma2/alpha2 or n.
    pi = PARAMS.index(param)
    si = 0 if sweep == "effect" else 1
    k = int(round(float(key) * 1000))
    return SEED + ((pi * 2 + si) * 100_003 + d * 1009 + (k % 100_000) * 7) % (2**31 - 1)


def _cell_tag(param, sweep, d, *, sigma2=None, alpha2=None, n=None):
    if sweep == "effect":
        v = sigma2 if param == "sigma" else alpha2
        body = f"{param}_effect_d{d}_v{v:g}"
    else:
        body = f"{param}_sample_d{d}_n{n}"
    return f"{METHOD}_{body}".replace("-", "m").replace(".", "p")


def collect_pvalues(param, sweep, d, *, sigma2, alpha2, n):
    """p-values over NEXP experiments for one cell (cached to .npy)."""
    tag = _cell_tag(param, sweep, d, sigma2=sigma2, alpha2=alpha2, n=n)
    path = OUT_DIR / f"pvals_{tag}.npy"
    if USE_CACHE and path.is_file():
        cached = np.load(path)
        if len(cached) >= NEXP:
            return cached[:NEXP]

    if sweep == "effect":
        key = sigma2 if param == "sigma" else alpha2
    else:
        key = n
    base_seed = _cell_seed(param, sweep, d, key)
    jobs = [{"param": param, "n": n, "d": d, "sigma2": sigma2, "alpha2": alpha2,
             "method": METHOD, "seed": base_seed + 1000 * e} for e in range(NEXP)]

    if JOBS <= 1:
        pvals = np.array([experiment_pvalue(j) for j in jobs])
    else:
        from concurrent.futures import ProcessPoolExecutor
        pvals = np.empty(NEXP)
        with ProcessPoolExecutor(max_workers=JOBS) as pool:
            for i, p in enumerate(pool.map(experiment_pvalue, jobs, chunksize=4)):
                pvals[i] = p
    np.save(path, pvals)
    return pvals


def _roc_rows(pvals, *, param, sweep, d, effect_label, n):
    rates = [float(np.mean(pvals < t)) for t in LEVELS]
    power = float(np.mean(pvals < 0.05))
    return [{"param": param, "sweep": sweep, "d": d, "effect": effect_label, "n": n,
             "level": t, "rejection_rate": r, "power_at_005": power}
            for t, r in zip(LEVELS, rates)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"TLG ROC  method={METHOD} quick={QUICK} seed={SEED} jobs={JOBS} "
          f"nexp={NEXP} nsteps={NSTEPS}"
          + (f" nreps={NREPS}" if METHOD == "anova" else "")
          + f"\n  d={DS} n_effect={N_EFFECT} ns={NS}\n"
          f"  base sigma1={SIGMA1} alpha1={ALPHA1}")
    rows = []
    for param in PARAMS:
        effects = SIGMA2_EFFECT if param == "sigma" else ALPHA2_EFFECT
        base_val = SIGMA1 if param == "sigma" else ALPHA1
        # --- effect-size variant (fixed n = N_EFFECT) ---
        for d in DS:
            for v in effects:
                s2 = v if param == "sigma" else SIGMA1
                a2 = v if param == "alpha" else ALPHA1
                pv = collect_pvalues(param, "effect", d, sigma2=s2, alpha2=a2,
                                     n=N_EFFECT)
                lbl = abs(v - base_val)
                rows += _roc_rows(pv, param=param, sweep="effect", d=d,
                                  effect_label=lbl, n=N_EFFECT)
                print(f"  [{param} effect] d={d} v={v:g} |Δ|={lbl:.3g}: "
                      f"power@.05={np.mean(pv < 0.05):.2f}")
        # --- sample-size variant (fixed effect) ---
        s2 = SIGMA2_SAMPLE if param == "sigma" else SIGMA1
        a2 = ALPHA2_SAMPLE if param == "alpha" else ALPHA1
        fixed = abs((SIGMA2_SAMPLE - SIGMA1) if param == "sigma"
                    else (ALPHA2_SAMPLE - ALPHA1))
        for d in DS:
            for n in NS:
                pv = collect_pvalues(param, "sample", d, sigma2=s2, alpha2=a2, n=n)
                rows += _roc_rows(pv, param=param, sweep="sample", d=d,
                                  effect_label=fixed, n=n)
                print(f"  [{param} sample |Δ|={fixed:.3g}] d={d} n={n}: "
                      f"power@.05={np.mean(pv < 0.05):.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "roc_long.csv", index=False)
    _plot(df, "effect", OUT_DIR / "roc_effect.png")
    _plot(df, "sample", OUT_DIR / "roc_sample.png")
    print(f"\nWrote {OUT_DIR}/ (roc_long.csv, roc_effect.png, roc_sample.png)")


def _plot(df, sweep, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cb = ["#999999", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
    sub = df[df["sweep"] == sweep]
    if sub.empty:
        return
    ds = sorted(sub["d"].unique())
    fig, axes = plt.subplots(len(PARAMS), len(ds), figsize=(4.6 * len(ds), 8.4),
                             squeeze=False, sharex=True, sharey=True)
    series_key = "n" if sweep == "sample" else "effect"

    for ri, param in enumerate(PARAMS):
        psub = sub[sub["param"] == param]
        keys = sorted(psub[series_key].unique())
        colors = {k: cb[i % len(cb)] for i, k in enumerate(keys)}
        for ci, d in enumerate(ds):
            ax = axes[ri][ci]
            ax.plot([0, 1], [0, 1], color="#bbbbbb", ls=":", lw=1.2, zorder=1)
            for k in keys:
                c = psub[(psub["d"] == d) & (psub[series_key] == k)].sort_values("level")
                # the null effect (|Δ|=0) is the chance line; draw it thin/grey
                is_null = (sweep == "effect" and abs(k) < 1e-9)
                ax.plot(c["level"], c["rejection_rate"], color=colors[k],
                        lw=1.4 if is_null else 2.2,
                        ls="--" if is_null else "-", zorder=3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.2)
            if ri == 0:
                ax.set_title(f"d = {d}")
            if ci == 0:
                lab = r"$\hat{\sigma}$-test" if param == "sigma" else r"$\hat{\alpha}$-test"
                ax.set_ylabel(f"{lab}\nrejection rate (power)")
            if ri == len(PARAMS) - 1:
                ax.set_xlabel("significance level")

    # per-row legends (effect sizes differ between sigma and alpha)
    for ri, param in enumerate(PARAMS):
        psub = sub[sub["param"] == param]
        keys = sorted(psub[series_key].unique())
        colors = {k: cb[i % len(cb)] for i, k in enumerate(keys)}
        sym = r"\sigma" if param == "sigma" else r"\alpha"
        if sweep == "effect":
            handles = [Line2D([], [], color=colors[k],
                              lw=1.4 if abs(k) < 1e-9 else 2.2,
                              ls="--" if abs(k) < 1e-9 else "-",
                              label=(f"$|\\Delta {sym}|={k:g}$ (null)" if abs(k) < 1e-9
                                     else f"$|\\Delta {sym}|={k:g}$"))
                       for k in keys]
        else:
            handles = [Line2D([], [], color=colors[k], lw=2.2, label=f"$n={int(k)}$")
                       for k in keys]
        axes[ri][-1].legend(handles=handles, fontsize=8, loc="lower right",
                            title=None)

    fixed_txt = ""
    if sweep == "sample":
        fixed_txt = (f" — $|\\Delta\\sigma|={abs(SIGMA2_SAMPLE-SIGMA1):g}$, "
                     f"$|\\Delta\\alpha|={abs(ALPHA2_SAMPLE-ALPHA1):g}$")
    title = ("TLG ROC: effect size" if sweep == "effect"
             else "TLG ROC: sample size")
    fig.suptitle(f"{title} ({METHOD} test; null = chance diagonal){fixed_txt}",
                 y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
