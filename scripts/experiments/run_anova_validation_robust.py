#!/usr/bin/env python3
"""Validate the single-graph dyadic-robust Wald test by simulation (ROC curves).

The real-data ANOVAs (Twitch, connectomes) compare sigma across groups with one
graph per group, using the dyadic-cluster-robust SE + Wald test. This script
validates *that same test* under the same one-graph-per-group design, replacing
the older ROC machinery that (a) validated a different test (ANOVA over n_reps
replicate graphs, impossible on one real graph) and (b) faked SE proportional to
1/n via a fractional pair-subsample knob (ROCSweepConfig.subsample_pairs).

Why a *standardized* effect. A Wald/z-test's discrimination is governed entirely
by the non-centrality delta = (sigma1 - sigma2) / sqrt(SE1^2 + SE2^2): the ROC
AUC is exactly Phi(delta / sqrt(2)). With honest SEs (SE ~ 1/sqrt(n)), a fixed
*raw* effect like 0.5 gives a huge delta at n>=100, so the ROC saturates (AUC=1)
-- which is correct but uninformative. We therefore sweep the standardized effect
delta directly: at each (n, d) we estimate the typical SE from a pilot and set
the raw gap sigma1 - sigma2 = delta * sqrt(2) * SE so the realized non-centrality
is ~delta. This traces the test's intrinsic ROC (no saturation, no subsample
knob), keeps the original graph sizes, and lets us check the empirical AUC
against the theoretical Phi(delta/sqrt(2)).

Two figures:
  * roc_effect_robust  -- ROC (TPR vs FPR) for delta in {0.5,1,1.5,2,3} at
    n_effect=500. Curves span diagonal -> top-left; empirical AUC ~= theory.
  * roc_sample_robust  -- vary n at ONE fixed raw effect (sigma gap), pinned so
    the mid-grid n lands at DELTA_REF. Larger n -> larger non-centrality ->
    better AUC, so the ROC curves separate (discrimination grows with graph
    size) without saturating -- the honest version of the old Fig 4.

Per replicate we generate two *independent* graphs (experiments.sweeps.simulate_graph:
d=0 direct ER at p=expit(sigma); d=1 Layer-2 Gibbs at beta=1, adaptive_stopping
disabled), fit offset-logit sigma-hat + dyadic-robust SE on each
(logit_graph.robust_se), and form z = (s1-s2)/sqrt(SE1^2+SE2^2). Reproducible:
fixed seed (LG_AVR_SEED), BLAS threads pinned to 1. Writes only under
runs/anova_validation_robust/.

Env knobs (all optional):
  LG_AVR_SEED (12345)     LG_AVR_QUICK (0 -> full; 1 -> d=0, small n, few reps)
  LG_AVR_ALPHA (0.05)     LG_AVR_DS ("0,1")
  LG_AVR_N_EXP (200)      d=0 experiments per scenario point
  LG_AVR_D1_N_EXP (80)    d=1 experiments per scenario point
  LG_AVR_D1_MAX_N (500)   skip d=1 points above this n (logged)

  make anova-validation-robust         full run
  make anova-validation-robust-quick   smoke (d=0, small n, few reps)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
for p in (_src, _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")

from logit_graph.robust_se import fit_sigma_with_robust_se  # noqa: E402
from logit_graph.experiments.sweeps import simulate_graph  # noqa: E402


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


def _float(env, default):
    raw = os.environ.get(env)
    return float(raw) if raw is not None else default


QUICK = os.environ.get("LG_AVR_QUICK", "0") == "1"
SEED = _int("LG_AVR_SEED", 12345)
ALPHA = _float("LG_AVR_ALPHA", 0.05)
DS = [0] if QUICK else [int(x) for x in os.environ.get("LG_AVR_DS", "0,1").split(",")]
N_EXP = _int("LG_AVR_N_EXP", 30 if QUICK else 200)
D1_N_EXP = _int("LG_AVR_D1_N_EXP", 20 if QUICK else 80)
D1_MAX_N = _int("LG_AVR_D1_MAX_N", 500)
FEATURE_MODE = "incremental"

# d=1 Gibbs burn-in per independent sample.
D1_BURN_MIN = _int("LG_AVR_D1_BURN_MIN", 8000)
D1_BURN_PER_N = _int("LG_AVR_D1_BURN_PER_N", 40)

# Sweep design (standardized effect -- see module docstring).
SIGMA1 = -1.0
DELTAS = [0.5, 1.0] if QUICK else [0.5, 1.0, 1.5, 2.0, 3.0]  # standardized effects
N_EFFECT = 500                                # graph size for the effect ROC
# Sample ROC: fix ONE raw effect (sigma gap) and vary n -> discrimination
# improves with graph size (the honest version of the old Fig 4). The raw gap is
# pinned so the mid-grid n lands at standardized DELTA_REF; smaller/larger n then
# spread below/above it, giving visibly separated (non-overlapping) ROC curves.
DELTA_REF = 1.5
N_SAMPLE = [50, 100] if QUICK else [50, 100, 200, 350, 500]
PILOT_K = _int("LG_AVR_PILOT_K", 6 if QUICK else 16)  # graphs to estimate typical SE


def _n_exp(d):
    return D1_N_EXP if d >= 1 else N_EXP


def _pilot_k(d):
    return max(4, PILOT_K // 2) if d >= 1 else PILOT_K


def _gen(n, d, sigma, seed):
    if d == 0:
        return simulate_graph(n, 0, sigma=sigma, n_iter=0, seed=seed)
    n_iter = max(D1_BURN_MIN, D1_BURN_PER_N * n)
    return simulate_graph(n, d, sigma=sigma, beta=1.0, n_iter=n_iter,
                          feature_mode=FEATURE_MODE, seed=seed,
                          adaptive_stopping=False)


def _fit(adj, d):
    return fit_sigma_with_robust_se(adj, d, feature_mode=FEATURE_MODE)


def _wald_p(adj1, adj2, d):
    s1, se1, _ = _fit(adj1, d)
    s2, se2, _ = _fit(adj2, d)
    denom = math.sqrt(se1 ** 2 + se2 ** 2)
    if not (denom > 0):
        return float("nan")
    return 2.0 * norm.sf(abs((s1 - s2) / denom))


def _typical_se(n, d, sigma, k, seed0):
    """Median dyadic-robust SE of sigma_hat over k independent graphs at sigma."""
    ses = []
    for r in range(k):
        _, se, _ = _fit(_gen(n, d, sigma, seed0 + r), d)
        if np.isfinite(se) and se > 0:
            ses.append(se)
    return float(np.median(ses)) if ses else float("nan")


def _collect_pvalues(n, d, sigma1, sigma2, n_exp, seed0):
    ps = []
    for r in range(n_exp):
        a1 = _gen(n, d, sigma1, seed0 + 2 * r)
        a2 = _gen(n, d, sigma2, seed0 + 2 * r + 1)
        p = _wald_p(a1, a2, d)
        if not math.isnan(p):
            ps.append(p)
    return np.asarray(ps)


def _roc(h0_p, h1_p):
    """Return (fpr, tpr, auc) of the H0-vs-H1 classifier scored by 1-p."""
    if len(h0_p) == 0 or len(h1_p) == 0:
        return np.array([0, 1]), np.array([0, 1]), float("nan")
    y = np.r_[np.zeros(len(h0_p)), np.ones(len(h1_p))]
    s = np.r_[1.0 - h0_p, 1.0 - h1_p]
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, float(roc_auc_score(y, s))


def _theory_auc(delta):
    """ROC AUC of the TWO-sided Wald test (score = |z|) at non-centrality delta.

    Under H0 z~N(0,1), under H1 z~N(delta,1); the two-sided p ranks by |z|, so
    AUC = P(|z_H1| > |z_H0|) = E_U[P(V>U)] with U=|N(0,1)|, V=|N(delta,1)|.
    (The one-sided value Phi(delta/sqrt(2)) would overstate it.)
    """
    u = np.linspace(0.0, 12.0, 6001)
    f_u = 2.0 * norm.pdf(u)                       # half-normal density of |z_H0|
    p_v_gt_u = 1.0 - norm.cdf(u - delta) + norm.cdf(-u - delta)  # P(|z_H1| > u)
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(p_v_gt_u * f_u, u))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = _here / "runs" / "anova_validation_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"robust-Wald validation (standardized-effect ROC)  seed={SEED}  "
          f"quick={QUICK}  alpha={ALPHA}  d={DS}  n_exp(d0)={N_EXP}  n_exp(d1)={D1_N_EXP}")
    print(f"  effect ROC: delta={DELTAS} at n={N_EFFECT}")
    print(f"  sample ROC: fixed raw effect (delta_ref={DELTA_REF}) over n={N_SAMPLE} "
          f"(d1 cap {D1_MAX_N})")

    # --- Effect ROC: sweep standardized delta at n_effect --------------------
    effect_rows, effect_curves = [], {}
    for d in DS:
        n = N_EFFECT if not (d >= 1 and N_EFFECT > D1_MAX_N) else D1_MAX_N
        se_typ = _typical_se(n, d, SIGMA1, _pilot_k(d), SEED + 100 * (d + 1))
        h0 = _collect_pvalues(n, d, SIGMA1, SIGMA1, _n_exp(d), SEED + 1000 * (d + 1))
        typeI = float(np.mean(h0 < ALPHA))
        print(f"  [Effect] d={d} n={n}  SE_typ={se_typ:.4f}  typeI={typeI:.3f}")
        for k, delta in enumerate(DELTAS):
            t0 = time.perf_counter()
            dsig = delta * math.sqrt(2.0) * se_typ
            sigma2 = SIGMA1 - dsig
            h1 = _collect_pvalues(n, d, SIGMA1, sigma2, _n_exp(d),
                                  SEED + 2000 * (d + 1) + 50 * k)
            fpr, tpr, auc = _roc(h0, h1)
            effect_curves[(d, delta)] = (fpr, tpr)
            effect_rows.append(dict(d=d, n=n, delta=delta, raw_dsigma=dsig,
                                    sigma2=sigma2, se_typ=se_typ,
                                    power=float(np.mean(h1 < ALPHA)), auc_emp=auc,
                                    auc_theory=_theory_auc(delta), typeI=typeI,
                                    n_exp=len(h1)))
            print(f"    delta={delta:.1f} (raw dsigma={dsig:.4f}): "
                  f"AUC_emp={auc:.3f} AUC_theory={_theory_auc(delta):.3f} "
                  f"power={float(np.mean(h1 < ALPHA)):.3f}  [{time.perf_counter()-t0:.0f}s]")

    # --- Sample ROC: ONE fixed raw effect, vary n (discrimination grows with n) -
    sample_rows, sample_curves, sample_dsig = [], {}, {}
    for d in DS:
        grid = [n for n in N_SAMPLE if not (d >= 1 and n > D1_MAX_N)]
        n_ref = grid[len(grid) // 2]
        se_ref = _typical_se(n_ref, d, SIGMA1, _pilot_k(d), SEED + 600 * (d + 1))
        raw_ds = DELTA_REF * math.sqrt(2.0) * se_ref   # pinned across all n
        sigma2 = SIGMA1 - raw_ds
        sample_dsig[d] = raw_ds
        print(f"  [Sample] d={d}  raw dsigma={raw_ds:.4f} "
              f"(pinned at n_ref={n_ref}, delta_ref={DELTA_REF})  sigma2={sigma2:.4f}")
        for k, n in enumerate(grid):
            t0 = time.perf_counter()
            se_n = _typical_se(n, d, SIGMA1, _pilot_k(d), SEED + 650 * (d + 1) + 7 * k)
            delta_n = raw_ds / (math.sqrt(2.0) * se_n) if se_n > 0 else float("nan")
            h0 = _collect_pvalues(n, d, SIGMA1, SIGMA1, _n_exp(d),
                                  SEED + 4000 * (d + 1) + 50 * k)
            h1 = _collect_pvalues(n, d, SIGMA1, sigma2, _n_exp(d),
                                  SEED + 5000 * (d + 1) + 50 * k)
            fpr, tpr, auc = _roc(h0, h1)
            sample_curves[(d, n)] = (fpr, tpr)
            sample_rows.append(dict(d=d, n=n, raw_dsigma=raw_ds, sigma2=sigma2,
                                    se_typ=se_n, delta=delta_n,
                                    typeI=float(np.mean(h0 < ALPHA)),
                                    power=float(np.mean(h1 < ALPHA)), auc_emp=auc,
                                    auc_theory=_theory_auc(delta_n), n_exp=len(h1)))
            print(f"    n={n:4d}: delta~={delta_n:.2f}  AUC_emp={auc:.3f} "
                  f"(theory {_theory_auc(delta_n):.3f})  "
                  f"typeI={float(np.mean(h0 < ALPHA)):.3f}  [{time.perf_counter()-t0:.0f}s]")

    effect_df = pd.DataFrame(effect_rows)
    sample_df = pd.DataFrame(sample_rows)
    effect_df.to_csv(out_dir / "roc_effect_robust.csv", index=False)
    sample_df.to_csv(out_dir / "roc_sample_robust.csv", index=False)
    if not sample_df.empty:
        sample_df[["d", "n", "typeI", "n_exp"]].to_csv(out_dir / "typeI.csv", index=False)

    _plot_roc(
        out_dir / "roc_effect_robust.png",
        rf"Robust-Wald ROC vs standardized effect $\delta=\Delta\sigma/SE$ (n={N_EFFECT})",
        {d: [(rf"$\delta$={delta:.1f} (AUC {_field(effect_rows, d, 'delta', delta, 'auc_emp'):.2f})",
              *effect_curves[(d, delta)])
             for delta in DELTAS if (d, delta) in effect_curves] for d in DS},
    )
    _plot_roc(
        out_dir / "roc_sample_robust.png",
        r"Robust-Wald ROC vs graph size at a fixed raw effect "
        r"(discrimination grows with $n$)",
        {d: [(f"n={n} ($\\delta${chr(0x2248)}{_field(sample_rows, d, 'n', n, 'delta'):.1f}, "
              f"AUC {_field(sample_rows, d, 'n', n, 'auc_emp'):.2f})", *sample_curves[(d, n)])
             for n in N_SAMPLE if (d, n) in sample_curves] for d in DS},
        title_suffix={d: rf"$\Delta\sigma$={sample_dsig[d]:.3f}" for d in DS},
    )

    (out_dir / "results.json").write_text(json.dumps({
        "alpha": ALPHA, "seed": SEED, "ds": DS, "sigma1": SIGMA1,
        "effect": effect_rows, "sample": sample_rows,
    }, indent=2, default=float))

    print("\n" + "=" * 70)
    print("Empirical vs theoretical AUC = Phi(delta/sqrt(2)):")
    for r in effect_rows:
        print(f"  [effect] d={r['d']} n={r['n']} delta={r['delta']:.1f}: "
              f"AUC_emp={r['auc_emp']:.3f}  theory={r['auc_theory']:.3f}  "
              f"(raw dsigma={r['raw_dsigma']:.4f})")
    print("Sample ROC (fixed raw dsigma per d; discrimination grows with n):")
    for r in sample_rows:
        print(f"  [sample] d={r['d']} n={r['n']:4d} (delta~={r['delta']:.2f}): "
              f"AUC_emp={r['auc_emp']:.3f}  theory={r['auc_theory']:.3f}  "
              f"(raw dsigma={r['raw_dsigma']:.4f}, typeI={r['typeI']:.3f})")
    print(f"\nWrote {out_dir}/ (roc_effect_robust.{{png,csv}}, "
          f"roc_sample_robust.{{png,csv}}, typeI.csv, results.json)")


def _field(rows, d, key, val, field):
    for r in rows:
        if r["d"] == d and r[key] == val:
            return r[field]
    return float("nan")


def _plot_roc(path, suptitle, curves_by_d, title_suffix=None):
    ds = sorted(curves_by_d.keys())
    fig, axes = plt.subplots(1, len(ds), figsize=(5.2 * len(ds), 4.6), squeeze=False)
    for ax, d in zip(axes[0], ds):
        for label, fpr, tpr in curves_by_d[d]:
            ax.plot(fpr, tpr, lw=1.8, label=label)
        ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--", label="chance")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        suffix = f"  ({title_suffix[d]})" if title_suffix and d in title_suffix else ""
        ax.set_title(f"d={d}{suffix}")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
