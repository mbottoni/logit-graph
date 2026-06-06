#!/usr/bin/env python3
"""Validate the single-graph dyadic-robust Wald test by simulation (ROC curves).

The real-data ANOVAs (Twitch, connectomes) compare sigma across groups with one
graph per group, using the dyadic-cluster-robust SE + Wald test. This script
validates *that same test* under the same one-graph-per-group design, replacing
the older ROC machinery that (a) validated a different test (ANOVA over n_reps
replicate graphs, impossible on one real graph) and (b) faked SE proportional to
1/n via a fractional pair-subsample knob (ROCSweepConfig.subsample_pairs), which
ignores dyadic dependence and overstates power growth in n.

It mirrors the original ROC experiment (notebooks/anova/12-0-anova-roc.ipynb and
experiments.presets.ROCSweepConfig) in design and graph sizes:

  * ROC curve = rejection rate vs p-value threshold (the empirical p-value CDF).
    Scenario A (different sigma, H1) bows toward the top-left; Scenario B (equal
    sigma, H0) tracks the diagonal y=x (calibration). Better discrimination =>
    higher AUC.
  * sample-size sweep: sigma1=-1.0 vs sigma2_fixed=-1.5 over n in
    {10,100,500,1000,2000} (d=1 capped, Gibbs cost).
  * effect-size sweep: sigma1=-1.0 vs sigma2 in {-1.0,-1.5,-2.0,-2.5} at
    n_effect=500.

Per replicate we generate two *independent* graphs, fit offset-logit sigma-hat +
dyadic-robust SE on each (logit_graph.robust_se), and form the two-sided Wald
z = (s1 - s2)/sqrt(SE1^2 + SE2^2). No subsample knob anywhere.

Graph generation reuses experiments.sweeps.simulate_graph (d=0 direct ER at
p=expit(sigma); d=1 Layer-2 Gibbs at beta=1, adaptive_stopping disabled).
Reproducible: fixed seed (LG_AVR_SEED), BLAS threads pinned to 1. Writes only
under runs/anova_validation_robust/.

Env knobs (all optional):
  LG_AVR_SEED (12345)     LG_AVR_QUICK (0 -> full; 1 -> d=0, small n, few reps)
  LG_AVR_ALPHA (0.05)     LG_AVR_DS ("0,1")
  LG_AVR_N_EXP (200)      d=0 experiments per scenario point
  LG_AVR_D1_N_EXP (80)    d=1 experiments per scenario point (Gibbs is slower)
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
from sklearn.metrics import roc_auc_score

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

# Sweep design -- mirrors experiments.presets.ROCSweepConfig (original ROC fig).
SIGMA1 = -1.0
EFFECT_SIGMAS = [-1.0, -1.5, -2.0, -2.5]   # sigma2 (first = null -> H0 diagonal)
N_EFFECT = 500                             # graph size for the effect-size sweep
SIGMA2_FIXED = -1.5                        # fixed effect for the sample-size sweep
N_VALUES = ([10, 100] if QUICK
            else [10, 100, 500, 1000, 2000])  # sample-size sweep grid
THRESHOLDS = np.linspace(0.0, 1.0, 101)


def _n_exp(d):
    return D1_N_EXP if d >= 1 else N_EXP


def _gen(n, d, sigma, seed):
    if d == 0:
        return simulate_graph(n, 0, sigma=sigma, n_iter=0, seed=seed)
    n_iter = max(D1_BURN_MIN, D1_BURN_PER_N * n)
    return simulate_graph(n, d, sigma=sigma, beta=1.0, n_iter=n_iter,
                          feature_mode=FEATURE_MODE, seed=seed,
                          adaptive_stopping=False)


def _wald_p(adj1, adj2, d):
    s1, se1, _ = fit_sigma_with_robust_se(adj1, d, feature_mode=FEATURE_MODE)
    s2, se2, _ = fit_sigma_with_robust_se(adj2, d, feature_mode=FEATURE_MODE)
    denom = math.sqrt(se1 ** 2 + se2 ** 2)
    if not (denom > 0):
        return float("nan")
    return 2.0 * norm.sf(abs((s1 - s2) / denom))


def _collect_pvalues(n, d, sigma1, sigma2, n_exp, seed0):
    """Two-sided robust-Wald p over n_exp replicates of two independent graphs."""
    ps = []
    for r in range(n_exp):
        a1 = _gen(n, d, sigma1, seed0 + 2 * r)
        a2 = _gen(n, d, sigma2, seed0 + 2 * r + 1)
        p = _wald_p(a1, a2, d)
        if not math.isnan(p):
            ps.append(p)
    return np.asarray(ps)


def _roc_curve(pvals):
    """Rejection rate vs p-value threshold (empirical CDF of p), original style."""
    return np.array([float(np.mean(pvals <= t)) for t in THRESHOLDS])


def _auc(h0_p, h1_p):
    """AUC of the H0-vs-H1 classifier scored by 1-p (higher = more likely H1)."""
    if len(h0_p) == 0 or len(h1_p) == 0:
        return float("nan")
    y = np.r_[np.zeros(len(h0_p)), np.ones(len(h1_p))]
    s = np.r_[1.0 - h0_p, 1.0 - h1_p]
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = _here / "runs" / "anova_validation_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"robust-Wald validation (ROC)  seed={SEED}  quick={QUICK}  alpha={ALPHA}  "
          f"d={DS}  n_exp(d0)={N_EXP}  n_exp(d1)={D1_N_EXP}  d1_max_n={D1_MAX_N}")
    print(f"  sample sweep: sigma1={SIGMA1} vs sigma2={SIGMA2_FIXED} over n={N_VALUES}")
    print(f"  effect sweep: sigma1={SIGMA1} vs sigma2={EFFECT_SIGMAS} at n={N_EFFECT}")

    # H0 p-values per (d, n): reused for calibration, sample-AUC, and (at n=500)
    # the effect-sweep null.
    h0_by_dn = {}
    sample_rows, sample_curves = [], {}
    for d in DS:
        for k, n in enumerate(N_VALUES):
            if d >= 1 and n > D1_MAX_N:
                print(f"  [Sample] d={d} n={n}: SKIPPED (n>d1_max_n={D1_MAX_N})")
                continue
            t0 = time.perf_counter()
            h0 = _collect_pvalues(n, d, SIGMA1, SIGMA1, _n_exp(d),
                                  SEED + 1000 * (d + 1) + 50 * k)
            h1 = _collect_pvalues(n, d, SIGMA1, SIGMA2_FIXED, _n_exp(d),
                                  SEED + 7000 * (d + 1) + 50 * k)
            h0_by_dn[(d, n)] = h0
            sample_curves[(d, n)] = (_roc_curve(h0), _roc_curve(h1))
            typeI = float(np.mean(h0 < ALPHA))
            power = float(np.mean(h1 < ALPHA))
            auc = _auc(h0, h1)
            sample_rows.append(dict(d=d, n=n, sigma1=SIGMA1, sigma2=SIGMA2_FIXED,
                                    effect=abs(SIGMA2_FIXED - SIGMA1), typeI=typeI,
                                    power=power, auc=auc, n_exp=len(h1)))
            print(f"  [Sample] d={d} n={n}: typeI={typeI:.3f} power={power:.3f} "
                  f"AUC={auc:.3f}  [{time.perf_counter()-t0:.0f}s]")

    effect_rows, effect_curves = [], {}
    for d in DS:
        n = N_EFFECT if not (d >= 1 and N_EFFECT > D1_MAX_N) else D1_MAX_N
        h0 = h0_by_dn.get((d, n))
        if h0 is None:
            h0 = _collect_pvalues(n, d, SIGMA1, SIGMA1, _n_exp(d),
                                  SEED + 1000 * (d + 1) + 9999)
            h0_by_dn[(d, n)] = h0
        for k, s2 in enumerate(EFFECT_SIGMAS):
            t0 = time.perf_counter()
            if s2 == SIGMA1:
                h1 = h0
            else:
                h1 = _collect_pvalues(n, d, SIGMA1, s2, _n_exp(d),
                                      SEED + 2000 * (d + 1) + 50 * k)
            effect_curves[(d, s2)] = _roc_curve(h1)
            power = float(np.mean(h1 < ALPHA))
            auc = _auc(h0, h1)
            effect_rows.append(dict(d=d, n=n, sigma1=SIGMA1, sigma2=s2,
                                    effect=abs(s2 - SIGMA1), power=power,
                                    auc=auc, n_exp=len(h1)))
            print(f"  [Effect] d={d} n={n} sigma2={s2:+.1f} (effect={abs(s2-SIGMA1):.1f}): "
                  f"power={power:.3f} AUC={auc:.3f}  [{time.perf_counter()-t0:.0f}s]")

    sample_df = pd.DataFrame(sample_rows)
    effect_df = pd.DataFrame(effect_rows)
    sample_df.to_csv(out_dir / "roc_sample_robust.csv", index=False)
    effect_df.to_csv(out_dir / "roc_effect_robust.csv", index=False)
    sample_df[["d", "n", "typeI", "n_exp"]].to_csv(out_dir / "typeI.csv", index=False)

    # --- Plots: rejection rate vs p-value threshold (original ROC style) -----
    _plot_threshold_roc(
        out_dir / "roc_sample_robust.png",
        f"Robust-Wald ROC vs graph size (effect {abs(SIGMA2_FIXED-SIGMA1):.1f})",
        {d: [(f"n={n}", sample_curves[(d, n)][1])
             for n in N_VALUES if (d, n) in sample_curves] for d in DS},
        {d: [(f"H0 n={n}", sample_curves[(d, n)][0])
             for n in N_VALUES if (d, n) in sample_curves] for d in DS},
    )
    _plot_threshold_roc(
        out_dir / "roc_effect_robust.png",
        f"Robust-Wald ROC vs effect size (n={N_EFFECT})",
        {d: [(f"eff={abs(s2-SIGMA1):.1f}", effect_curves[(d, s2)])
             for s2 in EFFECT_SIGMAS if s2 != SIGMA1 and (d, s2) in effect_curves]
            for d in DS},
        {d: [("H0 (eff=0)", effect_curves[(d, SIGMA1)])]
            for d in DS if (d, SIGMA1) in effect_curves},
    )

    (out_dir / "results.json").write_text(json.dumps({
        "alpha": ALPHA, "seed": SEED, "ds": DS,
        "sample": sample_rows, "effect": effect_rows,
    }, indent=2, default=float))

    print("\n" + "=" * 70)
    print("AUC (1=perfect discrimination, 0.5=chance) and Type-I (~alpha):")
    for r in sample_rows:
        print(f"  [sample] d={r['d']} n={r['n']:4d}: AUC={r['auc']:.3f}  "
              f"typeI={r['typeI']:.3f}  power={r['power']:.3f}")
    for r in effect_rows:
        print(f"  [effect] d={r['d']} n={r['n']} effect={r['effect']:.1f}: "
              f"AUC={r['auc']:.3f}  power={r['power']:.3f}")
    print(f"\nWrote {out_dir}/ (roc_sample_robust.{{png,csv}}, "
          f"roc_effect_robust.{{png,csv}}, typeI.csv, results.json)")


def _plot_threshold_roc(path, suptitle, h1_curves_by_d, h0_curves_by_d):
    ds = sorted(h1_curves_by_d.keys())
    fig, axes = plt.subplots(1, len(ds), figsize=(5.2 * len(ds), 4.2), squeeze=False)
    for ax, d in zip(axes[0], ds):
        for label, curve in h1_curves_by_d[d]:
            ax.plot(THRESHOLDS, curve, marker="", lw=1.8, label=label)
        for label, curve in h0_curves_by_d.get(d, []):
            ax.plot(THRESHOLDS, curve, color="0.6", lw=0.9, ls=":")
        ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls="--", label="y=x (H0 ideal)")
        ax.axvline(ALPHA, color="r", lw=0.7, ls=":")
        ax.set_xlabel("p-value threshold")
        ax.set_ylabel("rejection rate")
        ax.set_title(f"d={d}")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
