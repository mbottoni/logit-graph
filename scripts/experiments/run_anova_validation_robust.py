#!/usr/bin/env python3
"""Validate the single-graph dyadic-robust Wald test by simulation.

The real-data ANOVAs (Twitch, connectomes) compare sigma across groups with one
graph per group, using the dyadic-cluster-robust SE + Wald test. This script
validates *that same test* under the same one-graph-per-group design, replacing
the older ROC machinery that (a) validated a different test (ANOVA over n_reps
replicate graphs, which cannot be run on a single real graph) and (b) faked
SE proportional to 1/n via a fractional pair-subsample knob (which ignores
dyadic dependence and overstates power growth in n).

Here, for each Monte-Carlo replicate we generate two *independent* graphs, fit
offset-logit sigma-hat + dyadic-robust SE on each (logit_graph.robust_se), and
form the two-sided Wald z = (s1 - s2) / sqrt(SE1^2 + SE2^2):

  * Type-I calibration: both graphs at the SAME sigma -> empirical reject rate
    should be ~ alpha (tests whether the robust SE is calibrated on finite graphs).
  * Power: graphs at DIFFERENT sigma -> reject rate = power. Swept over effect
    size (sigma2 - sigma1, n fixed) and sample size (n, effect fixed), per
    d in {0,1}. With the honest robust SE, power grows with n and effect size
    WITHOUT any subsample knob.

Graph generation reuses experiments.sweeps.simulate_graph (d=0 direct ER at
p=expit(sigma); d=1 Layer-2 Gibbs at beta=1, adaptive_stopping disabled).
Reproducible: fixed seed (LG_AVR_SEED), BLAS threads pinned to 1. Writes only
under runs/anova_validation_robust/.

Env knobs (all optional):
  LG_AVR_SEED (12345)     LG_AVR_QUICK (0 -> full; 1 -> d=0 only, few reps)
  LG_AVR_ALPHA (0.05)     LG_AVR_DS ("0,1")
  LG_AVR_N_EXP (300)      d=0 Monte-Carlo replicates per point
  LG_AVR_D1_N_EXP (80)    d=1 replicates per point (Gibbs is slower)
  LG_AVR_D1_MAX_N (500)   skip d=1 sample-sweep points above this n (logged)

  make anova-validation-robust         full run
  make anova-validation-robust-quick   smoke (d=0, few reps)
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
N_EXP = _int("LG_AVR_N_EXP", 40 if QUICK else 300)
D1_N_EXP = _int("LG_AVR_D1_N_EXP", 20 if QUICK else 80)
D1_MAX_N = _int("LG_AVR_D1_MAX_N", 500)
FEATURE_MODE = "incremental"

# d=1 Gibbs burn-in per independent sample.
D1_BURN_MIN = _int("LG_AVR_D1_BURN_MIN", 8000)
D1_BURN_PER_N = _int("LG_AVR_D1_BURN_PER_N", 40)

# Sweep design (matches the real-data application: one graph per group).
# With the honest robust SE (~1/sqrt(n)), power transitions sharply, so the
# sweeps are centered in the informative regime: a small fixed n for the
# effect sweep, and a small fixed effect over a fine n-grid for the sample
# sweep (a sharp rise IS the honest finding -- the old fractional-subsample
# knob faked a gradual SE~1/n curve).
SIGMA_BASE = -1.0                       # reference sigma_1
TYPEI_NS = [50, 200, 1000]              # check calibration across n (asymptotic Wald)
EFFECT_SIGMAS = [-1.0, -1.25, -1.5, -1.75, -2.0]  # sigma_2 (first = null -> Type-I)
EFFECT_N = 20                           # fixed (small) n for the effect-size sweep
SAMPLE_NS = [10, 25, 50, 100, 250, 500, 1000, 2000]  # fine n-grid at fixed effect
SAMPLE_SIGMA2 = -1.5                    # fixed sigma_2 (effect 0.5) for the n-sweep


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


def _reject_rate(n, d, sigma1, sigma2, n_exp, seed0):
    """Fraction of MC replicates with two-sided Wald p < ALPHA (two indep. graphs)."""
    rej, valid = 0, 0
    for r in range(n_exp):
        a1 = _gen(n, d, sigma1, seed0 + 2 * r)
        a2 = _gen(n, d, sigma2, seed0 + 2 * r + 1)
        p = _wald_p(a1, a2, d)
        if math.isnan(p):
            continue
        valid += 1
        rej += int(p < ALPHA)
    return (rej / valid if valid else float("nan")), valid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = _here / "runs" / "anova_validation_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"robust-Wald validation  seed={SEED}  quick={QUICK}  alpha={ALPHA}  "
          f"d={DS}  n_exp(d0)={N_EXP}  n_exp(d1)={D1_N_EXP}  d1_max_n={D1_MAX_N}")

    # --- Type-I calibration: both graphs at the same sigma -------------------
    typeI_rows = []
    for d in DS:
        for k, n in enumerate(TYPEI_NS):
            if d >= 1 and n > D1_MAX_N:
                print(f"  [Type-I] d={d} n={n}: SKIPPED (n > d1_max_n={D1_MAX_N})")
                continue
            t0 = time.perf_counter()
            rate, valid = _reject_rate(n, d, SIGMA_BASE, SIGMA_BASE, _n_exp(d),
                                       SEED + 1000 * (d + 1) + 50 * k)
            typeI_rows.append(dict(d=d, n=n, sigma=SIGMA_BASE, alpha=ALPHA,
                                   reject_rate=rate, n_exp=valid))
            print(f"  [Type-I] d={d} n={n} sigma={SIGMA_BASE}: reject={rate:.3f} "
                  f"(target alpha={ALPHA}, n_exp={valid})  [{time.perf_counter()-t0:.0f}s]")

    # --- Power vs effect size (n fixed) -------------------------------------
    effect_rows = []
    for d in DS:
        n = EFFECT_N if d == 0 else min(EFFECT_N, D1_MAX_N)
        for k, s2 in enumerate(EFFECT_SIGMAS):
            t0 = time.perf_counter()
            rate, valid = _reject_rate(n, d, SIGMA_BASE, s2, _n_exp(d),
                                       SEED + 2000 * (d + 1) + 50 * k)
            effect_rows.append(dict(d=d, n=n, sigma1=SIGMA_BASE, sigma2=s2,
                                    effect=abs(s2 - SIGMA_BASE), power=rate, n_exp=valid))
            print(f"  [Effect] d={d} n={n} sigma2={s2:+.1f} (effect={abs(s2-SIGMA_BASE):.1f}): "
                  f"power={rate:.3f}  [{time.perf_counter()-t0:.0f}s]")

    # --- Power vs sample size (effect fixed) --------------------------------
    sample_rows = []
    for d in DS:
        for k, n in enumerate(SAMPLE_NS):
            if d >= 1 and n > D1_MAX_N:
                print(f"  [Sample] d={d} n={n}: SKIPPED (n > d1_max_n={D1_MAX_N}, "
                      f"Gibbs too slow)")
                continue
            t0 = time.perf_counter()
            rate, valid = _reject_rate(n, d, SIGMA_BASE, SAMPLE_SIGMA2, _n_exp(d),
                                       SEED + 3000 * (d + 1) + 50 * k)
            sample_rows.append(dict(d=d, n=n, sigma1=SIGMA_BASE, sigma2=SAMPLE_SIGMA2,
                                    effect=abs(SAMPLE_SIGMA2 - SIGMA_BASE),
                                    power=rate, n_exp=valid))
            print(f"  [Sample] d={d} n={n} effect={abs(SAMPLE_SIGMA2-SIGMA_BASE):.1f}: "
                  f"power={rate:.3f}  [{time.perf_counter()-t0:.0f}s]")

    typeI_df = pd.DataFrame(typeI_rows)
    effect_df = pd.DataFrame(effect_rows)
    sample_df = pd.DataFrame(sample_rows)
    typeI_df.to_csv(out_dir / "typeI.csv", index=False)
    effect_df.to_csv(out_dir / "roc_effect_robust.csv", index=False)
    sample_df.to_csv(out_dir / "roc_sample_robust.csv", index=False)

    # --- Plots --------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    for d in DS:
        sub = effect_df[effect_df["d"] == d].sort_values("effect")
        plt.plot(sub["effect"], sub["power"], marker="o", label=f"d={d}")
    plt.axhline(ALPHA, color="gray", ls="--", lw=1, label=f"alpha={ALPHA}")
    plt.xlabel(r"effect size $|\sigma_2-\sigma_1|$")
    plt.ylabel("power (reject rate)")
    plt.title(f"Robust-Wald power vs effect size (n={EFFECT_N})")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_effect_robust.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    for d in DS:
        sub = sample_df[sample_df["d"] == d].sort_values("n")
        if not sub.empty:
            plt.plot(sub["n"], sub["power"], marker="o", label=f"d={d}")
    plt.xlabel("n (nodes per graph)")
    plt.ylabel("power (reject rate)")
    plt.title(f"Robust-Wald power vs n (effect={abs(SAMPLE_SIGMA2-SIGMA_BASE):.1f})")
    plt.xscale("log")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_sample_robust.png", dpi=150)
    plt.close()

    (out_dir / "results.json").write_text(json.dumps({
        "alpha": ALPHA, "seed": SEED, "ds": DS,
        "typeI": typeI_rows, "effect": effect_rows, "sample": sample_rows,
    }, indent=2, default=float))

    print("\n" + "=" * 70)
    print("Type-I (should be ~ alpha):")
    for r in typeI_rows:
        flag = "" if abs(r["reject_rate"] - ALPHA) < 3 * math.sqrt(
            ALPHA * (1 - ALPHA) / max(r["n_exp"], 1)) else "  <-- off"
        print(f"  d={r['d']} n={r['n']}: reject={r['reject_rate']:.3f}{flag}")
    print(f"\nWrote {out_dir}/ (typeI.csv, roc_effect_robust.{{png,csv}}, "
          f"roc_sample_robust.{{png,csv}}, results.json)")


if __name__ == "__main__":
    main()
