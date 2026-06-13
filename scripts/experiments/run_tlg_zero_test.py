#!/usr/bin/env python3
"""Single-graph significance test for the Temporal Logit-Graph (TLG): is a parameter zero?

The companion experiments (run_tlg_twitch_anova_robust.py / run_tlg_connectomes_anova_robust.py)
answer a BETWEEN-graph question: do k networks share the same sigma / alpha (H0: theta_1=...=theta_k)?
That needs many graphs. This script answers the complementary SINGLE-graph question raised in the
paper (Section 3.5 review note):

    Given ONE observed network, are its edges actually shaped by the model's effects?
    H0: theta = 0   vs   H1: theta != 0   for theta in {sigma, alpha}.

The substantively interesting one is **alpha = 0**: "do this network's edges depend on node degree?"
Reject => the bounded degree feature D carries real signal; fail to reject => the graph is consistent
with degree-independent (intercept-only) edge formation. We also report **sigma = 0** (baseline
log-odds = 0, i.e. baseline density 1/2).

For each network we fit logit P[edge_ij] = sigma + alpha*D_ij (cross-sectional logistic MLE on the
upper-triangle dyads, D = bounded degree feature at depth d), form a 2-parameter dyadic-cluster-robust
SE, and report the per-parameter Wald test

    z_theta = theta_hat / SE_robust(theta_hat),   p = 2 * Phi(-|z_theta|),

Bonferroni-corrected across the networks in each dataset. The dyadic-robust SE matters: under dyadic
(shared-node) dependence the naive logistic SE is anticonservative, so an alpha=0 test on the naive SE
would over-reject. The fit and the 2x2 sandwich SE are reused verbatim from
run_tlg_twitch_anova_robust.py (dataset-agnostic).

Run with LG_ZT_VALIDATE=1 for a Monte-Carlo calibration/power study. NOTE on the estimator: the
cross-sectional fit above is biased for alpha (the equilibrium degree-endogeneity / row-sum problem
documented in FINDINGS -- on a static snapshot D_ij for an existing edge is computed on a graph that
already contains that edge), so it cannot be validated by simulating from a known alpha. The
simulation therefore uses the *consistent* temporal-snapshot MLE (logit_graph.temporal): we grow a
TLG at a known alpha (storing snapshots), build the at-risk design from G(t-1)->G(t), and fit by
ordinary logistic regression -- here formations are conditionally independent given the predetermined
snapshot, so the model is the exact MLE and the ordinary logistic SE is valid (no dyadic clustering
needed). We report the rejection rate of the alpha=0 Wald test vs the true alpha (Type I error at
alpha=0; power for alpha>0) plus a ROC curve (null vs the largest alpha). This validates the *test*
(calibration + power) under the generative model; the real-data section applies the analogous test to
the single observed snapshot with the cross-sectional fit, matching the paper's ANOVA experiments.

Output under runs/tlg_zero_test/ (gitignored):
  twitch_zero_test.csv  connectomes_zero_test.csv   per-network theta_hat, robust SE, z, p, p_bonf
  tlg_zero_test_real.png      forest plots (theta_hat +/- 1.96 SE, 0-line) per dataset
  tlg_zero_test_validation.png   rejection-rate vs alpha + ROC   (LG_ZT_VALIDATE=1)
  results.json

Env: LG_ZT_DATASETS (twitch,connectomes), LG_ZT_D (1), LG_ZT_NMIN (10), LG_ZT_QUICK (0),
LG_ZT_VALIDATE (0); sim knobs LG_ZT_SIM_N (50), LG_ZT_SIM_STEPS (2), LG_ZT_SIM_REPS (200),
LG_ZT_SIM_SIGMA (-2.0), LG_ZT_SIM_ALPHAS (0,0.02,0.05,0.1,0.2), LG_ZT_SEED (12345).

  make tlg-zero-test            real-data tests on twitch + connectomes
  make tlg-zero-test-quick      smoke
  make tlg-zero-test-validate   Monte-Carlo calibration / power / ROC
"""
from __future__ import annotations

import glob
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
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
for p in (_repo_root / "src", _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")

# Reuse the dataset-agnostic TLG fit + 2-param dyadic-robust SE + the twitch loader.
import run_tlg_twitch_anova_robust as TW  # noqa: E402
from logit_graph.temporal import (  # noqa: E402
    grow_graph, growth_design_from_snapshots, fit_growth_params,
)

PARAMS = TW.PARAMS                          # ("sigma", "alpha")
PLABEL = {"sigma": r"$\hat{\sigma}$ (intercept)", "alpha": r"$\hat{\alpha}$ (degree slope)"}


def _int(env, d):
    v = os.environ.get(env); return int(v) if v else d


def _float(env, d):
    v = os.environ.get(env); return float(v) if v else d


def _floats(env, d):
    v = os.environ.get(env)
    return [float(x) for x in v.split(",")] if v else d


QUICK = os.environ.get("LG_ZT_QUICK", "0") == "1"
DATASETS = os.environ.get("LG_ZT_DATASETS", "twitch,connectomes").split(",")
D = _int("LG_ZT_D", 1)
NMIN = _int("LG_ZT_NMIN", 10)
SEED = _int("LG_ZT_SEED", 12345)

# simulation (validation) knobs. Small graphs / few steps keep the pooled snapshot design
# small enough that the power curve is informative (otherwise power saturates at 1).
SIM_N = _int("LG_ZT_SIM_N", 50)
SIM_STEPS = _int("LG_ZT_SIM_STEPS", 2)
SIM_REPS = (40 if QUICK else _int("LG_ZT_SIM_REPS", 200))
SIM_SIGMA = _float("LG_ZT_SIM_SIGMA", -2.0)
SIM_ALPHAS = _floats("LG_ZT_SIM_ALPHAS", [0.0, 0.02, 0.05, 0.1, 0.2])
JOBS = _int("LG_ZT_JOBS", max(1, (os.cpu_count() or 4) - 2))

OUT = _here / "runs" / "tlg_zero_test"


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Per-graph zero test
# ---------------------------------------------------------------------------

def _zero_test(adj, d):
    """Fit (sigma, alpha) cross-sectionally, return the per-parameter Wald z, p vs H0: theta=0.

    Returns dict with theta_hat, se_robust, se_naive, z, p_raw for each parameter, plus the
    2x2 robust covariance. None if the fit is degenerate (quasi-perfect separation)."""
    n = adj.shape[0]
    sigma, alpha, Dv, lab = TW._fit_tlg(adj, d)
    se, se_n, V = TW._dyadic_robust_se2(n, sigma, alpha, Dv, lab)
    vals = {"sigma": sigma, "alpha": alpha}
    if (not np.all(np.isfinite([sigma, alpha, se["sigma"], se["alpha"]]))
            or se["sigma"] < 1e-4 or se["alpha"] < 1e-4 or abs(sigma) > 200):
        return None
    out = {}
    for par in PARAMS:
        z = vals[par] / se[par]
        out[par] = dict(hat=vals[par], se_robust=se[par], se_naive=se_n[par],
                        z=z, p_raw=2.0 * norm.sf(abs(z)))
    out["_V"] = V
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _twitch_networks():
    data_dir = _repo_root / "data" / "twitch" / "graphs_processed"
    regions = (["PTBR", "RU"] if QUICK
               else os.environ.get("LG_ZT_REGIONS", "DE,ENGB,ES,FR,PTBR,RU").split(","))
    for region in regions:
        path = data_dir / f"{region}_graph.edges"
        if not path.exists():
            log(f"  {region}: file not found — skipping"); continue
        G = TW._load_region(path)
        yield TW.DISPLAY.get(region, region), G


def _connectome_networks():
    files = sorted(glob.glob(str(_repo_root / "data" / "connectomes" / "*.graphml")))
    if QUICK:
        files = files[:4]
    for f in files:
        stem = Path(f).stem
        try:
            G = nx.Graph(nx.read_graphml(f))
            G.remove_edges_from(nx.selfloop_edges(G))
            cc = max(nx.connected_components(G), key=len)
            G = nx.convert_node_labels_to_integers(G.subgraph(cc).copy())
        except Exception as ex:
            log(f"  {stem}: load failed ({ex}) — skipping"); continue
        yield _short(stem), G


def _short(stem):
    s = stem.replace("_neural", "").replace(".synaptic", "").replace("_brain", ".br") \
            .replace(".cortex", ".ctx").replace("interareal.cortical.network", "iacn") \
            .replace("norvegicus", "norv").replace("herm_pharynx", "herm")
    return (s[:16]) if len(s) > 16 else s


NETWORKS = {"twitch": _twitch_networks, "connectomes": _connectome_networks}


# ---------------------------------------------------------------------------
# Real-data run
# ---------------------------------------------------------------------------

def run_dataset(name):
    log(f"\n=== {name}: single-graph zero test (H0: theta=0)  d={D} ===")
    rows = []
    for label, G in NETWORKS[name]():
        n, m = G.number_of_nodes(), G.number_of_edges()
        if n < NMIN:
            log(f"  {label}: n={n} < {NMIN} — skipping"); continue
        t0 = time.perf_counter()
        adj = nx.to_numpy_array(G, weight=None)
        res = _zero_test(adj, D)
        if res is None:
            log(f"  {label:16s}  n={n:5d}  E={m:7d}  DEGENERATE fit — excluded"); continue
        row = dict(network=label, n=n, edges=m)
        for par in PARAMS:
            r = res[par]
            row.update({f"{par}_hat": r["hat"], f"se_{par}_robust": r["se_robust"],
                        f"se_{par}_naive": r["se_naive"], f"z_{par}": r["z"],
                        f"p_{par}_raw": r["p_raw"]})
        rows.append(row)
        log(f"  {label:16s}  n={n:5d}  E={m:7d}  "
            f"alpha={res['alpha']['hat']:+.3f} (SE {res['alpha']['se_robust']:.3f}, "
            f"z={res['alpha']['z']:+.1f}, p={res['alpha']['p_raw']:.1e})  "
            f"[{time.perf_counter()-t0:.0f}s]")

    if not rows:
        log(f"  {name}: no usable networks."); return None
    df = pd.DataFrame(rows)
    # Bonferroni across the networks in this dataset, per parameter.
    for par in PARAMS:
        df[f"p_{par}_bonf"] = multipletests(df[f"p_{par}_raw"], method="bonferroni")[1]
    return df


# ---------------------------------------------------------------------------
# Simulation: Type I error + power + ROC for the alpha=0 test
# ---------------------------------------------------------------------------

def _sim_one(job):
    """Grow a TLG at (sigma, alpha), fit the CONSISTENT temporal-snapshot MLE, and return the
    alpha=0 Wald p-value. Formations are conditionally independent given the predetermined
    snapshot, so the ordinary logistic SE is valid (no dyadic clustering needed)."""
    alpha, seed = job
    res = grow_graph(SIM_N, d=D, sigma=SIM_SIGMA, alpha=alpha, n_steps=SIM_STEPS,
                     seed=seed, store_snapshots=True)
    X, lab = growth_design_from_snapshots(res.snapshots, d=D)
    f = fit_growth_params(X, lab)
    z = f["alpha"] / f["se_alpha"]
    return np.nan if not np.isfinite(z) else 2.0 * norm.sf(abs(z))


def run_validation():
    log(f"\n=== Monte-Carlo calibration / power for the alpha=0 test "
        f"(consistent temporal-snapshot MLE) ===")
    log(f"  n={SIM_N} steps={SIM_STEPS} reps={SIM_REPS} sigma={SIM_SIGMA} "
        f"alphas={SIM_ALPHAS} jobs={JOBS}")
    from concurrent.futures import ProcessPoolExecutor
    pvals = {}
    for a in SIM_ALPHAS:
        jobs = [(a, SEED + 9173 * i + int(round(1000 * a))) for i in range(SIM_REPS)]
        if JOBS <= 1:
            ps = [_sim_one(j) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=JOBS) as pool:
                ps = list(pool.map(_sim_one, jobs, chunksize=1))
        ps = np.array([p for p in ps if np.isfinite(p)])
        pvals[a] = ps
        rej05 = float(np.mean(ps < 0.05))
        tag = "Type I error" if a == 0.0 else "power"
        log(f"  alpha={a:.2f}: reject@0.05 = {rej05:.3f}  ({tag}; {len(ps)} usable reps)")
    return pvals


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _forest(ax, df, par, title):
    order = np.argsort(df[f"{par}_hat"].to_numpy())
    val = df[f"{par}_hat"].to_numpy()[order]
    se = df[f"se_{par}_robust"].to_numpy()[order]
    p_bonf = df[f"p_{par}_bonf"].to_numpy()[order]
    labels = [df["network"].to_numpy()[i] for i in order]
    y = np.arange(len(order))
    sig = p_bonf < 0.05
    colors = np.where(sig, "#c1121f", "#6c757d")
    ax.axvline(0.0, color="black", lw=1.4, ls="-", zorder=1)
    for yi, v, s, c in zip(y, val, se, colors):
        ax.errorbar(v, yi, xerr=1.96 * s, fmt="o", color=c, ecolor=c,
                    elinewidth=1.6, capsize=3, ms=5, zorder=2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(PLABEL[par])
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.4)
    n_sig = int(sig.sum())
    ax.text(0.98, 0.02, f"reject $H_0$: {n_sig}/{len(order)}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.8))


def plot_real(dfs, out_path):
    names = list(dfs.keys())
    fig, axes = plt.subplots(len(names), 2, squeeze=False,
                             figsize=(11, max(3.0, 0.32 * sum(len(d) for d in dfs.values())) + 1))
    for ri, name in enumerate(names):
        df = dfs[name]
        _forest(axes[ri][0], df, "sigma", f"{name}: $H_0:\\sigma=0$")
        _forest(axes[ri][1], df, "alpha",
                f"{name}: $H_0:\\alpha=0$ (edges depend on degree?)")
    fig.suptitle("TLG single-graph significance test (Wald, dyadic-cluster-robust SE; "
                 "1.96·SE bars, Bonferroni-significant in red)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150); plt.close(fig)
    log(f"Saved {out_path}")


def plot_validation(pvals, out_path):
    alphas = sorted(pvals.keys())
    rej = [float(np.mean(pvals[a] < 0.05)) for a in alphas]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(alphas, rej, "o-", color="#0072B2", lw=2, ms=7)
    ax[0].axhline(0.05, color="#c1121f", ls="--", lw=1.3, label="nominal 0.05")
    ax[0].set_xlabel(r"true $\alpha$ (0 = null)")
    ax[0].set_ylabel(r"reject $H_0:\alpha=0$ at level 0.05")
    ax[0].set_title("Calibration (at $\\alpha=0$) and power (at $\\alpha>0$)")
    ax[0].set_ylim(-0.02, 1.02); ax[0].grid(alpha=.3); ax[0].legend()

    # ROC: null (alpha=0) vs each alternative, sweeping the significance level.
    null = pvals[0.0] if 0.0 in pvals else pvals[alphas[0]]
    alts = [a for a in alphas if a != 0.0]
    levels = np.linspace(0, 1, 201)
    fpr = np.array([float(np.mean(null < t)) for t in levels])
    aucs = {}
    cols = ["#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    for i, a in enumerate(alts):
        tpr = np.array([float(np.mean(pvals[a] < t)) for t in levels])
        auc = float(np.trapezoid(tpr, fpr)); aucs[a] = auc
        ax[1].plot(fpr, tpr, "-", color=cols[i % len(cols)], lw=2,
                   label=f"$\\alpha={a:g}$ (AUC={auc:.3f})")
    ax[1].plot([0, 1], [0, 1], ":", color="#888", lw=1.2, label="chance")
    ax[1].set_xlabel("false positive rate (reject under $\\alpha=0$)")
    ax[1].set_ylabel("true positive rate (reject under $\\alpha>0$)")
    ax[1].set_title("ROC of the $\\alpha=0$ test (null vs each alternative)")
    ax[1].grid(alpha=.3); ax[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("TLG $\\alpha=0$ test: Monte-Carlo calibration, power, and ROC", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150); plt.close(fig)
    log(f"Saved {out_path}")
    return rej, aucs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}

    if os.environ.get("LG_ZT_VALIDATE", "0") == "1":
        pvals = run_validation()
        rej, aucs = plot_validation(pvals, OUT / "tlg_zero_test_validation.png")
        summary["validation"] = {
            "alphas": SIM_ALPHAS, "reject_at_0.05": rej,
            "roc_auc": {str(a): v for a, v in aucs.items()},
            "type_I_error": rej[SIM_ALPHAS.index(0.0)] if 0.0 in SIM_ALPHAS else None,
            "n": SIM_N, "steps": SIM_STEPS, "reps": SIM_REPS, "sigma": SIM_SIGMA,
        }
        (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=float))
        log(f"\nWrote {OUT}/")
        return

    dfs = {}
    for name in DATASETS:
        if name not in NETWORKS:
            log(f"unknown dataset '{name}' — skipping"); continue
        df = run_dataset(name)
        if df is None:
            continue
        dfs[name] = df
        df.to_csv(OUT / f"{name}_zero_test.csv", index=False)
        summary[name] = {
            "n_networks": int(len(df)),
            "sigma_reject_bonf": int((df["p_sigma_bonf"] < 0.05).sum()),
            "alpha_reject_bonf": int((df["p_alpha_bonf"] < 0.05).sum()),
            "networks": df.to_dict(orient="records"),
        }

    if not dfs:
        log("No datasets produced usable fits."); return
    plot_real(dfs, OUT / "tlg_zero_test_real.png")
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, default=float))

    log("\n" + "=" * 70)
    for name, df in dfs.items():
        na = int((df["p_alpha_bonf"] < 0.05).sum())
        ns = int((df["p_sigma_bonf"] < 0.05).sum())
        log(f"[{name}] degree dependence (reject alpha=0, Bonferroni): {na}/{len(df)} | "
            f"reject sigma=0: {ns}/{len(df)}")
    log("\nNOTE: SEs are dyadic-cluster-robust; alpha=0 asks 'do this graph's edges depend on degree?'")
    log(f"Wrote {OUT}/")


if __name__ == "__main__":
    main()
