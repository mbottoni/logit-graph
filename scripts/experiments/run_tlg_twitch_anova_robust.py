#!/usr/bin/env python3
"""Twitch ANOVA for the Temporal Logit-Graph (TLG) on BOTH parameters (sigma, alpha) with a 2x2
dyadic-cluster-robust SE: fit logit P[edge] = sigma + alpha*D per region, then compare each parameter
across the six Twitch communities (omnibus Cochran-Q + pairwise Wald; joint 2-df Wald on the pair)."""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import norm, chi2
from statsmodels.stats.multitest import multipletests

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
for p in (_src, _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")

from logit_graph.lg_features import build_pair_dataset  # noqa: E402
from logit_graph.temporal import fit_growth_params  # noqa: E402


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


QUICK = os.environ.get("LG_TTA_QUICK", "0") == "1"
REGIONS = (["PTBR", "RU"] if QUICK
           else os.environ.get("LG_TTA_REGIONS", "DE,ENGB,ES,FR,PTBR,RU").split(","))
D = _int("LG_TTA_D", 1)             # degree-feature depth (the TLG uses d=1)
DEGREE_MODE = "bounded"            # D_ij = log(1+S_i) + log(1+S_j)
SEED = _int("LG_TTA_SEED", 12345)
DISPLAY = {"DE": "DE", "ENGB": "EN", "ES": "ES", "FR": "FR", "PTBR": "PT", "RU": "RU"}
PARAMS = ("sigma", "alpha")
PLABEL = {"sigma": r"$\hat{\sigma}$ (intercept)", "alpha": r"$\hat{\alpha}$ (degree slope)"}


def _load_region(path):
    """Undirected, self-loop-free largest connected component, relabeled 0..n-1."""
    G = nx.read_edgelist(path, comments="#", nodetype=int)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


# ---------------------------------------------------------------------------
# (sigma, alpha) fit + 2-parameter dyadic-cluster-robust SE on the full graph
# ---------------------------------------------------------------------------

def _fit_tlg(adj, d):
    """Cross-sectional logistic MLE of logit P = sigma + alpha*D over upper-triangle dyads."""
    Draw, labels = build_pair_dataset(adj, d=d, mode=DEGREE_MODE, layer2=True)
    Dvec = np.asarray(Draw, dtype=np.float64)
    fit = fit_growth_params(Dvec, labels)
    return fit["sigma"], fit["alpha"], Dvec, np.asarray(labels, dtype=np.float64)


def _dyadic_robust_se2(n, sigma, alpha, Dvec, labels):
    """2x2 sandwich Var = A^{-1} B A^{-1} for (sigma, alpha) with dyadic (shared-node) clustering,
    X = [1, D]. Bread A = X' diag(p(1-p)) X (logistic Fisher info); meat B = sum_m T_m T_m' -
    sum_i s_i s_i', s_i = (y_i - p_i)[1, D_i], T_m = sum of s_i over dyads incident to node m."""
    p = expit(sigma + alpha * Dvec)
    w = p * (1.0 - p)
    r = labels - p
    A = np.array([[float(w.sum()),          float((w * Dvec).sum())],
                  [float((w * Dvec).sum()), float((w * Dvec * Dvec).sum())]])
    s0 = r                      # d/d sigma
    s1 = r * Dvec               # d/d alpha
    T0 = np.zeros(n); T1 = np.zeros(n)
    start = 0
    for i in range(n - 1):
        cnt = n - 1 - i
        s0_row = s0[start:start + cnt]; s1_row = s1[start:start + cnt]
        T0[i] += float(s0_row.sum()); T0[i + 1:] += s0_row
        T1[i] += float(s1_row.sum()); T1[i + 1:] += s1_row
        start += cnt
    B = np.array([[float((T0 * T0).sum() - (s0 * s0).sum()),
                   float((T0 * T1).sum() - (s0 * s1).sum())],
                  [float((T0 * T1).sum() - (s0 * s1).sum()),
                   float((T1 * T1).sum() - (s1 * s1).sum())]])
    Ainv = np.linalg.inv(A)
    V_robust = Ainv @ B @ Ainv
    V_naive = Ainv
    se = dict(sigma=math.sqrt(max(V_robust[0, 0], 0.0)),
              alpha=math.sqrt(max(V_robust[1, 1], 0.0)))
    se_naive = dict(sigma=math.sqrt(max(V_naive[0, 0], 0.0)),
                    alpha=math.sqrt(max(V_naive[1, 1], 0.0)))
    return se, se_naive, V_robust


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _omnibus_wald(vals, ses):
    """Cochran's Q equality test: H0 all values equal. ~ chi2(k-1)."""
    w = 1.0 / np.asarray(ses) ** 2
    x = np.asarray(vals)
    x_bar = float(np.sum(w * x) / np.sum(w))
    Q = float(np.sum(w * (x - x_bar) ** 2))
    dof = len(vals) - 1
    return Q, dof, float(chi2.sf(Q, dof))


def _pairwise_wald(regions, vals, ses):
    rows = []
    for a, b in combinations(range(len(regions)), 2):
        z = (vals[a] - vals[b]) / math.sqrt(ses[a] ** 2 + ses[b] ** 2)
        rows.append({"region_i": regions[a], "region_j": regions[b],
                     "z": z, "p_raw": 2.0 * norm.sf(abs(z))})
    df = pd.DataFrame(rows)
    df["p_bonf"] = multipletests(df["p_raw"], method="bonferroni")[1]
    df["p_fdr"] = multipletests(df["p_raw"], method="fdr_bh")[1]
    return df


def _pairwise_joint(regions, sig, alp, covs):
    """Joint 2-df Wald on the (sigma, alpha) vector: W = d' (V_a+V_b)^{-1} d ~ chi2(2)."""
    rows = []
    for a, b in combinations(range(len(regions)), 2):
        d = np.array([sig[a] - sig[b], alp[a] - alp[b]])
        Vab = covs[a] + covs[b]
        W = float(d @ np.linalg.solve(Vab, d))
        rows.append({"region_i": regions[a], "region_j": regions[b],
                     "W": W, "p_raw": float(chi2.sf(W, 2))})
    df = pd.DataFrame(rows)
    df["p_bonf"] = multipletests(df["p_raw"], method="bonferroni")[1]
    return df


# ---------------------------------------------------------------------------
# Validation: 2-param robust SE vs Monte-Carlo sampling SD
# ---------------------------------------------------------------------------

def _validate():
    """Canonical dyadic-cluster-robust validation: generate y_ij ~ Bernoulli(expit(sigma + alpha*x_ij
    + u_i + u_j)) on a fixed covariate x_ij=z_i+z_j with OMITTED node random effects u, fit without u
    (misspecified), and compare robust/naive SE to the Monte-Carlo SD (expect robust ~= MC SD > naive)."""
    print("=== 2-param dyadic-robust SE validation vs Monte-Carlo (node random effects) ===")
    rng = np.random.default_rng(SEED)
    n, sigma, alpha, tau, M = 400, -2.0, 0.6, 0.4, 200
    rows, cols = np.triu_indices(n, 1)
    z = rng.normal(size=n)                     # fixed node covariate -> identified alpha
    x = (z[rows] + z[cols]).astype(np.float64)

    def sample(seed):
        r = np.random.default_rng(seed)
        u = r.normal(scale=tau, size=n)        # omitted node effects -> dyadic dependence
        p = expit(sigma + alpha * x + u[rows] + u[cols])
        return (r.random(len(p)) < p).astype(np.float64)

    # robust/naive SE averaged over a few graphs (single-graph robust SE is itself noisy)
    se_r = {"sigma": [], "alpha": []}; se_nv = {"sigma": [], "alpha": []}
    for g in range(8):
        y = sample(SEED + g)
        f = fit_growth_params(x, y)
        se, se_n, _ = _dyadic_robust_se2(n, f["sigma"], f["alpha"], x, y)
        for par in PARAMS:
            se_r[par].append(se[par]); se_nv[par].append(se_n[par])
    mc_s, mc_a = [], []
    for r in range(M):
        f = fit_growth_params(x, sample(SEED + 1000 + r))
        mc_s.append(f["sigma"]); mc_a.append(f["alpha"])
    mc = {"sigma": np.std(mc_s, ddof=1), "alpha": np.std(mc_a, ddof=1)}
    print(f"  n={n}, sigma={sigma}, alpha={alpha}, node-RE sd={tau}  "
          f"(robust SE averaged over 8 graphs):")
    for par in PARAMS:
        print(f"    {par}: SE_robust={np.mean(se_r[par]):.4f}  SE_naive={np.mean(se_nv[par]):.4f}"
              f"  MC_SD={mc[par]:.4f}  (robust>=naive, robust~=MC expected)")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _heatmap(ax, regions, pdf, title):
    k = len(regions)
    P = np.ones((k, k)); idx = {r: i for i, r in enumerate(regions)}
    for _, r in pdf.iterrows():
        i, j = idx[r["region_i"]], idx[r["region_j"]]
        P[i, j] = P[j, i] = r["p_bonf"]
    Mlog = -np.log10(np.clip(P, 1e-300, 1.0)); np.fill_diagonal(Mlog, np.nan)
    codes = [DISPLAY.get(r, r) for r in regions]
    im = ax.imshow(Mlog, cmap="viridis", vmin=0, vmax=min(80, np.nanmax(Mlog)))
    ax.set_xticks(range(k)); ax.set_xticklabels(codes)
    ax.set_yticks(range(k)); ax.set_yticklabels(codes)
    ax.set_title(title)
    return im


def _plot(df, pdf, regions, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ri, par in enumerate(PARAMS):
        order = np.argsort(df[f"{par}_hat"].to_numpy())
        val = df[f"{par}_hat"].to_numpy()[order]
        se_r = df[f"se_{par}_robust"].to_numpy()[order]
        se_n = df[f"se_{par}_naive"].to_numpy()[order]
        disp = [DISPLAY.get(regions[i], regions[i]) for i in order]
        y = np.arange(len(order))
        ax0 = axes[ri][0]
        ax0.errorbar(val, y, xerr=se_r, fmt="o", color="#1f77b4", ecolor="#1f77b4",
                     elinewidth=1.6, capsize=4, label="robust SE", zorder=2)
        ax0.errorbar(val, y, xerr=se_n, fmt="none", ecolor="#d62728", elinewidth=2.4,
                     capsize=0, label="naive SE", zorder=3)
        ax0.set_yticks(y); ax0.set_yticklabels(disp)
        ax0.set_xlabel(PLABEL[par]); ax0.set_title(f"Per-region {par} (robust vs naive SE)")
        ax0.grid(axis="x", ls=":", alpha=0.4); ax0.legend(fontsize=8, loc="lower right")
        im = _heatmap(axes[ri][1], regions, pdf[par],
                      rf"{par}: pairwise $-\log_{{10}}$(Bonferroni $p$)")
        cbar = fig.colorbar(im, ax=axes[ri][1], fraction=0.046, pad=0.04)
        cbar.set_label(r"$-\log_{10} p_{\mathrm{bonf}}$ ($>1.3\Rightarrow$ sig.@0.05)")
    fig.suptitle("Twitch TLG ANOVA: sigma and alpha across communities "
                 "(dyadic-cluster-robust SE)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if os.environ.get("LG_TTA_VALIDATE", "0") == "1":
        _validate(); return

    data_dir = _repo_root / "data" / "twitch" / "graphs_processed"
    out_dir = _here / "runs" / "twitch_tlg_anova_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"twitch TLG dyadic-robust ANOVA (sigma, alpha)  seed={SEED}  quick={QUICK}  "
          f"regions={REGIONS}  d={D}")

    rows, covs = [], {}
    for region in REGIONS:
        path = data_dir / f"{region}_graph.edges"
        if not path.exists():
            print(f"  {region}: file not found — skipping"); continue
        t0 = time.perf_counter()
        G = _load_region(path); n, m = G.number_of_nodes(), G.number_of_edges()
        adj = nx.to_numpy_array(G)
        sigma, alpha, Dv, lab = _fit_tlg(adj, D)
        se, se_n, V = _dyadic_robust_se2(n, sigma, alpha, Dv, lab)
        covs[region] = V
        rows.append(dict(region=region, display=DISPLAY.get(region, region), n=n, edges=m,
                         sigma_hat=sigma, alpha_hat=alpha,
                         se_sigma_robust=se["sigma"], se_sigma_naive=se_n["sigma"],
                         se_alpha_robust=se["alpha"], se_alpha_naive=se_n["alpha"]))
        print(f"  {DISPLAY.get(region, region):3s}  n={n:5d}  E={m:7d}  "
              f"sigma={sigma:+.4f}(SE {se['sigma']:.4f})  alpha={alpha:+.4f}(SE {se['alpha']:.4f})  "
              f"[{time.perf_counter()-t0:.0f}s]")

    if len(rows) < 2:
        print("\nNeed >=2 regions."); return
    df = pd.DataFrame(rows)
    regions = list(df["region"])

    omni, pdf = {}, {}
    for par in PARAMS:
        vals = df[f"{par}_hat"].to_numpy()
        ses = df[f"se_{par}_robust"].to_numpy()
        omni[par] = _omnibus_wald(vals, ses)
        pdf[par] = _pairwise_wald(regions, vals, ses)
    pjoint = _pairwise_joint(regions, df["sigma_hat"].to_numpy(),
                             df["alpha_hat"].to_numpy(), [covs[r] for r in regions])

    df.to_csv(out_dir / "summary.csv", index=False)
    for par in PARAMS:
        d2 = pdf[par].copy()
        d2["region_i"] = [DISPLAY.get(r, r) for r in d2["region_i"]]
        d2["region_j"] = [DISPLAY.get(r, r) for r in d2["region_j"]]
        d2.to_csv(out_dir / f"pairwise_{par}.csv", index=False)
    pjoint.to_csv(out_dir / "pairwise_joint.csv", index=False)
    _plot(df, pdf, regions, out_dir / "twitch_tlg_anova_robust.png")
    (out_dir / "results.json").write_text(json.dumps({
        "omnibus": {par: {"Q": omni[par][0], "dof": omni[par][1], "p": omni[par][2]}
                    for par in PARAMS},
        "regions": df.to_dict(orient="records"),
        "sig_bonferroni": {par: int((pdf[par]["p_bonf"] < 0.05).sum()) for par in PARAMS},
        "sig_bonferroni_joint": int((pjoint["p_bonf"] < 0.05).sum()),
        "n_pairs": int(len(pjoint)),
    }, indent=2, default=float))

    print("\n" + "=" * 70)
    for par in PARAMS:
        Q, dof, p = omni[par]
        nb = int((pdf[par]["p_bonf"] < 0.05).sum())
        print(f"[{par}] omnibus Q={Q:.1f} dof={dof} p={p:.2e} | "
              f"Bonferroni-significant pairs: {nb}/{len(pdf[par])}")
    nj = int((pjoint["p_bonf"] < 0.05).sum())
    print(f"[joint (sigma,alpha) 2-df] Bonferroni-significant pairs: {nj}/{len(pjoint)}")
    print("\nNOTE: SEs are dyadic-cluster-robust (real sampling interpretation, not "
          "dial-able by subsample size).")
    print(f"Wrote {out_dir}/")


if __name__ == "__main__":
    main()
