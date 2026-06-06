#!/usr/bin/env python3
"""Twitch ANOVA on sigma-hat with a dyadic-cluster-robust SE (no pseudo-replication)."""
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
from logit_graph.offset_logit import fit_offset_logit_fast  # noqa: E402
from logit_graph.graph import GraphModel  # noqa: E402


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


QUICK = os.environ.get("LG_TWA_QUICK", "0") == "1"
REGIONS = (["PTBR", "RU"] if QUICK
           else os.environ.get("LG_TWA_REGIONS", "DE,ENGB,ES,FR,PTBR,RU").split(","))
# d candidates default to {0,1}: the paper's observed Twitch optima are only ever
# 0 or 1, and the d>=2 offset feature is O(n^3) at full n (per-pair multi-hop BFS
# over ~n^2/2 pairs), infeasible on DE (n~9500). Raise LG_TWA_D_MAX to search wider.
D_MAX = _int("LG_TWA_D_MAX", 1)
FEATURE_MODE = "incremental"
SEED = _int("LG_TWA_SEED", 12345)
DISPLAY = {"DE": "DE", "ENGB": "EN", "ES": "ES", "FR": "FR", "PTBR": "PT", "RU": "RU"}


def _load_region(path):
    """Undirected, self-loop-free largest connected component, relabeled 0..n-1."""
    G = nx.read_edgelist(path, comments="#", nodetype=int)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


# ---------------------------------------------------------------------------
# sigma fit + dyadic-cluster-robust SE on the full graph
# ---------------------------------------------------------------------------

def _fit_sigma(adj, d):
    """offset-logit MLE sigma_hat at radius d on all upper-triangle pairs."""
    offsets, labels = build_pair_dataset(adj, d=d, mode=FEATURE_MODE, layer2=True)
    sigma, ll = fit_offset_logit_fast(offsets, labels)
    aic = -2.0 * ll + 2.0
    return sigma, ll, aic, offsets, labels


def _select_d(adj, d_candidates):
    """AIC d-selection; returns (d_hat, sigma_hat, offsets, labels, aic_by_d)."""
    best = None
    labels = None
    aic_by_d = {}
    for d in d_candidates:
        sigma, ll, aic, offsets, lab = _fit_sigma(adj, d)
        aic_by_d[d] = aic
        if labels is None:
            labels = lab
        if best is None or aic < best[0]:
            best = (aic, d, sigma, offsets)
    return best[1], best[2], best[3], labels, aic_by_d


def _dyadic_robust_se(n, sigma_hat, offsets, labels):
    """Sandwich SE for sigma_hat with dyadic (shared-node) clustering.

    Bread A = sum p(1-p); meat B = sum_m T_m^2 - sum s^2 with T_m the sum of
    score residuals s=y-p over dyads incident to node m (row-slicing keeps it
    O(n^2) time / O(n) extra memory). Var = B / A^2.
    """
    p = expit(sigma_hat + offsets)
    s = labels.astype(np.float64) - p
    A = float(np.sum(p * (1.0 - p)))
    sum_s2 = float(np.sum(s * s))

    T = np.zeros(n, dtype=np.float64)
    start = 0
    for i in range(n - 1):
        cnt = n - 1 - i
        s_row = s[start:start + cnt]          # pairs (i, i+1..n-1), row-major
        T[i] += float(s_row.sum())
        T[i + 1:] += s_row
        start += cnt

    B = float(np.sum(T * T) - sum_s2)
    se_robust = math.sqrt(max(B, 0.0)) / A if A > 0 else float("nan")
    se_naive = 1.0 / math.sqrt(A) if A > 0 else float("nan")
    return se_robust, se_naive


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _omnibus_wald(sigmas, ses):
    """Cochran's Q / fixed-effect equality test: H0 all sigma equal. ~ chi2(k-1)."""
    w = 1.0 / np.asarray(ses) ** 2
    s = np.asarray(sigmas)
    s_bar = float(np.sum(w * s) / np.sum(w))
    Q = float(np.sum(w * (s - s_bar) ** 2))
    dof = len(sigmas) - 1
    return Q, dof, float(chi2.sf(Q, dof))


def _pairwise_wald(regions, sigmas, ses):
    rows = []
    for a, b in combinations(range(len(regions)), 2):
        z = (sigmas[a] - sigmas[b]) / math.sqrt(ses[a] ** 2 + ses[b] ** 2)
        p = 2.0 * norm.sf(abs(z))
        rows.append({"region_i": regions[a], "region_j": regions[b],
                     "z": z, "p_raw": p})
    df = pd.DataFrame(rows)
    df["p_bonf"] = multipletests(df["p_raw"], method="bonferroni")[1]
    df["p_fdr"] = multipletests(df["p_raw"], method="fdr_bh")[1]
    return df


# ---------------------------------------------------------------------------
# Validation: dyadic-robust SE vs Monte-Carlo sampling SD
# ---------------------------------------------------------------------------

def _validate():
    print("=== SE validation vs Monte-Carlo ground truth ===")
    rng = np.random.default_rng(SEED)

    # (a) d=0 / ER: dyads independent -> robust ~ naive ~ MC SD.
    n, p_true, M = 300, 0.02, 400
    g0 = nx.to_numpy_array(nx.erdos_renyi_graph(n, p_true, seed=SEED))
    sig0, _, _, off0, lab0 = _fit_sigma(g0, 0)
    se_r, se_n = _dyadic_robust_se(n, sig0, off0, lab0)
    mc = []
    for r in range(M):
        gr = nx.to_numpy_array(nx.erdos_renyi_graph(n, p_true, seed=int(rng.integers(1 << 30))))
        s, _, _, _, _ = _fit_sigma(gr, 0)
        mc.append(s)
    print(f"  d=0 ER (n={n}, p={p_true}): SE_robust={se_r:.4f}  SE_naive={se_n:.4f}  "
          f"MC_SD={np.std(mc, ddof=1):.4f}  (robust~=naive~=MC expected)")

    # (b) d=1 / LG: shared-degree offsets -> dependence -> robust > naive ~ MC SD.
    n, sigma, M = 200, -2.0, 120
    def lg_sample(seed):
        gm = GraphModel(n=n, d=1, sigma=sigma, er_p=float(np.clip(expit(sigma), 0.02, 0.5)),
                        layer2=True, feature_mode=FEATURE_MODE, seed=seed)
        for _ in range(max(6000, 30 * n)):
            gm.add_remove_edge()
        return gm.graph
    g1 = lg_sample(SEED)
    sig1, _, _, off1, lab1 = _fit_sigma(g1, 1)
    se_r1, se_n1 = _dyadic_robust_se(n, sig1, off1, lab1)
    mc1 = []
    for r in range(M):
        gr = lg_sample(SEED + 1 + r)
        s, _, _, _, _ = _fit_sigma(gr, 1)
        mc1.append(s)
    print(f"  d=1 LG (n={n}, sigma={sigma}): SE_robust={se_r1:.4f}  SE_naive={se_n1:.4f}  "
          f"MC_SD={np.std(mc1, ddof=1):.4f}  (robust>naive, robust~=MC expected)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write_tex(pdf, regions, out_path):
    disp = [DISPLAY.get(r, r) for r in regions]
    pmat = {}
    for _, row in pdf.iterrows():
        pmat[(row["region_i"], row["region_j"])] = row["p_bonf"]
        pmat[(row["region_j"], row["region_i"])] = row["p_bonf"]
    lines = [r"\begin{tabular}{l" + "c" * (len(regions) - 1) + "}", r"\toprule",
             " & " + " & ".join(disp[1:]) + r" \\", r"\midrule"]
    for ri in range(1, len(regions)):
        cells = []
        for ci in range(len(regions) - 1):
            if ci < ri:
                pv = pmat[(regions[ri], regions[ci])]
                cells.append(f"$\\mathbf{{{pv:.1e}}}$" if pv < 0.05 else f"${pv:.1e}$")
            else:
                cells.append("")
        lines.append(f"{disp[ri]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n")


def _plot(df, pdf, regions, out_path):
    """Left: sigma_hat with robust (wide) vs naive (narrow) SE per region.
    Right: pairwise -log10(Bonferroni p) heatmap."""
    order = np.argsort(df["sigma_hat"].to_numpy())
    sig = df["sigma_hat"].to_numpy()[order]
    se_r = df["se_robust"].to_numpy()[order]
    se_n = df["se_naive"].to_numpy()[order]
    disp = [DISPLAY.get(regions[i], regions[i]) for i in order]
    y = np.arange(len(order))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1]})
    ax0.errorbar(sig, y, xerr=se_r, fmt="o", color="#1f77b4", ecolor="#1f77b4",
                 elinewidth=1.6, capsize=4, label="robust SE", zorder=2)
    ax0.errorbar(sig, y, xerr=se_n, fmt="none", ecolor="#d62728", elinewidth=2.4,
                 capsize=0, label="naive SE", zorder=3)
    ax0.set_yticks(y); ax0.set_yticklabels(disp)
    ax0.set_xlabel(r"$\hat{\sigma}$ (offset-logit MLE)")
    ax0.set_title("Per-region sigma (robust vs naive SE)")
    ax0.grid(axis="x", ls=":", alpha=0.4)
    ax0.legend(fontsize=8, loc="lower right")

    k = len(regions)
    P = np.ones((k, k))
    idx = {r: i for i, r in enumerate(regions)}
    for _, r in pdf.iterrows():
        i, j = idx[r["region_i"]], idx[r["region_j"]]
        P[i, j] = P[j, i] = r["p_bonf"]
    M = -np.log10(np.clip(P, 1e-300, 1.0))
    np.fill_diagonal(M, np.nan)
    codes = [DISPLAY.get(r, r) for r in regions]
    im = ax1.imshow(M, cmap="viridis", vmin=0, vmax=min(80, np.nanmax(M)))
    ax1.set_xticks(range(k)); ax1.set_xticklabels(codes)
    ax1.set_yticks(range(k)); ax1.set_yticklabels(codes)
    ax1.set_title(r"Pairwise $-\log_{10}$(Bonferroni $p$); white diag")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label(r"$-\log_{10} p_{\mathrm{bonf}}$  ($>1.3 \Rightarrow$ sig. @0.05)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if os.environ.get("LG_TWA_VALIDATE", "0") == "1":
        _validate()
        return

    data_dir = _repo_root / "data" / "twitch" / "graphs_processed"
    out_dir = _here / "runs" / "twitch_anova_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"twitch dyadic-robust ANOVA  seed={SEED}  quick={QUICK}  regions={REGIONS}  "
          f"d_candidates=0..{D_MAX}")

    rows = []
    for region in REGIONS:
        path = data_dir / f"{region}_graph.edges"
        if not path.exists():
            print(f"  {region}: file not found — skipping")
            continue
        t0 = time.perf_counter()
        G = _load_region(path)
        n, m = G.number_of_nodes(), G.number_of_edges()
        adj = nx.to_numpy_array(G)
        d_hat, sigma_hat, offsets, labels, aic_by_d = _select_d(adj, list(range(D_MAX + 1)))
        se_r, se_n = _dyadic_robust_se(n, sigma_hat, offsets, labels)
        rows.append(dict(region=region, display=DISPLAY.get(region, region), n=n, edges=m,
                         d_hat=d_hat, sigma_hat=sigma_hat, se_robust=se_r, se_naive=se_n,
                         aic_by_d=aic_by_d))
        print(f"  {DISPLAY.get(region, region):3s}  n={n:5d}  E={m:7d}  d_hat={d_hat}  "
              f"sigma_hat={sigma_hat:+.4f}  SE_robust={se_r:.4f}  SE_naive={se_n:.4f}  "
              f"(robust/naive={se_r / se_n:.1f}x)  [{time.perf_counter() - t0:.0f}s]")

    if len(rows) < 2:
        print("\nNeed >=2 regions.")
        return
    df = pd.DataFrame(rows)
    regions = list(df["region"])
    sigmas = df["sigma_hat"].to_numpy()
    ses = df["se_robust"].to_numpy()

    Q, dof, p_omni = _omnibus_wald(sigmas, ses)
    pdf = _pairwise_wald(regions, sigmas, ses)

    df.drop(columns=["aic_by_d"]).to_csv(out_dir / "summary.csv", index=False)
    pdf_disp = pdf.copy()
    pdf_disp["region_i"] = [DISPLAY.get(r, r) for r in pdf["region_i"]]
    pdf_disp["region_j"] = [DISPLAY.get(r, r) for r in pdf["region_j"]]
    pdf_disp.to_csv(out_dir / "pairwise.csv", index=False)
    _write_tex(pdf, regions, out_dir / "twitch_pairwise_robust.tex")
    _plot(df, pdf, regions, out_dir / "twitch_anova_robust.png")
    (out_dir / "results.json").write_text(json.dumps({
        "omnibus": {"Q": Q, "dof": dof, "p": p_omni},
        "regions": df.drop(columns=["aic_by_d"]).to_dict(orient="records"),
        "n_pairs": int(len(pdf)),
        "sig_raw": int((pdf["p_raw"] < 0.05).sum()),
        "sig_bonferroni": int((pdf["p_bonf"] < 0.05).sum()),
        "sig_fdr": int((pdf["p_fdr"] < 0.05).sum()),
    }, indent=2, default=float))

    print("\n" + "=" * 70)
    print(f"Omnibus Wald (H0: all sigma equal): Q={Q:.1f}  dof={dof}  p={p_omni:.3e}")
    print("Pairwise Wald (Bonferroni-adjusted), * = significant at 0.05:")
    for _, r in pdf.iterrows():
        star = "*" if r["p_bonf"] < 0.05 else " "
        print(f"  {DISPLAY.get(r['region_i'], r['region_i']):3s} vs "
              f"{DISPLAY.get(r['region_j'], r['region_j']):3s}  z={r['z']:+7.2f}  "
              f"p_raw={r['p_raw']:.2e}  p_bonf={r['p_bonf']:.2e} {star}")
    print(f"\nsignificant: raw={int((pdf['p_raw'] < 0.05).sum())}/{len(pdf)}  "
          f"bonferroni={int((pdf['p_bonf'] < 0.05).sum())}/{len(pdf)}  "
          f"fdr={int((pdf['p_fdr'] < 0.05).sum())}/{len(pdf)}")
    d_set = sorted(set(df["d_hat"]))
    if len(d_set) == 1:
        cross_d = (f"All regions selected d={d_set[0]} (full-graph AIC), so the "
                   f"comparison is like-for-like at a single d.")
    else:
        cross_d = (f"Regions span d in {d_set}; sigma is not strictly like-for-like "
                   f"across different d, so read cross-d pairs with care.")
    print("\nNOTE: SEs are dyadic-cluster-robust (real sampling interpretation, not "
          f"dial-able by subsample size). {cross_d}")
    print(f"Wrote {out_dir}/ (summary.csv, pairwise.csv, twitch_pairwise_robust.tex, "
          f"twitch_anova_robust.png, results.json)")


if __name__ == "__main__":
    main()
