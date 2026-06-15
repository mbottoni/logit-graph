#!/usr/bin/env python3
"""Connectomes ANOVA for the Temporal Logit-Graph (TLG) on BOTH parameters (sigma, alpha): fit
logit P[edge] = sigma + alpha*D per connectome with a 2x2 dyadic-cluster-robust SE, then compare
each parameter across the 18 networks (omnibus Cochran-Q + pairwise Wald). Fit/SE reused from twitch."""
from __future__ import annotations

import glob
import json
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

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
for p in (_repo_root / "src", _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")

# Reuse the dataset-agnostic TLG fit + 2-param dyadic-robust SE + tests from the twitch script.
import run_tlg_twitch_anova_robust as TW  # noqa: E402

PARAMS = TW.PARAMS
PLABEL = TW.PLABEL


def _int(env, d):
    v = os.environ.get(env); return int(v) if v else d


QUICK = os.environ.get("LG_TCA_QUICK", "0") == "1"
D = _int("LG_TCA_D", 1)
NMIN = _int("LG_TCA_NMIN", 10)        # skip degenerate tiny graphs
SEED = _int("LG_TCA_SEED", 12345)


def log(*a):
    print(*a, flush=True)


def _short(stem):
    """Compact display label for a connectome filename stem."""
    s = stem.replace("_neural", "").replace(".synaptic", "").replace("_brain", ".br") \
            .replace(".cortex", ".ctx").replace("interareal.cortical.network", "iacn") \
            .replace("norvegicus", "norv").replace("herm_pharynx", "herm")
    return (s[:16]) if len(s) > 16 else s


def _load_connectome(path):
    """Undirected, binarized (weights dropped), self-loop-free largest CC, relabeled."""
    G = nx.Graph(nx.read_graphml(path))     # undirected simple graph
    G.remove_edges_from(nx.selfloop_edges(G))
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def _plot(df, pdf, labels, out_path):
    k = len(labels)
    fig, axes = plt.subplots(2, 2, figsize=(max(13, 0.5 * k + 6), 13))
    for ri, par in enumerate(PARAMS):
        order = np.argsort(df[f"{par}_hat"].to_numpy())
        val = df[f"{par}_hat"].to_numpy()[order]
        se_r = df[f"se_{par}_robust"].to_numpy()[order]
        disp = [labels[i] for i in order]
        y = np.arange(len(order))
        ax0 = axes[ri][0]
        ax0.errorbar(val, y, xerr=se_r, fmt="o", color="#1f77b4", ecolor="#1f77b4",
                     elinewidth=1.4, capsize=3, ms=4)
        ax0.set_yticks(y); ax0.set_yticklabels(disp, fontsize=7)
        ax0.set_xlabel(PLABEL[par]); ax0.set_title(f"Per-connectome {par} (robust SE)")
        ax0.grid(axis="x", ls=":", alpha=0.4)
        # Bonferroni heatmap
        P = np.ones((k, k)); idx = {labels[i]: i for i in range(k)}
        for _, r in pdf[par].iterrows():
            i, j = idx[r["region_i"]], idx[r["region_j"]]
            P[i, j] = P[j, i] = r["p_bonf"]
        Mlog = -np.log10(np.clip(P, 1e-300, 1.0)); np.fill_diagonal(Mlog, np.nan)
        ax1 = axes[ri][1]
        im = ax1.imshow(Mlog, cmap="viridis", vmin=0, vmax=min(80, np.nanmax(Mlog)))
        ax1.set_xticks(range(k)); ax1.set_xticklabels(labels, rotation=90, fontsize=6)
        ax1.set_yticks(range(k)); ax1.set_yticklabels(labels, fontsize=6)
        ax1.set_title(rf"{par}: pairwise $-\log_{{10}}$(Bonferroni $p$)")
        cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label(r"$-\log_{10} p_{\mathrm{bonf}}$ ($>1.3\Rightarrow$ sig.@0.05)")
    fig.suptitle(f"Connectomes TLG ANOVA: sigma and alpha across {k} networks "
                 "(dyadic-cluster-robust SE)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98]); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    files = sorted(glob.glob(str(_repo_root / "data" / "connectomes" / "*.graphml")))
    if QUICK:
        files = files[:4]
    out_dir = _here / "runs" / "connectomes_tlg_anova_robust"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"connectomes TLG dyadic-robust ANOVA (sigma, alpha)  d={D}  {len(files)} graphml")

    rows, covs, labels = [], {}, []
    for f in files:
        stem = Path(f).stem
        try:
            G = _load_connectome(f)
        except Exception as ex:
            log(f"  {stem}: load failed ({ex}) — skipping"); continue
        n, m = G.number_of_nodes(), G.number_of_edges()
        if n < NMIN:
            log(f"  {stem}: n={n} < {NMIN} — skipping"); continue
        t0 = time.perf_counter()
        adj = nx.to_numpy_array(G, weight=None)        # binarized adjacency
        sigma, alpha, Dv, lab = TW._fit_tlg(adj, D)
        se, se_n, V = TW._dyadic_robust_se2(n, sigma, alpha, Dv, lab)
        # very dense connectomes (density > ~0.5) perfectly separate -> the MLE diverges
        # (|sigma| huge, SE -> 0). Such fits are not usable for an inverse-variance test.
        if (not np.all(np.isfinite([sigma, alpha, se["sigma"], se["alpha"]]))
                or se["sigma"] < 1e-4 or se["alpha"] < 1e-4 or abs(sigma) > 200):
            log(f"  {_short(stem):16s}  n={n:5d}  E={m:7d}  density={2*m/(n*(n-1)):.2f}  "
                f"DEGENERATE fit (quasi-perfect separation) — excluded")
            continue
        short = _short(stem)
        covs[short] = V; labels.append(short)
        rows.append(dict(connectome=stem, label=short, n=n, edges=m,
                         sigma_hat=sigma, alpha_hat=alpha,
                         se_sigma_robust=se["sigma"], se_sigma_naive=se_n["sigma"],
                         se_alpha_robust=se["alpha"], se_alpha_naive=se_n["alpha"]))
        log(f"  {short:16s}  n={n:5d}  E={m:7d}  sigma={sigma:+.3f}(SE {se['sigma']:.3f})  "
            f"alpha={alpha:+.3f}(SE {se['alpha']:.3f})  [{time.perf_counter()-t0:.0f}s]")

    if len(rows) < 2:
        log("\nNeed >=2 connectomes."); return
    df = pd.DataFrame(rows)
    regions = list(df["label"])

    omni, pdf = {}, {}
    for par in PARAMS:
        vals = df[f"{par}_hat"].to_numpy(); ses = df[f"se_{par}_robust"].to_numpy()
        omni[par] = TW._omnibus_wald(vals, ses)
        pdf[par] = TW._pairwise_wald(regions, vals, ses)
    pjoint = TW._pairwise_joint(regions, df["sigma_hat"].to_numpy(),
                                df["alpha_hat"].to_numpy(), [covs[r] for r in regions])

    df.to_csv(out_dir / "summary.csv", index=False)
    for par in PARAMS:
        pdf[par].to_csv(out_dir / f"pairwise_{par}.csv", index=False)
    pjoint.to_csv(out_dir / "pairwise_joint.csv", index=False)
    _plot(df, pdf, regions, out_dir / "connectomes_tlg_anova_robust.png")
    (out_dir / "results.json").write_text(json.dumps({
        "omnibus": {par: {"Q": omni[par][0], "dof": omni[par][1], "p": omni[par][2]}
                    for par in PARAMS},
        "connectomes": df.to_dict(orient="records"),
        "n_pairs": int(len(pjoint)),
        "sig_bonferroni": {par: int((pdf[par]["p_bonf"] < 0.05).sum()) for par in PARAMS},
        "sig_bonferroni_joint": int((pjoint["p_bonf"] < 0.05).sum()),
    }, indent=2, default=float))

    log("\n" + "=" * 70)
    for par in PARAMS:
        Q, dof, p = omni[par]
        nb = int((pdf[par]["p_bonf"] < 0.05).sum())
        print(f"[{par}] omnibus Q={Q:.1f} dof={dof} p={p:.2e} | "
              f"Bonferroni-significant pairs: {nb}/{len(pdf[par])}")
    nj = int((pjoint["p_bonf"] < 0.05).sum())
    print(f"[joint (sigma,alpha) 2-df] Bonferroni-significant pairs: {nj}/{len(pjoint)}")
    print("\nNOTE: SEs are dyadic-cluster-robust. With 153 pairwise tests Bonferroni is strict.")
    print(f"Wrote {out_dir}/")


if __name__ == "__main__":
    main()
