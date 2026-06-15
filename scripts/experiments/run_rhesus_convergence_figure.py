#!/usr/bin/env python3
"""Convergence-dynamics figure for the connectome case study (thesis rhesus_iteration.png) using
the latent multi-feature LG: traces budgeted add-only growth to E_real, recording Edge / Spectrum /
GIC differences per step (coefficients + selected d/kernel/rank read from the cached fit)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
_repo = _here.parents[1]
for p in (_repo / "src", _repo / "scripts" / "closedform"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tlg_latent_gic_common as C   # noqa: E402  (loader, features, cached fit, KL scorer)
import run_tlg_twitch_gic as tw     # noqa: E402  (community feature)

NET_ID = os.environ.get("LG_CASE_NET", "rhesus_brain_1")
OUT = _here / "runs" / "rhesus_case_study"
N_POINTS = int(os.environ.get("LG_CONV_POINTS", "300"))
SEED = int(os.environ.get("LG_CONV_SEED", "100"))

DISPLAY = {"rhesus_brain_1": "Rhesus macaque brain",
           "rhesus_cerebral.cortex_1": "Rhesus macaque cerebral cortex"}


def _net_display(nid):
    return DISPLAY.get(nid, nid.replace("_", " "))


def _latent_growth_trace(G, scorer, real_den):
    """Budgeted add-only latent-LG growth to E_real (mirrors C._grow_one's softmax), recording
    edge / spectrum / GIC differences in ~N_POINTS increments. GIC uses the same ensemble-mean-density
    KL as the case study; coefficients + selected (d, kernel, rank) come from the cached fit."""
    cache = json.loads((C._out_dir("connectome") / "cache" / f"{NET_ID}.json").read_text())
    sel = cache["tlg_selected"]
    tr = next(t for t in cache["tlg_trace"]
              if t["d"] == sel["d"] and t["kernel"] == sel["kernel"] and t["k"] == sel["k"])
    d = sel["d"]
    alpha, gc, gf, lam = tr["alpha"], max(0.0, tr["gc"]), max(0.0, tr["gf"]), tr["lam"]

    adj = nx.to_numpy_array(G)
    n = G.number_of_nodes(); e_real = G.number_of_edges()
    rows, cols = np.triu_indices(n, k=1)
    Bc, _ = tw._community_feature(G, rows, cols, C.SEED, resolution=1.0)
    Bf, _ = tw._community_feature(G, rows, cols, C.SEED, resolution=C.FINE_RES)
    w_eig, U_eig = np.linalg.eigh(adj)
    L = C._latent_from_eig(w_eig, U_eig, sel["k"], rows, cols, sel["kernel"])
    real_lap = np.sort(np.linalg.eigvalsh(np.diag(adj.sum(1)) - adj))   # L = D - A (observed)

    seeds = list(C.EVAL_SEEDS)
    rngs = [np.random.default_rng(s) for s in seeds]
    As = [np.zeros((n, n)) for _ in seeds]
    step = max(1, e_real // N_POINTS)
    edge_diffs, spec_diffs, gic_vals = [], [], []
    cur = 0
    while cur < e_real:
        take = min(step, e_real - cur)
        for A, rng in zip(As, rngs):
            ne = np.where(A[rows, cols] == 0)[0]
            if len(ne) == 0:
                continue
            t = min(take, len(ne))
            D_ne = C._deg_feature_nonedges(A, rows, cols, ne, d)
            lo = alpha * D_ne + gc * Bc[ne] + gf * Bf[ne] + lam * L[ne]
            wgt = np.exp(lo - lo.max()); wgt /= wgt.sum()
            t = min(t, int(np.count_nonzero(wgt)))
            pick = rng.choice(ne, size=t, replace=False, p=wgt)
            A[rows[pick], cols[pick]] = 1.0
            A[cols[pick], rows[pick]] = 1.0
        cur = int(As[0].sum() // 2)
        dens = [scorer.compute_spectral_density(nx.from_numpy_array(A))[0] for A in As]
        laps = [np.sort(np.linalg.eigvalsh(np.diag(A.sum(1)) - A)) for A in As]
        edge_diffs.append(e_real - cur)
        spec_diffs.append(float(np.linalg.norm(np.mean(laps, axis=0) - real_lap)))
        gic_vals.append(float(entropy(real_den + 1e-10, np.mean(dens, axis=0) + 1e-10)))
    return d, edge_diffs, spec_diffs, gic_vals


def main():
    G = C._graphml(C.DATA / "connectomes" / f"{NET_ID}.graphml")
    n, m = G.number_of_nodes(), G.number_of_edges()
    scorer = C.GraphInformationCriterion(G, model="LG", dist="KL")
    real_den, _ = scorer.compute_spectral_density(G)
    C.log(f"{_net_display(NET_ID)} ({NET_ID}): n={n} m={m}  (latent LG growth trace)")
    d, edge_diff, spec_diff, gic_values = _latent_growth_trace(G, scorer, real_den)
    C.log(f"  d={d}  trace points={len(edge_diff)}  "
          f"edge_diff {edge_diff[0]}->{edge_diff[-1]}  GIC {gic_values[0]:.3f}->{gic_values[-1]:.3f}")

    # Skip the first few near-empty-graph steps, whose spectral divergence (~empty vs observed)
    # is huge and uninformative, so the panels show the meaningful convergence on a linear axis.
    warm = max(3, len(edge_diff) // 15)
    window = 10
    series = [("Edge Differences", "Difference", edge_diff),
              ("Spectrum Differences", "Difference", spec_diff),
              ("GIC Values", "GIC", gic_values)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, ylab, raw) in zip(axes, series):
        x = np.arange(len(raw))[warm:]
        r = np.asarray(raw)[warm:]
        ma = pd.Series(r).rolling(window=window).mean()
        ax.plot(x, r, color="#9ecae1", alpha=0.6, lw=1.2, label="Raw")
        ax.plot(x, ma, "r-", lw=2.0, label=f"{window}-point Moving Avg")
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Iteration", fontsize=13); ax.set_ylabel(ylab, fontsize=13)
        ax.legend(fontsize=11)
    fig.suptitle(f"{_net_display(NET_ID)}: convergence of the latent LG growth process "
                 f"(d={d}; degree + community + latent features)", fontsize=15, y=1.03)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rhesus_convergence.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    C.log(f"Wrote {OUT}/rhesus_convergence.png/.pdf")


if __name__ == "__main__":
    main()
