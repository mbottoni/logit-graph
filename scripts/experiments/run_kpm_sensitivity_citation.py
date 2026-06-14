#!/usr/bin/env python3
"""KPM parameter-sensitivity study on the arXiv HEP-Th citation network.

The spectral GIC uses the Kernel Polynomial Method (logit_graph.gic.kpm_spectral_density)
to approximate the normalized-Laplacian spectral density once n > KPM_THRESHOLD, with two
knobs: the number of Chebyshev moments M (LG_GIC_KPM_MOMENTS, default 60) and the number of
random probe vectors P (LG_GIC_KPM_PROBES, default 20). This script checks that those
defaults are converged on a real large graph -- the cit-HepTh citation network (~27k nodes)
-- so the downstream KL/GIC numbers are not artefacts of the approximation:

  1. Convergence in M: fix P, sweep M, measure the spectral-density distance (KL + total
     variation) to a high-accuracy reference (M_ref, P_ref).
  2. Convergence in P: fix M, sweep P, measure distance to the reference AND the probe-to-
     probe variability (std of the density over independent probe seeds).
  3. Absolute accuracy: on a BFS subgraph small enough for an EXACT dense eigendecomposition
     (the ground-truth density), compare KPM(default) and KPM(high) to the exact density.
  4. Report whether the default (M=60, P=20) is within tolerance of the reference / exact.

Output under runs/kpm_sensitivity_citation/ (gitignored):
  moments.csv probes.csv exact.csv   the swept errors
  kpm_sensitivity_citation.png       convergence curves + density overlay
  results.json + console summary

Env: LG_KPM_REFM (256), LG_KPM_REFP (64), LG_KPM_MGRID, LG_KPM_PGRID, LG_KPM_EXACT_CAP
(1500 BFS subgraph for the exact comparison), LG_KPM_SEED (0), LG_KPM_QUICK (0).
"""
from __future__ import annotations

import json
import os
import sys
import time
import random
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import entropy

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

warnings.filterwarnings("ignore")

from logit_graph.gic import kpm_spectral_density  # noqa: E402


def _int(env, d):
    v = os.environ.get(env); return int(v) if v else d


QUICK = os.environ.get("LG_KPM_QUICK", "0") == "1"
REF_M = _int("LG_KPM_REFM", 128 if QUICK else 256)
REF_P = _int("LG_KPM_REFP", 32 if QUICK else 64)
M_GRID = [int(x) for x in os.environ.get(
    "LG_KPM_MGRID", "20,40,60,80,120,160,200").split(",")]
P_GRID = [int(x) for x in os.environ.get("LG_KPM_PGRID", "5,10,20,40,80").split(",")]
EXACT_CAP = _int("LG_KPM_EXACT_CAP", 1500)
SEED = _int("LG_KPM_SEED", 0)
FIX_P = _int("LG_KPM_FIXP", 40)     # P held fixed while sweeping M
FIX_M = _int("LG_KPM_FIXM", 80)     # M held fixed while sweeping P
N_PROBE_SEEDS = 3 if QUICK else 6   # for probe-to-probe variability

OUT = _here / "runs" / "kpm_sensitivity_citation"
DATA = _repo_root / "data" / "citation_networks" / "cit-HepTh.txt"


def log(*a):
    print(*a, flush=True)


def _largest_cc(G):
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def _bfs_subgraph(G, cap, seed=0):
    if G.number_of_nodes() <= cap:
        return G
    root = max(G.degree, key=lambda x: x[1])[0]
    seen, frontier, s = [root], [root], {root}
    rng = random.Random(seed)
    while frontier and len(s) < cap:
        nxt = []
        for u in frontier:
            nb = list(G.neighbors(u)); rng.shuffle(nb)
            for v in nb:
                if v not in s:
                    s.add(v); seen.append(v); nxt.append(v)
                if len(s) >= cap:
                    break
            if len(s) >= cap:
                break
        frontier = nxt
    return _largest_cc(G.subgraph(seen[:cap]).copy())


def _norm_lap(G):
    return nx.normalized_laplacian_matrix(G).astype(np.float64).tocsr()


def _dist(p, q, centers):
    """KL(p||q) and total-variation between two densities on common bin centers."""
    kl = float(entropy(p + 1e-12, q + 1e-12))
    tv = float(0.5 * np.trapezoid(np.abs(p - q), centers))
    return kl, tv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        log(f"citation data not found at {DATA} (gitignored) — place cit-HepTh.txt there.")
        return
    log(f"KPM sensitivity (cit-HepTh)  ref=(M{REF_M},P{REF_P})  M_grid={M_GRID} "
        f"P_grid={P_GRID}  exact_cap={EXACT_CAP}  quick={QUICK}")

    t0 = time.perf_counter()
    Gfull = _largest_cc(nx.read_edgelist(DATA, comments="#", nodetype=int))
    L = _norm_lap(Gfull)
    n = Gfull.number_of_nodes()
    bin_edges = np.linspace(0.0, 2.0, 51)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    log(f"  full graph: n={n} E={Gfull.number_of_edges()} ({time.perf_counter()-t0:.0f}s)")

    ref, _ = kpm_spectral_density(L, n_moments=REF_M, n_probes=REF_P, seed=SEED)

    # 1. convergence in moments (fixed P)
    mrows = []
    for M in M_GRID:
        d, _ = kpm_spectral_density(L, n_moments=M, n_probes=FIX_P, seed=SEED)
        kl, tv = _dist(ref, d, centers)
        mrows.append(dict(n_moments=M, n_probes=FIX_P, kl_to_ref=kl, tv_to_ref=tv))
        log(f"  [moments] M={M:4d} (P={FIX_P}): KL_to_ref={kl:.2e} TV={tv:.2e}")
    mdf = pd.DataFrame(mrows)

    # 2. convergence in probes (fixed M) + probe-to-probe variability
    prows = []
    for P in P_GRID:
        dens = [kpm_spectral_density(L, n_moments=FIX_M, n_probes=P, seed=SEED + 100 * s)[0]
                for s in range(N_PROBE_SEEDS)]
        dmean = np.mean(dens, axis=0)
        kl, tv = _dist(ref, dmean, centers)
        probe_std = float(np.mean(np.std(dens, axis=0)))   # mean over bins of cross-seed std
        prows.append(dict(n_probes=P, n_moments=FIX_M, kl_to_ref=kl, tv_to_ref=tv,
                          probe_std=probe_std))
        log(f"  [probes]  P={P:4d} (M={FIX_M}): KL_to_ref={kl:.2e} TV={tv:.2e} "
            f"probe_std={probe_std:.2e}")
    pdf = pd.DataFrame(prows)

    # 3. absolute accuracy vs exact eigendecomposition on a BFS subgraph
    Gsub = _bfs_subgraph(Gfull, EXACT_CAP, seed=SEED)
    Ls = _norm_lap(Gsub)
    exact_ev = np.linalg.eigvalsh(Ls.toarray())
    exact_den, _ = np.histogram(exact_ev, bins=bin_edges, density=True)
    erows = []
    for (M, P, tag) in [(60, 20, "default"), (REF_M, REF_P, "high"),
                        (20, 10, "low"), (FIX_M, FIX_P, "fix")]:
        d, _ = kpm_spectral_density(Ls, n_moments=M, n_probes=P, seed=SEED)
        kl, tv = _dist(exact_den, d, centers)
        erows.append(dict(setting=tag, n_moments=M, n_probes=P,
                          kl_to_exact=kl, tv_to_exact=tv))
        log(f"  [exact n={Gsub.number_of_nodes()}] {tag:8s} (M={M},P={P}): "
            f"KL_to_exact={kl:.3e} TV={tv:.3e}")
    edf = pd.DataFrame(erows)
    kl_default = float(_dist(ref, kpm_spectral_density(L, 50, 60, 20, SEED)[0], centers)[0])

    mdf.to_csv(OUT / "moments.csv", index=False)
    pdf.to_csv(OUT / "probes.csv", index=False)
    edf.to_csv(OUT / "exact.csv", index=False)
    _plot(mdf, pdf, edf, ref, exact_den, centers, Gsub.number_of_nodes(),
          OUT / "kpm_sensitivity_citation.png", L)
    (OUT / "results.json").write_text(json.dumps({
        "n_full": n, "ref": {"M": REF_M, "P": REF_P}, "fix_P": FIX_P, "fix_M": FIX_M,
        "default_kl_to_ref": kl_default,
        "moments": mdf.to_dict(orient="records"),
        "probes": pdf.to_dict(orient="records"),
        "exact": edf.to_dict(orient="records"),
    }, indent=2, default=float))

    log("\n" + "=" * 70)
    log(f"default (M=60,P=20) vs reference (M={REF_M},P={REF_P}): "
        f"KL={kl_default:.2e} (small => converged)")
    log(f"default vs EXACT (subgraph n={Gsub.number_of_nodes()}): "
        f"KL={float(edf[edf.setting=='default'].kl_to_exact.iloc[0]):.2e}")
    log(f"Wrote {OUT}/")


def _plot(mdf, pdf, edf, ref, exact_den, centers, n_sub, out_path, L):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].plot(mdf["n_moments"], mdf["kl_to_ref"], "o-", color="#0072B2", label="KL to ref")
    ax[0].plot(mdf["n_moments"], mdf["tv_to_ref"], "s--", color="#E69F00", label="TV to ref")
    ax[0].axvline(60, color="k", ls=":", lw=1, label="default M=60")
    ax[0].set_yscale("log"); ax[0].set_xlabel("# Chebyshev moments M")
    ax[0].set_ylabel("distance to reference density"); ax[0].grid(alpha=.3)
    ax[0].set_title(f"Convergence in moments (P={int(mdf['n_probes'].iloc[0])})"); ax[0].legend()

    ax[1].plot(pdf["n_probes"], pdf["kl_to_ref"], "o-", color="#0072B2", label="KL to ref")
    ax[1].plot(pdf["n_probes"], pdf["probe_std"], "^--", color="#009E73",
               label="probe-to-probe std")
    ax[1].axvline(20, color="k", ls=":", lw=1, label="default P=20")
    ax[1].set_yscale("log"); ax[1].set_xlabel("# probe vectors P")
    ax[1].set_ylabel("distance / variability"); ax[1].grid(alpha=.3)
    ax[1].set_title(f"Convergence in probes (M={int(pdf['n_moments'].iloc[0])})"); ax[1].legend()

    d_def, _ = kpm_spectral_density(L, n_moments=60, n_probes=20, seed=0)
    ax[2].plot(centers, ref, "-", color="#000000", lw=2, label=f"KPM ref")
    ax[2].plot(centers, d_def, "--", color="#D55E00", lw=1.6, label="KPM default (60,20)")
    ax[2].plot(centers, exact_den, ":", color="#56B4E9", lw=1.6,
               label="exact")
    ax[2].set_xlabel("normalized-Laplacian eigenvalue"); ax[2].set_ylabel("density")
    ax[2].set_title("Spectral density: KPM vs exact"); ax[2].grid(alpha=.3); ax[2].legend()
    fig.suptitle("KPM parameter sensitivity on the arXiv cit-HepTh citation network",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out_path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
