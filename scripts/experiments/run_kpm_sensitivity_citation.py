#!/usr/bin/env python3
"""KPM parameter-sensitivity study on the arXiv HEP-Th citation network (~27k nodes): checks the
default Chebyshev moments M=60 / probes P=20 of the normalized-Laplacian KPM spectral density are
converged vs a high-accuracy reference and an exact small-subgraph eigendecomposition."""
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
from scipy.sparse.linalg import eigsh
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

from logit_graph.gic import kpm_spectral_density, _jackson_kernel  # noqa: E402


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


def _adj_matrix(G):
    return nx.adjacency_matrix(G).astype(np.float64).tocsr()


def _adj_bounds(A, margin=1.02):
    """Symmetric spectral bounds [lo, hi] of the (binary) adjacency, estimated by Lanczos; the
    margin keeps the rescaled spectrum strictly inside (-1, 1) for the Chebyshev recurrence (the
    raw adjacency spectrum is graph-dependent, unlike the normalized Laplacian's [0, 2])."""
    hi = float(eigsh(A, k=1, which="LA", return_eigenvectors=False)[0])
    lo = float(eigsh(A, k=1, which="SA", return_eigenvectors=False)[0])
    return margin * lo, margin * hi


def _adj_kpm_density(A, lo, hi, n_bins=50, n_moments=60, n_probes=20, seed=0):
    """KPM spectral density of a symmetric matrix over a general range [lo, hi]: same Chebyshev
    machinery as gic.kpm_spectral_density but rescaling by the supplied bounds (x=(lambda-c)/d,
    c=(hi+lo)/2, d=(hi-lo)/2) so it applies to the raw adjacency. Returns (density, edges)."""
    n = A.shape[0]
    c = 0.5 * (hi + lo); d = 0.5 * (hi - lo)
    H = ((A - c * sp.eye(n, format="csr")) / d).tocsr()
    rng = np.random.default_rng(seed)
    moments = np.zeros(n_moments)
    for _ in range(n_probes):
        v0 = rng.choice([-1.0, 1.0], size=n)
        v_prev = v0.copy()
        moments[0] += float(np.dot(v0, v_prev))
        if n_moments > 1:
            v_curr = H @ v0
            moments[1] += float(np.dot(v0, v_curr))
            for k in range(2, n_moments):
                v_next = 2.0 * (H @ v_curr) - v_prev
                moments[k] += float(np.dot(v0, v_next))
                v_prev, v_curr = v_curr, v_next
    moments /= float(n_probes * n)

    g = _jackson_kernel(n_moments)
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    x = (centers - c) / d
    Tprev = np.ones_like(x); Tcurr = x.copy()
    f = g[0] * moments[0] * Tprev
    if n_moments > 1:
        f += 2.0 * g[1] * moments[1] * Tcurr
    for k in range(2, n_moments):
        Tnext = 2.0 * x * Tcurr - Tprev
        f += 2.0 * g[k] * moments[k] * Tnext
        Tprev, Tcurr = Tcurr, Tnext
    weight = 1.0 / (np.pi * np.sqrt(np.clip(1.0 - x * x, 1e-12, None)))
    density = np.maximum(weight * f, 0.0)
    integral = float(np.trapezoid(density, centers))
    if integral > 0:
        density /= integral
    return density, bin_edges


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
    lap_def, _ = kpm_spectral_density(L, n_moments=60, n_probes=20, seed=SEED)
    kl_default = float(_dist(ref, lap_def, centers)[0])
    lap = dict(centers=centers, ref=ref, default=lap_def, exact=exact_den)

    # 4. adjacency-matrix spectral density (raw binary A): same KPM ref/default vs exact
    # overlay as the Laplacian panel, but over the adjacency's own (wide) spectral range.
    A = _adj_matrix(Gfull)
    a_lo, a_hi = _adj_bounds(A)
    a_centers = 0.5 * (np.linspace(a_lo, a_hi, 51)[:-1] + np.linspace(a_lo, a_hi, 51)[1:])
    a_ref, a_edges = _adj_kpm_density(A, a_lo, a_hi, n_moments=REF_M, n_probes=REF_P, seed=SEED)
    a_def, _ = _adj_kpm_density(A, a_lo, a_hi, n_moments=60, n_probes=20, seed=SEED)
    a_exact, _ = np.histogram(np.linalg.eigvalsh(_adj_matrix(Gsub).toarray()),
                              bins=a_edges, density=True)
    adj = dict(centers=a_centers, ref=a_ref, default=a_def, exact=a_exact)
    adj_kl_def_ref = float(_dist(a_ref, a_def, a_centers)[0])
    adj_kl_def_exact = float(_dist(a_exact, a_def, a_centers)[0])
    log(f"  [adjacency] spectral range=[{a_lo:.1f},{a_hi:.1f}]  "
        f"KL(default,ref)={adj_kl_def_ref:.2e}  KL(default,exact)={adj_kl_def_exact:.2e}")

    mdf.to_csv(OUT / "moments.csv", index=False)
    pdf.to_csv(OUT / "probes.csv", index=False)
    edf.to_csv(OUT / "exact.csv", index=False)
    _plot(mdf, pdf, lap, adj, OUT / "kpm_sensitivity_citation.png")
    (OUT / "results.json").write_text(json.dumps({
        "n_full": n, "ref": {"M": REF_M, "P": REF_P}, "fix_P": FIX_P, "fix_M": FIX_M,
        "default_kl_to_ref": kl_default,
        "adjacency": {"spectral_range": [a_lo, a_hi],
                      "kl_default_to_ref": adj_kl_def_ref,
                      "kl_default_to_exact": adj_kl_def_exact},
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


def _density_panel(ax, dens, xlabel, title):
    """Overlay KPM reference, KPM default (60,20), and exact density on a common grid."""
    ax.plot(dens["centers"], dens["ref"], "-", color="#000000", lw=2, label="KPM ref")
    ax.plot(dens["centers"], dens["default"], "--", color="#D55E00", lw=1.6,
            label="KPM default (60,20)")
    ax.plot(dens["centers"], dens["exact"], ":", color="#56B4E9", lw=1.6, label="exact")
    ax.set_xlabel(xlabel); ax.set_ylabel("density")
    ax.set_title(title); ax.grid(alpha=.3); ax.legend()


def _plot(mdf, pdf, lap, adj, out_path):
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
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

    _density_panel(ax[2], lap, "normalized-Laplacian eigenvalue",
                   "Spectral density: KPM vs exact")
    _density_panel(ax[3], adj, "adjacency eigenvalue",
                   "Adjacency spectral density: KPM vs exact")
    fig.suptitle("KPM parameter sensitivity on the arXiv cit-HepTh citation network",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out_path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
