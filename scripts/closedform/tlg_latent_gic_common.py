#!/usr/bin/env python3
"""Shared machinery for the per-dataset latent-TLG GIC experiments.

This module holds the unified Temporal Logit-Graph (TLG) fit + the baseline families +
the (lossless) performance optimizations; the per-dataset entry points
(run_tlg_<dataset>_latent_gic.py) only supply a loader and call :func:`run_one`. Run all
datasets + a combined ranking with run_tlg_all_latent_gic.py.

Fit a representative network from a dataset with the unified Temporal Logit-Graph (TLG)
and rank families by the spectral GIC / raw KL.

The TLG here is the IDENTIFIABLE, SBM-beating model with three exogenous feature groups
(all recoverable by MLE in the add+remove Bernoulli model — validated on synthetic data;
none reads a dyad's own state, so the comparison is as honest as SBM, which uses a fixed
Louvain partition of the same graph):

  * D  — degree feature (depth d), RECOMPUTED on the GENERATED graph each growth batch
         (the forward/honest part). Captures hubs (regime A: twitch, arxiv, connectomes).
  * Bc, Bf — same-coarse-community and same-fine-community indicators from two FIXED
         Louvain partitions of G (coarse = community like SBM; fine = local-density proxy).
         Captures modularity (regime A).
  * L  — latent-space proximity from a FIXED adjacency spectral embedding (ASE) of G:
         top-k eigenvectors of A scaled by sqrt(|eigenvalue|); L_ij = z_i . z_j
         (standardized). Captures triangles/clustering SMOOTHLY where discrete blocks
         cannot (regime B: facebook/twitter/gplus/human-connectomes). The ADJACENCY
         embedding is used (not the normalized Laplacian) so we never peek at the operator
         whose spectral density the GIC scores. n_params counts only the coefficients
         (alpha, gc, gf, lam) = 4 — the embedding/partition DOF are not counted, exactly
         as SBM counts k(k+1)/2 block probabilities but not its n node assignments.

Generation: budgeted add-only growth to the real edge count E_real (per-batch budget ->
intercept irrelevant, edges matched by construction). Each non-edge is added with prob
proportional to exp(alpha*D + gc*Bc + gf*Bf + lam*L).

Estimator (FAIR — identical for TLG and every baseline): the KL is computed on the
ENSEMBLE-MEAN normalized-Laplacian spectral density over NRUNS independent draws (this is
what the closed-form baselines and SBM already use; the older twitch script used an
optimistic best-of-draws for the TLG only). Parameters are tuned by minimizing the
ensemble KL on TUNE seeds and the reported KL/GIC are scored on disjoint HELD-OUT seeds.

Datasets (representative graph each; largest connected component, undirected, size-capped
by BFS ball for the big ones): twitch, twitter, facebook, arxiv-cit, gplus, connectome
(animal), human-connectome (brain). Output under runs/tlg_multidataset_gic/ (gitignored).

Env knobs: LG_TLM_DATASETS (comma list; default all), LG_TLM_NRUNS (6),
LG_TLM_CAP (1500 BFS cap), LG_TLM_K (30 growth batches), LG_TLM_SEARCH (30 NM iters),
LG_TLM_KLIST (2,4,8 latent ranks to try), LG_TLM_FINE_RES (8), LG_TLM_SEED (12345),
LG_TLM_QUICK (0 -> tiny smoke).
"""
from __future__ import annotations

import contextlib
import glob
import json
import multiprocessing as mp
import os
import random
import sys
import time
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import networkx as nx
import pandas as pd
import scipy.sparse as sp
from scipy.stats import entropy
from scipy.optimize import minimize

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def log(*a):
    print(*a, flush=True)


from logit_graph.gic import (  # noqa: E402
    GraphInformationCriterion, kpm_spectral_density, KPM_THRESHOLD)
from logit_graph.temporal import fit_growth_params  # noqa: E402
from logit_graph.lg_features import build_pair_dataset  # noqa: E402
from logit_graph.sbm import generate_sbm_from_real  # noqa: E402
import run_tlg_twitch_gic as tw  # noqa: E402  (closed_form_params, _baseline_generators, _community_feature)

OUT_DIR = _here / "runs" / "tlg_latent_gic"
DATA = _repo_root / "data"


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


def _float(env, default):
    raw = os.environ.get(env)
    return float(raw) if raw is not None else default


QUICK = os.environ.get("LG_TLM_QUICK", "0") == "1"
NRUNS = _int("LG_TLM_NRUNS", 3 if QUICK else 6)         # ensemble draws for mean density
CAP = _int("LG_TLM_CAP", 800 if QUICK else 1500)         # BFS size cap for big graphs
TLG_K = _int("LG_TLM_K", 15 if QUICK else 30)            # growth batches to reach E_real
SEARCH = _int("LG_TLM_SEARCH", 12 if QUICK else 30)      # Nelder-Mead iterations
KLIST = [int(x) for x in os.environ.get("LG_TLM_KLIST", "2,4,8").split(",")]
KERNELS = os.environ.get("LG_TLM_KERNELS", "dot,dist").split(",")  # latent kernels to try
# Opt-in fast spectral density (direct sparse Laplacian, KPM path). It differs from the
# networkx Laplacian only at fp level (~1e-15), but the tuning search is chaotic, so that
# can perturb the final tuned KL by ~1e-3 (conclusions unchanged). Default OFF keeps runs
# bit-reproducible; enable for ~2-3x faster scoring on large/dense graphs (e.g. big sweeps).
FAST_SPECTRAL = os.environ.get("LG_TLM_FAST_SPECTRAL", "0") == "1"
FINE_RES = _float("LG_TLM_FINE_RES", 8.0)
SEED = _int("LG_TLM_SEED", 12345)
DEG_D = 1   # the fast separable degree feature (_deg_feature_nonedges) assumes d=1
assert DEG_D == 1, "the vectorized degree feature is specialized to d=1"
TLG_N_PARAMS = 4
TUNE_SEEDS = range(4)
EVAL_SEEDS = range(100, 100 + NRUNS)
FAMILIES = ["TLG", "ER", "BA", "WS", "KR", "GRG", "SBM"]


# --------------------------------------------------------------------------- loaders
def _finalize(G):
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if G.number_of_nodes() > CAP:                         # BFS ball around the top hub
        root = max(G.degree, key=lambda x: x[1])[0]
        seen, frontier, s = [root], [root], {root}
        rng = random.Random(0)
        while frontier and len(s) < CAP:
            nxt = []
            for u in frontier:
                nb = list(G.neighbors(u)); rng.shuffle(nb)
                for v in nb:
                    if v not in s:
                        s.add(v); seen.append(v); nxt.append(v)
                    if len(s) >= CAP:
                        break
                if len(s) >= CAP:
                    break
            frontier = nxt
        G = G.subgraph(seen[:CAP]).copy()
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(G)


def _edges(p):
    return _finalize(nx.read_edgelist(p, comments="#", nodetype=int))


def _graphml(p):
    return _finalize(nx.read_graphml(p))


def _citation():
    G = nx.read_edgelist(DATA / "citation_networks" / "cit-HepTh.txt",
                         comments="#", nodetype=int)
    return _finalize(G)


def _smallest_ego(pattern):
    files = sorted(glob.glob(str(DATA / pattern)),
                   key=lambda f: sum(1 for _ in open(f)))
    return files


DATASETS = {
    "twitch":     lambda: _edges(DATA / "twitch" / "graphs_processed" / "PTBR_graph.edges"),
    "twitter":    lambda: _edges(_smallest_ego("misc/twitter/*.edges")[400]),
    "facebook":   lambda: _edges(DATA / "misc" / "facebook" / "348.edges"),
    "arxiv":      _citation,
    "gplus":      lambda: _edges(_smallest_ego("misc/gplus/*.edges")[100]),
    "connectome": lambda: _graphml(DATA / "connectomes" / "c.elegans.herm_pharynx_1.graphml"),
    "human":      lambda: _graphml(sorted(glob.glob(
                      str(DATA / "brain_graph" / "oasis3_graphmls_scale1" / "*.graphml")))[0]),
}


# --------------------------------------------------------------------------- features
def _latent_from_eig(w, U, k, rows, cols, kind):
    """Latent feature from a rank-k adjacency spectral embedding z (eigvecs scaled by
    sqrt|eigval|), reusing a precomputed eigendecomposition (w, U) of A. kind="dot":
    L_ij = z_i . z_j (RDPG inner product — low-rank / community structure). kind="dist":
    L_ij = -||z_i - z_j|| (Hoff latent-space distance — geometric / spatial structure,
    e.g. brain connectomes). Standardized."""
    idx = np.argsort(-np.abs(w))[:k]
    z = U[:, idx] * np.sqrt(np.abs(w[idx]))
    if kind == "dist":
        L = -np.sqrt(((z[rows] - z[cols]) ** 2).sum(1))
    else:
        L = (z[rows] * z[cols]).sum(1)
    return (L - L.mean()) / (L.std() + 1e-9)


def _norm_laplacian(A):
    """Normalized Laplacian I - D^{-1/2} A D^{-1/2} as CSR, directly from a numpy
    adjacency (matches networkx.normalized_laplacian_matrix; avoids the nx round-trip)."""
    As = sp.csr_matrix(A)
    n = A.shape[0]
    deg = np.asarray(As.sum(1)).ravel()
    dinv = np.zeros(n)
    nz = deg > 0
    dinv[nz] = 1.0 / np.sqrt(deg[nz])
    D = sp.diags(dinv)
    return (sp.identity(n, format="csr") - D @ As @ D).tocsr()


def _spectral_density_fast(A, scorer):
    """Same spectral density as scorer.compute_spectral_density, but for the KPM path
    (n > threshold) builds the Laplacian straight from the adjacency (validated identical
    to the nx path, ~2-3x faster on large/dense graphs). Small graphs use the exact nx
    path unchanged."""
    n = A.shape[0]
    if FAST_SPECTRAL and n > KPM_THRESHOLD:
        return kpm_spectral_density(_norm_laplacian(A), n_bins=50)
    return scorer.compute_spectral_density(nx.from_numpy_array(A))


def _deg_feature_nonedges(A, rows, cols, ne):
    """Degree feature D_ij = log(1+S_i)+log(1+S_j) over the non-edge set, with
    S_v^(1) = deg(v) + sum_{u~v} deg(u) = (deg + A @ deg)_v. For non-edges adj[i,j]=0, so
    no layer-2 subtraction applies -> identical to build_pair_dataset(d=1, "bounded",
    layer2=True) on those pairs (validated, maxdiff 0), but vectorized (no n^2 loop)."""
    deg = A.sum(1)
    f = np.log1p(deg + A @ deg)
    return f[rows[ne]] + f[cols[ne]]


def _metrics(G):
    n = G.number_of_nodes()
    e = G.number_of_edges()
    try:
        assort = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        assort = float("nan")
    return dict(nodes=int(n), edges=int(e), density=float(nx.density(G)),
                avg_degree=float(2 * e / n) if n else 0.0,
                clustering=float(nx.average_clustering(G)), assortativity=assort)


def _gmetrics(G):
    """Structural metrics of a generated/representative graph, NaN-safe when None."""
    if G is None:
        return dict(nodes=np.nan, edges=np.nan, density=np.nan, avg_degree=np.nan,
                    clustering=np.nan, assortativity=np.nan)
    return _metrics(G)


# --------------------------------------------------------------------------- TLG
def _grow_one(n, rows, cols, alpha, gc, gf, lam, Bc, Bf, L, e_real, seed):
    """One budgeted add-only growth to E_real; returns the final adjacency."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    batch = max(1, e_real // TLG_K)
    for _ in range(TLG_K + 10):
        cur = int(A.sum() // 2)
        if cur >= e_real:
            break
        ne = np.where(A[rows, cols] == 0)[0]
        if len(ne) == 0:
            break
        take = min(batch, e_real - cur, len(ne))
        D_ne = _deg_feature_nonedges(A, rows, cols, ne)  # d=1 separable fast path
        lo = alpha * D_ne + gc * Bc[ne] + gf * Bf[ne] + lam * L[ne]
        w = np.exp(lo - lo.max()); w /= w.sum()
        # underflow guard: when the softmax is so peaked that fewer than `take` weights
        # are non-zero, cap the batch at the count of non-zero dyads (a no-op otherwise,
        # so non-degenerate runs stay bit-identical to the unclipped original)
        take = min(take, int(np.count_nonzero(w)))
        pick = rng.choice(ne, size=take, replace=False, p=w)
        A[rows[pick], cols[pick]] = 1.0
        A[cols[pick], rows[pick]] = 1.0
    return A


def _ensemble_kl(x, n, rows, cols, Bc, Bf, L, e_real, scorer, real_den, seeds,
                 want_graph=False):
    """KL of the ENSEMBLE-MEAN spectral density over `seeds` growth draws (SBM's
    estimator). Returns (kl, representative_graph_or_None)."""
    a, gc, gf, lam = x
    dens, rep = [], None
    for s in seeds:
        A = _grow_one(n, rows, cols, a, max(0.0, gc), max(0.0, gf), lam,
                      Bc, Bf, L, e_real, s)
        dens.append(_spectral_density_fast(A, scorer)[0])
        if rep is None and want_graph:
            rep = nx.from_numpy_array(A)
    avg = np.mean(dens, axis=0)
    kl = float(entropy(real_den + 1e-10, avg + 1e-10))
    return (kl, rep) if want_graph else (kl, None)


def fit_tlg(G, scorer, real_den):
    """Tune (alpha, gc, gf, lam) + latent rank k by min ensemble-KL on TUNE seeds; the
    reported KL is scored on disjoint HELD-OUT seeds. alpha warm-started from the degree
    MLE. All features exogenous/identifiable; n_params = 4 (coefficients only)."""
    n = G.number_of_nodes()
    adj = nx.to_numpy_array(G)
    e_real = G.number_of_edges()
    rows, cols = np.triu_indices(n, k=1)
    t0 = time.perf_counter()
    Dr, lab = build_pair_dataset(adj, d=DEG_D, mode="bounded", layer2=True)
    alpha0 = float(fit_growth_params(Dr, lab)["alpha"])
    Bc, kc = tw._community_feature(G, rows, cols, SEED, resolution=1.0)
    Bf, kf = tw._community_feature(G, rows, cols, SEED, resolution=FINE_RES)
    w_eig, U_eig = np.linalg.eigh(adj)   # ASE embedding computed ONCE, sliced per (k,kind)
    log(f"    TLG fit: n={n} E={e_real} α0={alpha0:.3f} comms(coarse={kc},fine={kf}); "
        f"latent kernels {KERNELS} x k in {KLIST}; tuning by min ensemble-KL ...")

    best = {"kl": float("inf")}
    for kind in KERNELS:
        for k in KLIST:
            L = _latent_from_eig(w_eig, U_eig, k, rows, cols, kind)
            nev = {"n": 0}

            def obj(x):
                nev["n"] += 1
                kl, _ = _ensemble_kl(x, n, rows, cols, Bc, Bf, L, e_real, scorer,
                                     real_den, TUNE_SEEDS)
                return kl

            res = minimize(obj, x0=[alpha0, 2.0, 3.0, 2.0], method="Nelder-Mead",
                           options={"maxiter": SEARCH, "xatol": 0.1, "fatol": 3e-4})
            kl_eval, rep = _ensemble_kl(res.x, n, rows, cols, Bc, Bf, L, e_real,
                                        scorer, real_den, EVAL_SEEDS, want_graph=True)
            log(f"      {kind} k={k}: tuneKL={res.fun:.4f} evalKL={kl_eval:.4f} "
                f"(α={res.x[0]:.2f} γc={res.x[1]:.2f} γf={res.x[2]:.2f} λ={res.x[3]:.2f}) "
                f"{nev['n']} evals")
            if kl_eval < best["kl"]:
                best = dict(kl=kl_eval, k=k, kind=kind, x=res.x, graph=rep)

    x = best["x"]
    gic = 2.0 * best["kl"] + 2.0 * TLG_N_PARAMS
    param = (f"d={DEG_D}, {best['kind']} k={best['k']}, α={x[0]:.2f}, "
             f"γc={max(0,x[1]):.2f}, γf={max(0,x[2]):.2f}, λ={x[3]:.2f}")
    log(f"    TLG: best {best['kind']} k={best['k']} KL={best['kl']:.4f} GIC={gic:.4f} "
        f"({time.perf_counter()-t0:.1f}s)")
    return dict(gic=gic, kl=best["kl"], n_params=TLG_N_PARAMS, graph=best["graph"],
                param=param)


def baseline_gic(G, gen_fn, n_params, real_den, scorer):
    """Ensemble-mean-density KL for a baseline family (same estimator as TLG)."""
    dens, rep = [], None
    for s in EVAL_SEEDS:
        try:
            g = gen_fn(SEED + s)
        except Exception:
            continue
        if rep is None:
            rep = g
        dens.append(scorer.compute_spectral_density(g)[0])
    if not dens:
        return None
    avg = np.mean(dens, axis=0)
    kl = float(entropy(real_den + 1e-10, avg + 1e-10))
    return dict(gic=2.0 * kl + 2.0 * n_params, kl=kl, n_params=n_params, graph=rep)


# --------------------------------------------------------------------------- driver
def process_dataset(name, loader):
    t0 = time.perf_counter()
    try:
        G = loader()
    except Exception as ex:
        log(f"  {name}: load failed ({ex}) — skipping")
        return None
    rm = _metrics(G)
    log(f"\n=== {name}: n={G.number_of_nodes()} E={rm['edges']} dens={rm['density']:.4f} "
        f"clust={rm['clustering']:.3f} assort={rm['assortativity']:.3f} ===")
    scorer = GraphInformationCriterion(G, model="LG", dist="KL")
    real_den, _ = scorer.compute_spectral_density(G)

    rows = [dict(dataset=name, model="Real", n_params=np.nan, gic=np.nan,
                 gic_fit=np.nan, gic_penalty=np.nan, kl=np.nan, param="—", **rm)]

    tlg = fit_tlg(G, scorer, real_den)
    rows.append(dict(dataset=name, model="TLG", n_params=tlg["n_params"], gic=tlg["gic"],
                     gic_fit=2.0 * tlg["kl"], gic_penalty=2.0 * tlg["n_params"],
                     kl=tlg["kl"], param=tlg["param"],
                     **(_metrics(tlg["graph"]) if tlg["graph"] is not None else
                        dict(edges=np.nan, density=np.nan, clustering=np.nan,
                             assortativity=np.nan))))

    cf = tw.closed_form_params(G)
    gens = tw._baseline_generators(G, cf)
    cf_params = dict(ER=f"p={cf['ER']:.4f}", BA=f"m={cf['BA']}",
                     WS=f"k={cf['WS_k']}, p={cf['WS_p']:.3f}", KR=f"d={cf['KR']}",
                     GRG=f"r={cf['GRG']:.3f}", SBM="Louvain blocks")
    for fam in ("ER", "BA", "WS", "KR", "GRG", "SBM"):
        gen_fn, n_params = gens[fam]
        if fam == "SBM":
            _, n_params = generate_sbm_from_real(G, seed=SEED)
        res = baseline_gic(G, gen_fn, n_params, real_den, scorer)
        if res is None:
            log(f"  {fam}: generation failed — skipping")
            continue
        rows.append(dict(dataset=name, model=fam, n_params=res["n_params"],
                         gic=res["gic"], gic_fit=2.0 * res["kl"],
                         gic_penalty=2.0 * res["n_params"], kl=res["kl"],
                         param=cf_params[fam], **_metrics(res["graph"])))
        log(f"  {fam}: np={res['n_params']} KL={res['kl']:.4f} GIC={res['gic']:.4f}")

    df = pd.DataFrame(rows)
    models = df[df["model"] != "Real"].sort_values("kl").copy()
    models["kl_rank"] = range(1, len(models) + 1)
    models = models.sort_values("gic")
    models["gic_rank"] = range(1, len(models) + 1)
    df = pd.concat([df[df["model"] == "Real"], models], ignore_index=True)
    for c in ("kl_rank", "gic_rank"):
        df[c] = df[c].astype("Int64")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "model", "kl_rank", "gic_rank", "kl", "gic", "gic_fit",
            "gic_penalty", "n_params", "edges", "density", "clustering",
            "assortativity", "param"]
    df[cols].to_csv(OUT_DIR / f"{name}_table.csv", index=False)
    _plot_bar(df, name, OUT_DIR / f"{name}_gic_bar.png")
    log(df[cols].to_string(index=False))
    log(f"  ({time.perf_counter()-t0:.1f}s)")
    return df[cols]


# --------------------------------------------------------------------------- plot
CB = {"TLG": "#0072B2", "ER": "#E69F00", "BA": "#009E73", "WS": "#CC79A7",
      "KR": "#D55E00", "GRG": "#56B4E9", "SBM": "#000000"}


def _plot_bar(df, name, out_path):
    """GIC by family (ranked; stacked 2*KL fit + 2*n_params penalty terms)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = df[df["model"] != "Real"].sort_values("gic")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = range(len(m))
    ax.bar(x, m["gic_fit"], color=[CB.get(mm, "#888") for mm in m["model"]],
           label="2·KL (fit)")
    ax.bar(x, m["gic_penalty"], bottom=m["gic_fit"], color="#cccccc",
           label="2·n_params (penalty)", edgecolor="white")
    ax.set_xticks(list(x)); ax.set_xticklabels(m["model"])
    ax.set_ylabel("GIC  (= 2·KL + 2·n_params)")
    ax.set_title(f"{name}: GIC by family (lower = better; stacked terms)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
    for i, (_, r) in enumerate(m.iterrows()):
        ax.text(i, r["gic"], f"{r['gic']:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- runners
def run_one(name, loader):
    """Run the latent-TLG GIC comparison for a single dataset (the per-dataset entry
    point). Writes <name>_table.csv + <name>_gic_bar.png under OUT_DIR."""
    log(f"latent-TLG GIC [{name}] (unified D+community+latent, fair ensemble-mean KL)  "
        f"quick={QUICK} NRUNS={NRUNS} cap={CAP} K={TLG_K} search={SEARCH} "
        f"kernels={KERNELS} klist={KLIST} fast_spectral={FAST_SPECTRAL} seed={SEED}")
    return process_dataset(name, loader)


def run_all(names=None):
    """Run every selected dataset and write a combined cross-dataset KL-rank summary."""
    names = names or (os.environ.get("LG_TLM_DATASETS").split(",")
                      if os.environ.get("LG_TLM_DATASETS") else list(DATASETS))
    log(f"latent-TLG GIC [ALL] datasets={names} quick={QUICK} NRUNS={NRUNS} cap={CAP} "
        f"K={TLG_K} search={SEARCH} kernels={KERNELS} klist={KLIST} seed={SEED}")
    tables = []
    for name in names:
        t = process_dataset(name, DATASETS[name])
        if t is not None:
            tables.append(t)
    if not tables:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(OUT_DIR / "all_tables.csv", index=False)
    m = combined[combined["model"] != "Real"]
    summ = m.pivot_table(index="model", columns="dataset", values="kl_rank",
                         aggfunc="first")
    summ["mean_kl_rank"] = summ.mean(axis=1)
    summ = summ.sort_values("mean_kl_rank")
    summ.to_csv(OUT_DIR / "summary_kl_rank.csv")
    log("\n=== KL rank by dataset (lower = better) ===")
    log(summ.to_string())
    # TLG vs SBM head-to-head on raw KL
    wins = []
    for ds in m["dataset"].unique():
        sub = m[m["dataset"] == ds]
        t = sub[sub["model"] == "TLG"]["kl"]
        s = sub[sub["model"] == "SBM"]["kl"]
        if len(t) and len(s):
            wins.append((ds, float(t.iloc[0]), float(s.iloc[0])))
    log("\n=== TLG vs SBM (raw KL) ===")
    nwin = 0
    for ds, t, s in wins:
        w = "TLG" if t < s else "SBM"
        nwin += t < s
        log(f"  {ds:12} TLG={t:.4f} SBM={s:.4f} -> {w}")
    log(f"  TLG beats SBM on {nwin}/{len(wins)} datasets (raw KL)")
    log(f"\nWrote {OUT_DIR}/")


# ===========================================================================
# Per-dataset SWEEP: fit EVERY network of a dataset, cached/resumable/parallel.
# Each per-dataset entry script (run_tlg_<dataset>_latent_gic.py) calls run_sweep(name);
# run_tlg_all_latent_gic.py invokes those individual scripts in turn.
# ===========================================================================

SWEEP_NMIN = _int("LG_SWEEP_NMIN", 50)        # per-ego BAND filter (twitter/gplus ONLY)
SWEEP_NMAX = _int("LG_SWEEP_NMAX", 1000)
SWEEP_FLOOR = _int("LG_SWEEP_FLOOR", 10)      # min-n fit-sanity floor for the "all" datasets
SWEEP_WORKERS = _int("LG_SWEEP_WORKERS", 8)
HUMAN_SCALE = os.environ.get("LG_SWEEP_HUMAN_SCALE", "1")
ARXIV_CAP = _int("LG_SWEEP_ARXIV_CAP", 1500)
SWEEP_DATASETS = ["twitch", "facebook", "connectome", "arxiv", "gplus", "twitter", "human"]
OVERALL_OUT = _here / "runs" / "tlg_latent_overall_gic"


def _out_dir(dataset):
    """Per-dataset results dir runs/tlg_latent_<dataset>_gic/ (network cache under cache/)."""
    return _here / "runs" / f"tlg_latent_{dataset}_gic"


def _sweep_enumerate(dataset):
    """List of (network_id, kind, arg, nmin, nmax) tasks for a dataset."""
    g = lambda *p: sorted(glob.glob(str(DATA.joinpath(*p))))
    if dataset == "twitch":
        return [(Path(f).stem, "edges", f, SWEEP_FLOOR, 10**9)
                for f in g("twitch", "graphs_processed", "*.edges")]
    if dataset == "twitter":
        return [(Path(f).stem, "edges", f, SWEEP_NMIN, SWEEP_NMAX)
                for f in g("misc", "twitter", "*.edges")]
    if dataset == "facebook":
        return [(Path(f).stem, "edges", f, SWEEP_FLOOR, 10**9)
                for f in g("misc", "facebook", "*.edges")]
    if dataset == "gplus":
        return [(Path(f).stem, "edges", f, SWEEP_NMIN, SWEEP_NMAX)
                for f in g("misc", "gplus", "*.edges")]
    if dataset == "connectome":
        return [(Path(f).stem, "graphml", f, SWEEP_FLOOR, 10**9)
                for f in g("connectomes", "*.graphml")]
    if dataset == "human":
        return [(Path(f).stem, "graphml", f, SWEEP_FLOOR, 10**9)
                for f in g("brain_graph", f"oasis3_graphmls_scale{HUMAN_SCALE}", "*.graphml")]
    if dataset == "arxiv":
        return [("cit-HepTh", "citation", str(ARXIV_CAP), SWEEP_FLOOR, 10**9)]
    raise ValueError(f"unknown dataset {dataset!r}")


def _sweep_load(kind, arg):
    if kind == "edges":
        return _edges(arg)
    if kind == "graphml":
        return _graphml(arg)
    if kind == "citation":
        global CAP
        old, CAP = CAP, int(arg)
        try:
            return _citation()
        finally:
            CAP = old
    raise ValueError(kind)


def _sweep_worker(task):
    """Fit one network (cache-aware). Returns (status, dataset, nid, info)."""
    dataset, nid, kind, arg, nmin, nmax = task
    cdir = _out_dir(dataset) / "cache"
    done = cdir / f"{nid}.json"
    skip = cdir / f"{nid}.skip"
    if done.exists():
        try:
            return ("cached", dataset, nid, json.loads(done.read_text()))
        except Exception:
            pass
    if skip.exists():
        return ("skipped", dataset, nid, None)
    try:
        G = _sweep_load(kind, arg)
    except Exception as ex:
        return ("error", dataset, nid, f"load: {ex}")
    n = G.number_of_nodes()
    if not (nmin < n < nmax):
        cdir.mkdir(parents=True, exist_ok=True)
        skip.write_text(json.dumps({"n": int(n), "reason": "out_of_range"}))
        return ("skipped", dataset, nid, int(n))
    try:
        with contextlib.redirect_stdout(open(os.devnull, "w")):  # silence inner fit log
            scorer = GraphInformationCriterion(G, model="LG", dist="KL")
            real_den, _ = scorer.compute_spectral_density(G)
            fams = []
            tlg = fit_tlg(G, scorer, real_den)
            fams.append(dict(model="TLG", kl=float(tlg["kl"]), gic=float(tlg["gic"]),
                             n_params=int(tlg["n_params"]), param=tlg["param"],
                             **_gmetrics(tlg["graph"])))
            cf = tw.closed_form_params(G)
            gens = tw._baseline_generators(G, cf)
            for fam in ("ER", "BA", "WS", "KR", "GRG", "SBM"):
                gen_fn, npar = gens[fam]
                if fam == "SBM":
                    _, npar = generate_sbm_from_real(G, seed=SEED)
                res = baseline_gic(G, gen_fn, npar, real_den, scorer)
                if res is not None:
                    fams.append(dict(model=fam, kl=float(res["kl"]), gic=float(res["gic"]),
                                     n_params=int(res["n_params"]),
                                     **_gmetrics(res.get("graph"))))
        result = dict(dataset=dataset, id=nid, real=_metrics(G), families=fams)
        cdir.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(result))
        return ("done", dataset, nid, result)
    except Exception as ex:
        return ("error", dataset, nid, f"fit: {ex}")


def _ranked_kl_str(result):
    """Full KL ranking of all families for one network: 'TLG#1 0.246 | SBM#2 0.598 | ...'."""
    fams = sorted(result.get("families", []), key=lambda f: f["kl"])
    return " | ".join(f"{f['model']}#{i} {f['kl']:.3f}" for i, f in enumerate(fams, 1))


def _aggregate_dataset(dataset):
    """Per-dataset: recompute per-graph family ranks + per-family win rates from the cache;
    write per_graph.csv + summary.csv under runs/tlg_latent_<dataset>_gic/. Returns the
    per-family summary DataFrame (or None)."""
    odir = _out_dir(dataset)
    jsons = sorted(glob.glob(str(odir / "cache" / "*.json")))
    rows = []
    for jp in jsons:
        try:
            d = json.loads(Path(jp).read_text())
        except Exception:
            continue
        fam = pd.DataFrame(d["families"])
        fam["kl_rank"] = fam["kl"].rank(method="min").astype(int)
        real = d.get("real", {})
        for _, fr in fam.iterrows():
            rows.append(dict(dataset=dataset, file=d["id"], model=fr["model"],
                             kl=fr["kl"], gic=fr["gic"], n_params=fr["n_params"],
                             kl_rank=fr["kl_rank"],
                             real_nodes=real.get("nodes"), real_edges=real.get("edges"),
                             real_clustering=real.get("clustering"),
                             real_assortativity=real.get("assortativity"),
                             gen_nodes=fr.get("nodes"), gen_edges=fr.get("edges"),
                             gen_avg_degree=fr.get("avg_degree"),
                             gen_clustering=fr.get("clustering"),
                             gen_assortativity=fr.get("assortativity")))
    if not rows:
        return None
    per = pd.DataFrame(rows)
    odir.mkdir(parents=True, exist_ok=True)
    per.to_csv(odir / "per_graph.csv", index=False)
    piv = per.pivot_table(index="file", columns="model", values="kl", aggfunc="first")
    tvs = (float((piv["TLG"] < piv["SBM"]).mean())
           if {"TLG", "SBM"} <= set(piv.columns) else float("nan"))
    agg = []
    for fam in FAMILIES:
        sub = per[per["model"] == fam]
        if not len(sub):
            continue
        agg.append(dict(dataset=dataset, model=fam, n_graphs=int(sub["file"].nunique()),
                        win_rate=float((sub["kl_rank"] == 1).mean()),
                        mean_kl_rank=float(sub["kl_rank"].mean()),
                        median_kl=float(sub["kl"].median()),
                        tlg_beats_sbm=(tvs if fam == "TLG" else float("nan"))))
    summ = pd.DataFrame(agg).sort_values("mean_kl_rank")
    summ.to_csv(odir / "summary.csv", index=False)
    return summ


def _report_overall(datasets):
    """Per-dataset family KL rankings + an OVERALL ranking macro-averaged across datasets
    (each dataset weighted equally, so human's 975 graphs don't drown twitch's 6)."""
    summaries = []
    for ds in datasets:
        summ = _aggregate_dataset(ds)
        if summ is None:
            continue
        summaries.append(summ)
        ng = int(summ["n_graphs"].max())
        tvs = float(summ[summ["model"] == "TLG"]["tlg_beats_sbm"].iloc[0])
        log(f"\n=== [{ds}] {ng} graphs — family KL ranking (mean rank; win_rate = % ranked #1) ===")
        log(summ.drop(columns=["tlg_beats_sbm"]).to_string(index=False))
        log(f"  TLG beats SBM on raw KL: {tvs*100:.1f}% of {ds} graphs")
    if not summaries:
        return
    alls = pd.concat(summaries, ignore_index=True)
    macro = (alls.groupby("model")
             .agg(datasets=("dataset", "nunique"),
                  mean_win_rate=("win_rate", "mean"),
                  mean_kl_rank=("mean_kl_rank", "mean"))
             .sort_values("mean_kl_rank").reset_index())
    OVERALL_OUT.mkdir(parents=True, exist_ok=True)
    alls.to_csv(OVERALL_OUT / "per_dataset_summary.csv", index=False)
    macro.to_csv(OVERALL_OUT / "overall_kl_ranking.csv", index=False)
    log(f"\n{'='*70}\n=== OVERALL family KL ranking across {len(summaries)} datasets "
        f"(macro-averaged; lower mean rank = better) ===")
    log(macro.to_string(index=False))
    log(f"\nWrote {OVERALL_OUT}/")


def _run_pool(tasks, workers, label):
    """Drive a single spawn Pool over the given (dataset, ...) tasks, streaming the full
    family KL ranking per network. This is the shared engine for both the single-dataset
    sweep and the all-datasets run (one global pool -> parallel across datasets AND across
    networks, with load balancing)."""
    n_total = len(tasks)
    log(f"latent-TLG SWEEP [{label}]: {n_total} networks | {workers} workers | "
        f"quick={QUICK} NRUNS={NRUNS} kernels={KERNELS} klist={KLIST} seed={SEED} "
        f"(reproducible: fixed seeds, results cached per network)")
    counts = dict(done=0, cached=0, skipped=0, error=0)
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for i, (status, ds, nid, info) in enumerate(
                pool.imap_unordered(_sweep_worker, tasks, chunksize=1), 1):
            counts[status if status in counts else "error"] += 1
            if status in ("done", "cached") and isinstance(info, dict):
                rn = (info.get("real") or {}).get("nodes", "?")
                log(f"  [{i}/{n_total}] {status} {ds}/{nid} (n={rn}): {_ranked_kl_str(info)}")
            elif status == "skipped":
                log(f"  [{i}/{n_total}] skip {ds}/{nid} (n={info})")
            elif status == "error":
                log(f"  [{i}/{n_total}] ERROR {ds}/{nid}: {info}")
            if i % 20 == 0 or i == n_total:
                el = time.perf_counter() - t0
                eta = (n_total - i) / (i / el) if el > 0 else 0
                log(f"  --- {i}/{n_total} | done {counts['done']} cached {counts['cached']} "
                    f"skip {counts['skipped']} err {counts['error']} | "
                    f"{el/60:.1f}m elapsed, ETA {eta/60:.1f}m ---")
    log(f"\n[{label}] processed {n_total} in {(time.perf_counter()-t0)/60:.1f}m: {counts}")


def run_sweep(dataset, workers=None):
    """Cached/resumable parallel sweep over EVERY network of ONE dataset (networks fit in
    parallel). Writes runs/tlg_latent_<dataset>_gic/{cache,per_graph.csv,summary.csv}."""
    tasks = [(dataset, *t) for t in _sweep_enumerate(dataset)]
    _run_pool(tasks, workers or SWEEP_WORKERS, dataset)
    _report_overall([dataset])


def run_sweep_multi(datasets, workers=None):
    """Sweep MANY datasets in ONE global pool -> parallel across datasets AND across
    networks within each dataset (better load balancing than per-dataset pools). Streams
    per-network rankings, writes each dataset under its own runs/tlg_latent_<dataset>_gic/,
    and prints the overall cross-dataset KL ranking."""
    tasks = [(ds, *t) for ds in datasets for t in _sweep_enumerate(ds)]
    _run_pool(tasks, workers or SWEEP_WORKERS, "+".join(datasets))
    _report_overall(datasets)
