#!/usr/bin/env python3
"""Additional modern baseline random-graph models, evaluated with the SAME estimator and metrics
as the main sweep so their numbers are directly comparable to the existing families.

This module fits and scores ONLY the new baselines (it does not re-run LG/TLG/ER/BA/WS/KR/GRG/SBM):

  ChungLu     : expected-degree model (Chung--Lu); reproduces the observed degree sequence.
  Config      : configuration model; matches the observed degree sequence exactly (stub-matching).
  RDPG        : (generalized) random dot-product graph; latent positions from the adjacency
                spectral embedding (the same ASE used for the TLG latent feature), edge prob
                = clip(<x_i, x_j>_signature, 0, 1).
  DCSBM       : degree-corrected stochastic block model; Louvain blocks + node degree corrections
                (custom fit, so graph-tool is not required).
  HolmeKim    : Barabasi--Albert growth with triad formation (tunable clustering).
  Hyperbolic  : threshold hyperbolic (H2) random graph; radial exponent from the degree power law,
                connection radius fitted to the observed edge count.

For every network of a dataset it generates an ensemble (the same EVAL_SEEDS as the main sweep),
computes the adjacency-spectral KL divergence to the observed graph, and the three structural
discrepancy metrics introduced for the main comparison (degree-distribution KS, clustering, and
assortativity differences). Results are cached per network (resumable, config-hashed by
CACHE_VERSION) and aggregated to per_graph.csv + summary.csv, mirroring the main sweep so the data
is easy to gather afterwards.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import entropy

warnings.filterwarnings("ignore")

_here = Path(__file__).resolve().parent
_repo = _here.parents[1]
for p in (_repo / "src", _repo / "scripts" / "closedform", _repo / "scripts" / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the main sweep's dataset loaders, spectral scorer, structural metrics, and the
# degree-KS / clustering / assortativity discrepancy helper, so everything is consistent.
import tlg_latent_gic_common as C  # noqa: E402

SEED = C.SEED
EVAL_SEEDS = C.EVAL_SEEDS
CACHE_VERSION = 1
# Above this node count the O(n^2) probability-matrix models (RDPG, DC-SBM) are skipped; the main
# datasets (connectomes, Twitch, ego components) are capped well below this by the loaders.
MAX_DENSE_N = int(os.environ.get("MB_MAX_DENSE_N", "3000"))
FAMILIES = ["ChungLu", "Config", "RDPG", "DCSBM", "HolmeKim", "Hyperbolic"]
DENSE = {"RDPG", "DCSBM"}  # build dense n x n probability matrices


def log(*a):
    print(*a, flush=True)


def _simple(g):
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def _gamma(deg):
    """Hill/MLE power-law exponent of a degree sequence (fallback 2.5)."""
    d = deg[deg >= 1]
    if len(d) < 10:
        return 2.5
    xmin = max(1, int(np.percentile(d, 10)))
    dd = d[d >= xmin]
    if len(dd) < 10:
        return 2.5
    return float(1.0 + len(dd) / np.sum(np.log(dd / (xmin - 0.5))))


# --------------------------------------------------------------------------- fitters
# Each returns (gen_fn(seed) -> nx.Graph, n_params).

def fit_chunglu(G, A, deg):
    d = [int(x) for x in deg]
    return (lambda s: _simple(nx.expected_degree_graph(d, seed=int(s), selfloops=False)), len(d))


def fit_config(G, A, deg):
    d = [int(x) for x in deg]
    return (lambda s: _simple(nx.configuration_model(d, seed=int(s))), len(d))


def fit_holmekim(G, A, deg):
    n, m = G.number_of_nodes(), G.number_of_edges()
    mm = max(1, round(m / n))                 # edges added per new node ~ <k>/2
    c_obs = nx.average_clustering(G)
    best = (1e9, 0.3)
    for p in np.linspace(0.0, 0.95, 10):      # pick triad prob p to match observed clustering
        c = nx.average_clustering(nx.powerlaw_cluster_graph(n, mm, float(p), seed=SEED))
        best = min(best, (abs(c - c_obs), float(p)))
    p = best[1]
    return (lambda s: nx.powerlaw_cluster_graph(n, mm, p, seed=int(SEED + s)), 2)


def fit_rdpg(G, A, deg):
    n, m = A.shape[0], G.number_of_edges()
    w, U = np.linalg.eigh(A)
    thr = 2.0 * np.sqrt(max(1e-9, 2.0 * m / n))            # noise scale ~ 2*sqrt(<k>)
    d = int(max(4, min(30, np.sum(np.abs(w) > thr))))      # elbow on |eigenvalues|
    idx = np.argsort(np.abs(w))[::-1][:d]
    X = U[:, idx] * np.sqrt(np.abs(w[idx]))
    sig = np.sign(w[idx])                                  # GRDPG signature (indefinite inner product)
    P = np.clip((X * sig) @ X.T, 0.0, 1.0)
    np.fill_diagonal(P, 0.0)

    def gen(s):
        r = np.random.default_rng(1000 + int(s))
        M = np.triu((r.random((n, n)) < P).astype(float), 1)
        return nx.from_numpy_array(M + M.T)
    return (gen, n * d)


def fit_dcsbm(G, A, deg):
    n = A.shape[0]
    comms = nx.community.louvain_communities(G, seed=SEED)
    z = np.zeros(n, int)
    for b, c in enumerate(comms):
        for v in c:
            z[v] = b
    K = len(comms)
    kappa = np.array([deg[z == r].sum() for r in range(K)], float)   # block degree mass
    W = np.zeros((K, K))
    for i, j in G.edges():
        W[z[i], z[j]] += 1.0
        W[z[j], z[i]] += 1.0                                          # degree-endpoints between blocks
    theta = deg / np.maximum(kappa[z], 1.0)                           # node degree propensity
    P = np.clip((theta[:, None] * theta[None, :]) * W[np.ix_(z, z)], 0.0, 1.0)
    np.fill_diagonal(P, 0.0)

    def gen(s):
        r = np.random.default_rng(2000 + int(s))
        M = np.triu((r.random((n, n)) < P).astype(float), 1)
        return nx.from_numpy_array(M + M.T)
    return (gen, K * (K + 1) // 2 + K)


def _hyper_gen(n, alpha, R, seed):
    r = np.random.default_rng(seed)
    u = r.random(n)
    rad = np.arccosh(1.0 + (np.cosh(alpha * R) - 1.0) * u) / alpha    # radial density ~ sinh(alpha r)
    th = r.random(n) * 2.0 * np.pi
    dth = np.pi - np.abs(np.pi - np.abs(th[:, None] - th[None, :]))
    ch = (np.cosh(rad[:, None]) * np.cosh(rad[None, :])
          - np.sinh(rad[:, None]) * np.sinh(rad[None, :]) * np.cos(dth))
    x = np.arccosh(np.clip(ch, 1.0, None))
    np.fill_diagonal(x, np.inf)
    M = np.triu((x <= R), 1).astype(float)                           # threshold (T=0) connection
    return nx.from_numpy_array(M + M.T)


def fit_hyper(G, A, deg):
    n, m = G.number_of_nodes(), G.number_of_edges()
    alpha = max(0.55, (_gamma(deg) - 1.0) / 2.0)                     # gamma = 2*alpha + 1
    lo, hi = 2.0, 30.0
    for _ in range(20):                                             # bisection on R -> edge count
        mid = (lo + hi) / 2.0
        e = _hyper_gen(n, alpha, mid, SEED).number_of_edges()
        lo, hi = (mid, hi) if e > m else (lo, mid)
    R = (lo + hi) / 2.0
    return (lambda s: _hyper_gen(n, alpha, R, int(SEED + s)), 3)


FITTERS = {"ChungLu": fit_chunglu, "Config": fit_config, "RDPG": fit_rdpg,
           "DCSBM": fit_dcsbm, "HolmeKim": fit_holmekim, "Hyperbolic": fit_hyper}


# --------------------------------------------------------------------------- per-network fit
def fit_network(G):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    deg = A.sum(1)
    scorer = C.GraphInformationCriterion(G, model="LG", dist="KL")
    real_den, _ = scorer.compute_spectral_density(G)
    rm = C._metrics(G)
    real_desc = ([int(x) for x in deg], rm["clustering"], rm["assortativity"])
    m_real = int(G.number_of_edges())

    fams = []
    for fam in FAMILIES:
        row = dict(model=fam)
        if fam in DENSE and n > MAX_DENSE_N:
            row.update(kl=None, note="skipped:n>MAX_DENSE_N")
            fams.append(row)
            continue
        try:
            gen, npar = FITTERS[fam](G, A, deg)
        except Exception as ex:
            row.update(kl=None, note=f"fit_error:{str(ex)[:50]}")
            fams.append(row)
            continue
        dens, graphs, edges = [], [], []
        for s in EVAL_SEEDS:
            try:
                g = gen(s)
            except Exception:
                continue
            if g.number_of_edges() == 0:
                continue
            dens.append(scorer.compute_spectral_density(g)[0])
            graphs.append(g)
            edges.append(g.number_of_edges())
        if not dens:
            row.update(kl=None, note="no_graphs")
            fams.append(row)
            continue
        kl = float(entropy(real_den + 1e-10, np.mean(dens, axis=0) + 1e-10))
        disc = C._fit_discrepancy(graphs, *real_desc)
        row.update(kl=kl, n_params=int(npar), mean_edges=float(np.mean(edges)),
                   edge_ratio=float(np.mean(edges) / max(1, m_real)),
                   ks_deg=disc["ks_deg"], d_clustering=disc["d_clustering"],
                   d_assortativity=disc["d_assortativity"])
        fams.append(row)
    return dict(cache_version=CACHE_VERSION, real=rm, real_edges=m_real, families=fams)


# --------------------------------------------------------------------------- driver + cache
def _out_dir(dataset):
    return _here / "runs" / f"more_baselines_{dataset}"


def run_dataset(dataset):
    items = C._sweep_enumerate(dataset)
    odir = _out_dir(dataset)
    cdir = odir / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    log(f"\n=== more-baselines [{dataset}]: {len(items)} networks | seed={SEED} "
        f"| NRUNS={C.NRUNS} | families={FAMILIES} ===")
    t0 = time.perf_counter()
    for i, (nid, kind, arg, nmin, nmax) in enumerate(items, 1):
        done = cdir / f"{nid}.json"
        if done.exists():
            try:
                if json.loads(done.read_text()).get("cache_version") == CACHE_VERSION:
                    log(f"  [{i}/{len(items)}] cached {nid}")
                    continue
            except Exception:
                pass
        try:
            G = C._sweep_load(kind, arg)
        except Exception as ex:
            log(f"  [{i}/{len(items)}] load-fail {nid}: {ex}")
            continue
        n = G.number_of_nodes()
        if not (nmin < n < nmax):
            log(f"  [{i}/{len(items)}] skip {nid} (n={n} out of range)")
            continue
        res = fit_network(G)
        res.update(dataset=dataset, id=nid)
        done.write_text(json.dumps(res))
        summ = "  ".join(f"{f['model']}={f['kl']:.3f}" if f.get("kl") is not None
                         else f"{f['model']}=NA" for f in res["families"])
        log(f"  [{i}/{len(items)}] {nid} (n={n}): {summ}")
    aggregate(dataset)
    log(f"  [{dataset}] done ({time.perf_counter()-t0:.1f}s)")


def aggregate(dataset):
    odir = _out_dir(dataset)
    rows = []
    for jp in sorted(glob.glob(str(odir / "cache" / "*.json"))):
        try:
            d = json.loads(Path(jp).read_text())
        except Exception:
            continue
        real = d.get("real", {})
        for f in d.get("families", []):
            rows.append(dict(
                dataset=dataset, id=d["id"], model=f["model"],
                kl=f.get("kl"), ks_deg=f.get("ks_deg"),
                d_clustering=f.get("d_clustering"), d_assortativity=f.get("d_assortativity"),
                n_params=f.get("n_params"), mean_edges=f.get("mean_edges"),
                real_edges=d.get("real_edges"), edge_ratio=f.get("edge_ratio"),
                real_nodes=real.get("nodes"), real_clustering=real.get("clustering"),
                real_assortativity=real.get("assortativity"), note=f.get("note")))
    if not rows:
        return None
    per = pd.DataFrame(rows)
    odir.mkdir(parents=True, exist_ok=True)
    per.to_csv(odir / "per_graph.csv", index=False)

    ok = per[per["kl"].notna()]
    agg = []
    for fam in FAMILIES:
        sub_all = per[per["model"] == fam]
        sub = ok[ok["model"] == fam]
        agg.append(dict(
            dataset=dataset, model=fam,
            n_nets=int(sub_all["id"].nunique()),
            n_fitted=int(sub["id"].nunique()),
            success_rate=float(len(sub) / max(1, len(sub_all))),
            median_kl=float(sub["kl"].median()) if len(sub) else float("nan"),
            median_ks_deg=float(sub["ks_deg"].median()) if len(sub) else float("nan"),
            median_d_clustering=float(sub["d_clustering"].median()) if len(sub) else float("nan"),
            median_d_assortativity=float(sub["d_assortativity"].median()) if len(sub) else float("nan"),
            median_edge_ratio=float(sub["edge_ratio"].median()) if len(sub) else float("nan")))
    summ = pd.DataFrame(agg).sort_values("median_kl")
    summ.to_csv(odir / "summary.csv", index=False)
    log(f"  wrote {odir/'per_graph.csv'} and {odir/'summary.csv'}")
    return summ
