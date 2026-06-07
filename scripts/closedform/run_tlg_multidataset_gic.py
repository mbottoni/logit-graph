#!/usr/bin/env python3
"""Fit a representative network from each dataset with the unified Temporal Logit-Graph
(TLG) and rank families by the spectral GIC / raw KL.

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

import glob
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

OUT_DIR = _here / "runs" / "tlg_multidataset_gic"
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
    try:
        assort = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        assort = float("nan")
    return dict(edges=G.number_of_edges(), density=float(nx.density(G)),
                clustering=float(nx.average_clustering(G)), assortativity=assort)


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
def process_dataset(name):
    t0 = time.perf_counter()
    try:
        G = DATASETS[name]()
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
    log(df[cols].to_string(index=False))
    log(f"  ({time.perf_counter()-t0:.1f}s)")
    return df[cols]


def main():
    sel = os.environ.get("LG_TLM_DATASETS")
    names = sel.split(",") if sel else list(DATASETS)
    log(f"TLG multi-dataset GIC (unified D+community+latent, fair ensemble-mean KL)  "
        f"quick={QUICK} datasets={names} NRUNS={NRUNS} cap={CAP} K={TLG_K} "
        f"search={SEARCH} klist={KLIST} seed={SEED}")
    tables = []
    for name in names:
        t = process_dataset(name)
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


if __name__ == "__main__":
    main()
