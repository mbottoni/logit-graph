#!/usr/bin/env python3
"""Family-fingerprint dimensionality-reduction experiment on Twitch.

For each Twitch country graph (largest CC, BFS-capped to ~1500 nodes — the same subgraph
the closed-form / latent-TLG experiments use), we:

  1. Estimate each family's parameters with the SAME closed-form estimation as
     scripts/closedform (ER/BA/WS/KR/GRG via moment matching, SBM via Louvain, TLG via the
     latent degree+community+latent-ASE fit — reused from the cached sweep when available,
     else fit fresh), and GENERATE n=10 graphs per family.
  2. Compute ~20 candidate graph-level structural features per graph (generated + real),
     then prune to a NON-COLINEAR subset (greedy |Pearson|<thresh) of ~10-20.
  3. Cache every cell (estimated params, features, n, edges, seed) so reruns resume.
  4. Embed all graphs in 2D with PCA, t-SNE and UMAP and plot the per-family clusters with
     the real graphs highlighted.

Output under runs/tlg_dimred_twitch/ (gitignored):
  cache/<region>_<family>_<rep>.json   per-graph record (resumable)
  features.csv          all graphs x (raw features + meta)
  kept_features.txt     the non-colinear feature subset used for the embeddings
  colinearity.png       feature correlation heatmap (candidates)
  embeddings.csv        2D coords (region, family, rep, method, x, y) per graph
  dimred_per_graph.png / .pdf   rows = Twitch network, cols = PCA/t-SNE/UMAP (fit per
                                network so families don't split by region), ★ = real graph

Env: LG_DR_REGIONS (all 6), LG_DR_NREPS (10), LG_DR_FAMILIES (ER,BA,WS,KR,GRG,SBM,TLG),
LG_DR_COLINEAR_THRESH (0.9), LG_DR_SEED (12345), LG_DR_TSNE_PERPLEXITY (30).
"""
from __future__ import annotations

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
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.stats import skew, kurtosis

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_closedform = _repo_root / "scripts" / "closedform"
for p in (_repo_root / "src", _closedform, _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import tlg_latent_gic_common as C  # noqa: E402  (loaders, latent-TLG sampler pieces, baselines)
from logit_graph.sbm import generate_sbm_from_real  # noqa: E402

OUT = _here / "runs" / "tlg_dimred_twitch"
CACHE = OUT / "cache"
DATA = _repo_root / "data" / "twitch" / "graphs_processed"


def _int(env, d):
    v = os.environ.get(env); return int(v) if v else d


def _float(env, d):
    v = os.environ.get(env); return float(v) if v else d


REGIONS = (os.environ.get("LG_DR_REGIONS") or "PTBR,RU,ES,ENGB,FR,DE").split(",")
NREPS = _int("LG_DR_NREPS", 10)
FAMILIES = (os.environ.get("LG_DR_FAMILIES") or "ER,BA,WS,KR,GRG,SBM,TLG").split(",")
THRESH = _float("LG_DR_COLINEAR_THRESH", 0.9)
SEED = _int("LG_DR_SEED", 12345)
PERPLEXITY = _float("LG_DR_TSNE_PERPLEXITY", 30.0)


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- features
def _gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _spectral(G):
    """Largest adjacency eigenvalue (normalized by mean degree), Fiedler value (algebraic
    connectivity) and normalized-Laplacian spectral gap, via sparse eigsh (dense fallback)."""
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, dtype=float, format="csr")
    deg = np.asarray(A.sum(1)).ravel()
    kbar = deg.mean() if deg.mean() else 1.0
    out = dict(spectral_radius=np.nan, algebraic_conn=np.nan, spectral_gap=np.nan)
    try:
        out["spectral_radius"] = float(sla.eigsh(A, k=1, which="LA",
                                                 return_eigenvectors=False)[0]) / kbar
    except Exception:
        pass
    try:
        Ln = sp.csgraph.laplacian(A, normed=True)
        if n <= 1200:
            ev = np.linalg.eigvalsh(Ln.toarray()); ev.sort()
        else:
            ev = np.sort(sla.eigsh(Ln, k=4, which="SA", return_eigenvectors=False))
        out["algebraic_conn"] = float(ev[1])
        out["spectral_gap"] = float(ev[2] - ev[1])
    except Exception:
        pass
    return out


def features(G):
    """~20 candidate graph-level structural features (size-normalized where natural)."""
    n = G.number_of_nodes(); E = G.number_of_edges()
    deg = np.array([d for _, d in G.degree], float)
    f = {}
    f["density"] = 2.0 * E / (n * (n - 1)) if n > 1 else 0.0
    f["avg_degree"] = float(deg.mean()) if n else 0.0
    f["clustering"] = nx.average_clustering(G)
    f["transitivity"] = nx.transitivity(G)
    try:
        f["assortativity"] = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        f["assortativity"] = 0.0
    f["degree_cv"] = float(deg.std() / deg.mean()) if deg.mean() else 0.0
    f["degree_skew"] = float(skew(deg)) if n > 2 else 0.0
    f["degree_kurt"] = float(kurtosis(deg)) if n > 3 else 0.0
    f["degree_gini"] = _gini(deg)
    f["leaf_frac"] = float(np.mean(deg == 1))
    part = nx.community.louvain_communities(G, seed=SEED)
    f["modularity"] = nx.community.modularity(G, part)
    f["n_comm_norm"] = len(part) / n
    core = np.array(list(nx.core_number(G).values()), float)
    f["max_core"] = float(core.max()) if len(core) else 0.0
    f["mean_core"] = float(core.mean()) if len(core) else 0.0
    f["avg_neighbor_deg"] = (float(np.mean(list(nx.average_neighbor_degree(G).values())))
                             / deg.mean()) if deg.mean() else 0.0
    tri = sum(nx.triangles(G).values()) / 3.0
    f["triangle_density"] = tri / (n * (n - 1) * (n - 2) / 6.0) if n > 2 else 0.0
    f["frac_in_lcc"] = (len(max(nx.connected_components(G), key=len)) / n) if E else 0.0
    f.update(_spectral(G))
    return f


# --------------------------------------------------------------------------- generators
def _load_region(region):
    """Same BFS-capped largest-CC subgraph the closed-form / latent experiments use."""
    return C._edges(str(DATA / f"{region}_graph.edges"))


def _tlg_sampler(G, region):
    """Reconstruct the cached latent-TLG fit for this region and return (sample_fn, params).
    sample_fn(seed) -> nx.Graph by growing to E_real with the fitted (d,kernel,k,coeffs).
    Falls back to fitting fresh if the sweep cache is absent."""
    n = G.number_of_nodes(); adj = nx.to_numpy_array(G); e = G.number_of_edges()
    rows, cols = np.triu_indices(n, 1)
    cache = C._out_dir("twitch") / "cache" / f"{region}_graph.json"
    if cache.exists():
        d = json.loads(cache.read_text()); sel = d["tlg_selected"]
        b = next(t for t in d["tlg_trace"]
                 if (t["d"], t["kernel"], t["k"]) == (sel["d"], sel["kernel"], sel["k"]))
    else:
        sc = C.GraphInformationCriterion(G, model="LG", dist="KL")
        rd, _ = sc.compute_spectral_density(G)
        fit = C.fit_tlg(G, sc, rd); sel = fit["selected"]
        b = next(t for t in fit["trace"]
                 if (t["d"], t["kernel"], t["k"]) == (sel["d"], sel["kernel"], sel["k"]))
    params = dict(d=sel["d"], kernel=sel["kernel"], k=sel["k"],
                  alpha=b["alpha"], gc=b["gc"], gf=b["gf"], lam=b["lam"])
    Bc, _ = C.tw._community_feature(G, rows, cols, C.SEED, 1.0)
    Bf, _ = C.tw._community_feature(G, rows, cols, C.SEED, C.FINE_RES)
    w, U = np.linalg.eigh(adj)
    L = C._latent_from_eig(w, U, params["k"], rows, cols, params["kernel"])

    def sample(seed):
        A = C._grow_one(n, rows, cols, params["alpha"], params["gc"], params["gf"],
                        params["lam"], Bc, Bf, L, e, seed, params["d"])
        return nx.from_numpy_array(A)
    return sample, params


def _family_setup(G, region):
    """Return {family: (sample_fn(seed)->Graph, est_params_dict)} for all families."""
    cf = C.tw.closed_form_params(G)
    gens = C.tw._baseline_generators(G, cf)
    est = dict(ER={"p": cf["ER"]}, BA={"m": cf["BA"]}, WS={"k": cf["WS_k"], "p": cf["WS_p"]},
               KR={"d": cf["KR"]}, GRG={"r": cf["GRG"]})
    setup = {}
    for fam in ("ER", "BA", "WS", "KR", "GRG"):
        gen_fn = gens[fam][0]
        setup[fam] = ((lambda s, gf=gen_fn: gf(s)), est[fam])
    _, sbm_np = generate_sbm_from_real(G, seed=SEED)
    setup["SBM"] = ((lambda s: generate_sbm_from_real(G, seed=s)[0]), {"n_params": int(sbm_np)})
    tlg_fn, tlg_params = _tlg_sampler(G, region)
    setup["TLG"] = (tlg_fn, tlg_params)
    return {f: setup[f] for f in FAMILIES if f in setup}


# --------------------------------------------------------------------------- driver
def _cell(region, family, rep, gen_fn, est_params):
    cf_path = CACHE / f"{region}_{family}_{rep}.json"
    if cf_path.exists():
        try:
            return json.loads(cf_path.read_text())
        except Exception:
            pass
    seed = SEED + 1009 * rep
    G = gen_fn(seed) if family != "real" else gen_fn
    rec = dict(region=region, family=family, rep=rep, seed=seed,
               n=G.number_of_nodes(), edges=G.number_of_edges(),
               est_params=est_params, features=features(G))
    CACHE.mkdir(parents=True, exist_ok=True)
    cf_path.write_text(json.dumps(rec))
    return rec


def generate_and_feature():
    records = []
    for region in REGIONS:
        t0 = time.perf_counter()
        G = _load_region(region)
        log(f"\n=== {region}: n={G.number_of_nodes()} E={G.number_of_edges()} ===")
        # real graph
        records.append(_cell(region, "real", 0, G, {}))
        setup = _family_setup(G, region)
        for family, (gen_fn, est) in setup.items():
            for rep in range(NREPS):
                records.append(_cell(region, family, rep, gen_fn, est))
            log(f"  {family:4} x{NREPS} done")
        log(f"  ({time.perf_counter()-t0:.1f}s)")
    return records


# --------------------------------------------------------------------------- dim-red
def _prune_colinear(F, thresh):
    """Greedy: keep a feature unless |corr| >= thresh with an already-kept one. Orders by
    descending variance so the most informative features are kept first."""
    order = F.var().sort_values(ascending=False).index.tolist()
    corr = F.corr().abs()
    keep = []
    for c in order:
        if all(corr.loc[c, k] < thresh for k in keep):
            keep.append(c)
    return keep


# Okabe-Ito colorblind-safe palette + distinct markers, one per family.
_OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _cov_ellipse(ax, x, y, color, nstd=2.0):
    """2-sigma covariance ellipse of a family's point cloud (cluster footprint)."""
    from matplotlib.patches import Ellipse
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    o = vals.argsort()[::-1]
    vals, vecs = vals[o], vecs[:, o]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nstd * np.sqrt(np.maximum(vals, 1e-12))
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang, facecolor=color,
                         alpha=0.13, edgecolor=color, lw=1.0, zorder=1))


def _grid_figure(meta, F, keep):
    """Publication figure: rows = Twitch networks, columns = PCA / t-SNE / UMAP, each
    embedding fit ON THAT NETWORK'S GRAPHS ONLY (so the per-region structure differences
    don't split each family into separate sub-clusters). Per-family covariance ellipses +
    distinct color/marker, real graph as a star. 300-dpi PNG + vector PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "legend.fontsize": 11,
                         "font.family": "DejaVu Sans", "axes.linewidth": 0.9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    has_umap = True
    try:
        import umap
    except Exception:
        has_umap = False
    methods = ["PCA", "t-SNE"] + (["UMAP"] if has_umap else [])
    regions = [r for r in REGIONS if r in set(meta["region"])]
    fams = [f for f in FAMILIES if f in set(meta["family"])]
    color = {f: _OKABE[i % len(_OKABE)] for i, f in enumerate(fams)}
    marker = {f: _MARKERS[i % len(_MARKERS)] for i, f in enumerate(fams)}

    fig, axes = plt.subplots(len(regions), len(methods),
                             figsize=(4.8 * len(methods), 4.3 * len(regions)), squeeze=False)
    emb_rows = []
    for ri, region in enumerate(regions):
        mask = (meta["region"] == region).values
        sub = meta[mask].reset_index(drop=True)
        Xr = StandardScaler().fit_transform(F[keep].values[mask])
        per = min(PERPLEXITY, max(5.0, (len(Xr) - 1) / 3))
        emb = {"PCA": PCA(2, random_state=SEED).fit_transform(Xr),
               "t-SNE": TSNE(2, random_state=SEED, init="pca", perplexity=per).fit_transform(Xr)}
        if has_umap:
            emb["UMAP"] = umap.UMAP(n_components=2, random_state=SEED,
                                    n_neighbors=min(15, len(Xr) - 1),
                                    min_dist=0.1).fit_transform(Xr)
        for ci, method in enumerate(methods):
            ax = axes[ri][ci]; e = emb[method]
            for f in fams:
                m = (sub["family"] == f).values
                _cov_ellipse(ax, e[m, 0], e[m, 1], color[f])
                ax.scatter(e[m, 0], e[m, 1], s=26, alpha=0.8, color=color[f],
                           marker=marker[f], edgecolors="white", linewidths=0.3,
                           zorder=3, label=f)
            m = (sub["family"] == "real").values
            ax.scatter(e[m, 0], e[m, 1], s=300, marker="*", color="black",
                       edgecolors="white", linewidths=1.2, zorder=6, label="real (Twitch)")
            ax.tick_params(labelbottom=False, labelleft=False, length=0)
            ax.grid(alpha=0.18, lw=0.5)
            if ri == 0:
                ax.set_title(method, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(region, fontweight="bold", fontsize=13, labelpad=8)
            for j in range(len(e)):
                emb_rows.append(dict(region=region, family=sub["family"][j],
                                     rep=int(sub["rep"][j]), method=method,
                                     x=float(e[j, 0]), y=float(e[j, 1])))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(fams) + 1, frameon=False,
               bbox_to_anchor=(0.5, -0.004), handletextpad=0.3, columnspacing=1.1)
    fig.suptitle("Per-Twitch-network graph-family clusters in structural-feature space "
                 "(rows = network, cols = embedding; ★ = real graph)", y=1.0, fontsize=15)
    fig.tight_layout(rect=[0, 0.022, 1, 0.99])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"dimred_per_graph.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(emb_rows).to_csv(OUT / "embeddings.csv", index=False)


def dimred(records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = pd.DataFrame([{k: r[k] for k in ("region", "family", "rep", "n", "edges")}
                         for r in records])
    F = pd.DataFrame([r["features"] for r in records])
    F = F.dropna(axis=1, how="any")                 # drop features undefined on any graph
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat([meta, F], axis=1).to_csv(OUT / "features.csv", index=False)

    keep = _prune_colinear(F, THRESH)
    (OUT / "kept_features.txt").write_text("\n".join(keep))
    log(f"\nfeatures: {F.shape[1]} candidates -> {len(keep)} non-colinear (|corr|<{THRESH}):")
    log("  " + ", ".join(keep))

    # colinearity heatmap (candidates)
    corr = F.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Candidate feature colinearity (|corr|>%.2f pruned)" % THRESH)
    fig.colorbar(im, shrink=.8); fig.tight_layout()
    fig.savefig(OUT / "colinearity.png", dpi=150); plt.close(fig)

    _grid_figure(meta, F, keep)   # one PCA/t-SNE/UMAP per Twitch network (rows), no mixing
    log(f"\nWrote {OUT}/")


def main():
    log(f"TLG dim-red [twitch]: regions={REGIONS} families={FAMILIES} nreps={NREPS} "
        f"colinear_thresh={THRESH} seed={SEED}")
    records = generate_and_feature()
    dimred(records)


if __name__ == "__main__":
    main()
