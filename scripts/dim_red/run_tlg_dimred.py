#!/usr/bin/env python3
"""Family-fingerprint dimensionality-reduction experiment (LG_DR_DATASET: twitch, connectome, ...):
for each network, generate graphs per family (ER/BA/WS/KR/GRG/SBM/TLG, same closed-form fits),
compute non-colinear structural features, and embed (PCA/t-SNE/UMAP) into per-family density regions."""
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
for p in (_repo_root / "src", _closedform, _repo_root / "scripts" / "more_baselines", _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import tlg_latent_gic_common as C  # noqa: E402  (loaders, latent-TLG sampler pieces, baselines)
import more_baselines_common as MB  # noqa: E402  (modern baseline fitters)
from logit_graph.sbm import generate_sbm_from_real  # noqa: E402

# Modern baselines: name -> fitter(G, A, deg) -> (gen_fn(seed), n_params). Included by default so
# the feature-space comparison spans them too; drop from LG_DR_FAMILIES to exclude.
_MODERN = {"ChungLu": MB.fit_chunglu, "Config": MB.fit_config, "RDPG": MB.fit_rdpg,
           "DCSBM": MB.fit_dcsbm, "HolmeKim": MB.fit_holmekim, "Hyperbolic": MB.fit_hyper}

def _int(env, d):
    v = os.environ.get(env); return int(v) if v else d


def _float(env, d):
    v = os.environ.get(env); return float(v) if v else d


def _net_label(nid):
    """Short display label for a network id (strip the '_graph' suffix Twitch files carry)."""
    return nid[:-6] if nid.endswith("_graph") else nid


# Dataset selection (any dataset tlg_latent_gic_common knows: twitch, connectome, ...). The
# networks, loaders and the cached latent-TLG fits all come from C's dataset machinery.
DATASET = os.environ.get("LG_DR_DATASET", "twitch")
_NETS = C._sweep_enumerate(DATASET)              # (id, kind, arg, nmin, nmax) per network
_sel = os.environ.get("LG_DR_NETWORKS")
if _sel:
    _want = set(_sel.split(","))
    _NETS = [t for t in _NETS if t[0] in _want or _net_label(t[0]) in _want]
NET_ORDER = [_net_label(t[0]) for t in _NETS]    # display labels, in order
OUT = _here / "runs" / f"tlg_dimred_{DATASET}"
CACHE = OUT / "cache"

NREPS = _int("LG_DR_NREPS", 10)
FAMILIES = (os.environ.get("LG_DR_FAMILIES")
            or "ER,BA,WS,KR,GRG,SBM,ChungLu,Config,RDPG,DCSBM,HolmeKim,Hyperbolic,TLG").split(",")
THRESH = _float("LG_DR_COLINEAR_THRESH", 0.85)   # pairwise |Pearson| pre-prune cutoff
VIF_THRESH = _float("LG_DR_VIF_THRESH", 5.0)      # max variance-inflation factor kept
SEED = _int("LG_DR_SEED", 12345)
PERPLEXITY = _float("LG_DR_TSNE_PERPLEXITY", 30.0)
MAIN_METHOD = os.environ.get("LG_DR_MAIN_METHOD", "t-SNE")  # method for the main figure
FOCUS_REGION = os.environ.get("LG_DR_FOCUS_REGION", "")     # network shown in the distance fig
DIST_METHOD = os.environ.get("LG_DR_DIST_METHOD", "UMAP")   # the single embedding shown there
FEATURE_MODE = os.environ.get("LG_DR_FEATURE_MODE", "balanced")  # "balanced" | "vif"

# Structural families -> candidate members (preference order). "balanced" selection forces ONE
# representative per family so every structural axis is represented (notably the community/clustering
# axes a purely statistical VIF prune can silently delete), fair to all models.
_FEATURE_FAMILIES = {
    "degree_level":        ["avg_degree", "density"],
    "degree_heterogeneity": ["degree_cv", "degree_gini", "degree_skew", "degree_kurt",
                             "leaf_frac"],
    "mixing":              ["assortativity", "avg_neighbor_deg"],
    "clustering":          ["clustering", "transitivity", "triangle_density"],
    "community":           ["modularity", "n_comm_norm"],
    "cohesion":            ["max_core", "mean_core", "frac_in_lcc"],
    "spectral":            ["algebraic_conn", "spectral_gap", "spectral_radius"],
}


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
def _tlg_sampler(G, net_id):
    """Reconstruct the cached latent-TLG fit for this network and return (sample_fn, params).
    sample_fn(seed) -> nx.Graph by growing to E_real with the fitted (d,kernel,k,coeffs).
    Falls back to fitting fresh if the sweep cache is absent."""
    n = G.number_of_nodes(); adj = nx.to_numpy_array(G); e = G.number_of_edges()
    rows, cols = np.triu_indices(n, 1)
    cache = C._out_dir(DATASET) / "cache" / f"{net_id}.json"
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


def _family_setup(G, net_id):
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
    # Modern baselines (fitted once per network; reuse the same generators as the KL comparison).
    if any(f in _MODERN for f in FAMILIES):
        A = nx.to_numpy_array(G)
        deg = A.sum(1)
        for name, fitter in _MODERN.items():
            if name not in FAMILIES:
                continue
            try:
                gen_fn, npar = fitter(G, A, deg)
            except Exception as ex:
                log(f"  {net_id}/{name}: fit failed ({ex}) — skipping")
                continue
            setup[name] = ((lambda s, gf=gen_fn: gf(s)), {"n_params": int(npar)})
    tlg_fn, tlg_params = _tlg_sampler(G, net_id)
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
    for nid, kind, arg, nmin, nmax in _NETS:
        label = _net_label(nid)
        t0 = time.perf_counter()
        try:
            G = C._sweep_load(kind, arg)
        except Exception as ex:
            log(f"  {label}: load failed ({ex}) — skipping"); continue
        if not (nmin < G.number_of_nodes() < nmax):
            log(f"  {label}: n={G.number_of_nodes()} out of range — skipping"); continue
        log(f"\n=== {label}: n={G.number_of_nodes()} E={G.number_of_edges()} ===")
        records.append(_cell(label, "real", 0, G, {}))   # real graph
        setup = _family_setup(G, nid)
        for family, (gen_fn, est) in setup.items():
            for rep in range(NREPS):
                records.append(_cell(label, family, rep, gen_fn, est))
            log(f"  {family:4} x{NREPS} done")
        log(f"  ({time.perf_counter()-t0:.1f}s)")
    return records


# --------------------------------------------------------------------------- dim-red
def _vif(Z):
    """Variance inflation factor per column of a standardized DataFrame Z: VIF_j = 1/(1-R_j^2),
    R_j^2 from regressing column j on all the others. Catches multicolinearity (a feature being a
    linear combination of several others), not just pairwise correlation."""
    from sklearn.linear_model import LinearRegression
    cols = list(Z.columns)
    out = {}
    for c in cols:
        others = [o for o in cols if o != c]
        if not others:
            out[c] = 1.0; continue
        r2 = LinearRegression().fit(Z[others].values, Z[c].values).score(
            Z[others].values, Z[c].values)
        out[c] = np.inf if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
    return out


def _prune_colinear(F, thresh):
    """Remove colinearity in two passes: (1) drop one of any pair with |Pearson| >= thresh
    (keep the higher-variance one); (2) iteratively drop the highest-VIF feature until every
    remaining feature has VIF < VIF_THRESH. Returns the non-colinear feature subset."""
    from sklearn.preprocessing import StandardScaler
    order = F.var().sort_values(ascending=False).index.tolist()
    corr = F.corr().abs()
    keep = []
    for c in order:
        if all(corr.loc[c, k] < thresh for k in keep):
            keep.append(c)
    Z = pd.DataFrame(StandardScaler().fit_transform(F[keep].values), columns=keep)
    cur = list(keep)
    while len(cur) > 2:
        vifs = _vif(Z[cur])
        worst = max(vifs, key=vifs.get)
        if vifs[worst] < VIF_THRESH:
            break
        cur.remove(worst)
    return cur


def _select_balanced(F):
    """One representative per structural family (first candidate present), so every axis --
    including community/modularity and clustering -- is represented. Returns (keep, mapping)."""
    keep, mapping = [], {}
    for fam, members in _FEATURE_FAMILIES.items():
        for m in members:
            if m in F.columns:
                keep.append(m); mapping[fam] = m; break
    return keep, mapping


# Explicit colorblind-safe (Okabe-Ito) color + marker per family. The two models of
# interest (TLG, SBM) get strong, high-contrast colors; baselines are muted.
_FAM_STYLE = {
    "ER":  ("#999999", "o"),   # gray
    "BA":  ("#E69F00", "s"),   # orange
    "WS":  ("#009E73", "^"),   # green
    "KR":  ("#56B4E9", "D"),   # sky blue
    "GRG": ("#CC79A7", "v"),   # pink
    "SBM": ("#0072B2", "P"),   # strong blue
    "TLG": ("#D55E00", "X"),   # strong vermillion (the model of interest)
    # modern baselines
    "ChungLu":    ("#7f7f7f", "o"),
    "Config":     ("#bcbd22", "s"),
    "RDPG":       ("#17becf", "^"),
    "DCSBM":      ("#1f77b4", "D"),
    "HolmeKim":   ("#8c564b", "v"),
    "Hyperbolic": ("#9467bd", "*"),
}


_DISPLAY_NAME = {"TLG": "LG"}
_DATASET_DISPLAY = {"twitch": "Twitch", "connectome": "Connectome"}


def _disp(f):
    """Display label for a family (the model is called LG in the paper, TLG in the code)."""
    return _DISPLAY_NAME.get(f, f)


def _real_label():
    """Legend label for the observed graphs, named after the active dataset."""
    return f"real ({_DATASET_DISPLAY.get(DATASET, DATASET)})"


def _fam_color(f):
    return _FAM_STYLE.get(f, ("#777777", "o"))[0]


def _fam_marker(f):
    return _FAM_STYLE.get(f, ("#777777", "o"))[1]


def _cov_ellipse(ax, x, y, color, nstd=2.0):
    """2-sigma covariance ellipse = a family's density region (translucent fill + solid
    edge). This is the family's representation in the figure (the points are not drawn)."""
    from matplotlib.patches import Ellipse
    from matplotlib.colors import to_rgba
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    o = vals.argsort()[::-1]
    vals, vecs = vals[o], vecs[:, o]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * nstd * np.sqrt(np.maximum(vals, 1e-12))
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang,
                         facecolor=to_rgba(color, 0.22), edgecolor=color, lw=1.8, zorder=1))


def _legend_handles(fams):
    """Legend: a filled patch (density region) per family + a star for the real graph."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.colors import to_rgba
    h = [Patch(facecolor=to_rgba(_fam_color(f), 0.30), edgecolor=_fam_color(f), lw=1.6,
               label=_disp(f)) for f in fams]
    h.append(Line2D([], [], marker="*", color="black", markersize=15, ls="none",
                    markeredgecolor="white", label=_real_label()))
    return h


def _combined_figure(meta, F, keep):
    """Pooled single-embedding view (companion to the per-network grid): all networks' graphs in
    ONE PCA/t-SNE/UMAP each (1x3 panels), per-family covariance ellipses + scatter, real graphs as
    black stars. Shows the dataset's global family geometry. 300-dpi PNG + vector PDF."""
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
    fams = [f for f in FAMILIES if f in set(meta["family"])]
    color = {f: _fam_color(f) for f in fams}

    X = StandardScaler().fit_transform(F[keep].values)        # pool ALL graphs into one space
    per = min(PERPLEXITY, max(5.0, (len(X) - 1) / 3))
    pca = PCA(2, random_state=SEED); pca_xy = pca.fit_transform(X)
    emb = {"PCA": pca_xy,
           "t-SNE": TSNE(2, random_state=SEED, init="pca", perplexity=per).fit_transform(X)}
    if has_umap:
        emb["UMAP"] = umap.UMAP(n_components=2, random_state=SEED,
                                n_neighbors=min(15, len(X) - 1),
                                min_dist=0.1).fit_transform(X)
    axis_lab = {"PCA": (f"PC 1 ({pca.explained_variance_ratio_[0] * 100:.0f}% var.)",
                        f"PC 2 ({pca.explained_variance_ratio_[1] * 100:.0f}% var.)"),
                "t-SNE": ("t-SNE 1", "t-SNE 2"), "UMAP": ("UMAP 1", "UMAP 2")}
    fam_arr = meta["family"].values

    fig, axes = plt.subplots(1, len(methods), figsize=(5.6 * len(methods), 5.2), squeeze=False)
    for ci, method in enumerate(methods):
        ax = axes[0][ci]; e = emb[method]
        for f in fams:                                # families: density region + the points
            m = (fam_arr == f)
            _cov_ellipse(ax, e[m, 0], e[m, 1], color[f])
            ax.scatter(e[m, 0], e[m, 1], s=13, alpha=0.75, color=color[f],
                       marker=_fam_marker(f), edgecolors="white", linewidths=0.2, zorder=3)
        m = (fam_arr == "real")                       # real graphs: highlighted stars
        ax.scatter(e[m, 0], e[m, 1], s=240, marker="*", color="black",
                   edgecolors="white", linewidths=1.2, zorder=6)
        ax.set_title(f"({chr(97 + ci)}) {method}", fontweight="bold", loc="left")
        ax.set_xlabel(axis_lab[method][0]); ax.set_ylabel(axis_lab[method][1])
        ax.tick_params(labelbottom=False, labelleft=False, length=0); ax.grid(alpha=0.18, lw=0.5)
    fig.legend(handles=_legend_handles(fams), loc="lower center", ncol=len(fams) + 1,
               frameon=False, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.1)
    fig.tight_layout(rect=[0, 0.05, 1, 0.99])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"dimred_combined.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _grid_figure(meta, F, keep):
    """Publication figure: rows = networks, columns = PCA/t-SNE/UMAP, each embedding fit ON THAT
    NETWORK'S GRAPHS ONLY (so per-region structure differences don't split a family). Per-family
    covariance ellipses + distinct color/marker, real graph as a star. 300-dpi PNG + vector PDF."""
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
    regions = [r for r in NET_ORDER if r in set(meta["region"])]
    fams = [f for f in FAMILIES if f in set(meta["family"])]
    color = {f: _fam_color(f) for f in fams}

    fig, axes = plt.subplots(len(regions), len(methods),
                             figsize=(4.8 * len(methods), 4.3 * len(regions)), squeeze=False)
    emb_rows = []
    region_embeds = {}
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
        region_embeds[region] = (emb, sub)
        for ci, method in enumerate(methods):
            ax = axes[ri][ci]; e = emb[method]
            for f in fams:                       # families: density region + the points
                m = (sub["family"] == f).values
                _cov_ellipse(ax, e[m, 0], e[m, 1], color[f])
                ax.scatter(e[m, 0], e[m, 1], s=13, alpha=0.75, color=color[f],
                           marker=_fam_marker(f), edgecolors="white", linewidths=0.2, zorder=3)
            m = (sub["family"] == "real").values   # real graph: the highlighted star
            ax.scatter(e[m, 0], e[m, 1], s=300, marker="*", color="black",
                       edgecolors="white", linewidths=1.2, zorder=6)
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
    fig.legend(handles=_legend_handles(fams), loc="lower center", ncol=len(fams) + 1,
               frameon=False, bbox_to_anchor=(0.5, -0.004), handletextpad=0.4,
               columnspacing=1.1)
    fig.suptitle("Per-Twitch-network family density regions in structural-feature space "
                 "(rows = network, cols = embedding; ★ = real graph)", y=1.0, fontsize=15)
    fig.tight_layout(rect=[0, 0.022, 1, 0.99])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"dimred_per_graph.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(emb_rows).to_csv(OUT / "embeddings.csv", index=False)
    return region_embeds


def _main_figure(region_embeds, method):
    """Clean single-method main figure: one panel per Twitch network (2 columns), with
    per-family covariance ellipses + a thin line from the real graph to its nearest family
    centroid. 300-dpi PNG + vector PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    regions = list(region_embeds)
    if not regions or method not in region_embeds[regions[0]][0]:
        return
    fams = [f for f in FAMILIES if f in set(region_embeds[regions[0]][1]["family"])]
    color = {f: _fam_color(f) for f in fams}
    ncol = 2
    nrow = int(np.ceil(len(regions) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.4 * nrow), squeeze=False)
    for idx, region in enumerate(regions):
        ax = axes.flat[idx]
        e, sub = region_embeds[region][0][method], region_embeds[region][1]
        cents = {}
        for f in fams:                          # families: density region + the points
            m = (sub["family"] == f).values
            _cov_ellipse(ax, e[m, 0], e[m, 1], color[f])
            ax.scatter(e[m, 0], e[m, 1], s=16, alpha=0.75, color=color[f],
                       marker=_fam_marker(f), edgecolors="white", linewidths=0.2, zorder=3)
            cents[f] = e[m].mean(0)
        m = (sub["family"] == "real").values
        rp = e[m][0]
        nearest = min(fams, key=lambda f: np.linalg.norm(rp - cents[f]))
        ax.plot([rp[0], cents[nearest][0]], [rp[1], cents[nearest][1]], color="black",
                lw=1.0, ls="--", alpha=0.6, zorder=4)
        ax.scatter(rp[0], rp[1], s=360, marker="*", color="black", edgecolors="white",
                   linewidths=1.3, zorder=6)               # real graph: the only point
        ax.set_title(f"{region}  (nearest: {_disp(nearest)})", fontweight="bold")
        ax.tick_params(labelbottom=False, labelleft=False, length=0); ax.grid(alpha=0.18, lw=0.5)
    for j in range(len(regions), nrow * ncol):
        axes.flat[j].axis("off")
    fig.legend(handles=_legend_handles(fams), loc="lower center", ncol=len(fams) + 1,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Twitch family density regions per network ({method}; "
                 f"dashed line: real → nearest family centroid)", fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"dimred_main.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _distance_figure(meta, F, keep, region_embeds):
    """Quantitative result + visual: LEFT heatmap of the distance (standardized feature space) from
    the real graph to each family centroid (closest boxed); RIGHT the PCA/t-SNE/UMAP embeddings for
    one focus network. Writes distance_to_real.csv + dimred_distance.{png,pdf}; prints the summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    regions = [r for r in NET_ORDER if r in set(meta["region"])]
    fams = [f for f in FAMILIES if f in set(meta["family"])]
    color = {f: _fam_color(f) for f in fams}
    D = np.full((len(regions), len(fams)), np.nan)
    for ri, region in enumerate(regions):
        mask = (meta["region"] == region).values
        Z = StandardScaler().fit_transform(F[keep].values[mask])
        sm = meta[mask].reset_index(drop=True)
        real = Z[(sm["family"] == "real").values]
        if not len(real):
            continue
        for ci, f in enumerate(fams):
            pts = Z[(sm["family"] == f).values]
            if len(pts):
                D[ri, ci] = float(np.linalg.norm(real[0] - pts.mean(0)))
    Ddf = pd.DataFrame(D, index=regions, columns=fams)
    Ddf.to_csv(OUT / "distance_to_real.csv")
    ranks = Ddf.rank(axis=1, method="min")
    mean_rank = ranks.mean(0).sort_values()
    order = mean_rank.index.tolist()
    n_closest = Ddf.idxmin(axis=1).value_counts()

    focus = FOCUS_REGION if FOCUS_REGION in region_embeds else regions[0]
    emb, sub = region_embeds[focus]
    method = DIST_METHOD if DIST_METHOD in emb else list(emb)[-1]

    fig = plt.figure(figsize=(max(16, 1.15 * len(order) + 6), max(5.5, 0.55 * len(regions) + 2.4)))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.1], wspace=0.16)
    # left: distance heatmap (NOT a dim-red result -- raw Euclidean distance in the
    # standardized 7-D feature space, real graph -> family centroid)
    axh = fig.add_subplot(gs[0, 0])
    H = Ddf[order]
    im = axh.imshow(H.values, aspect="auto", cmap="viridis")
    axh.set_xticks(range(len(order)))
    axh.set_xticklabels([_disp(o) for o in order], rotation=40, ha="right", fontsize=9)
    axh.set_yticks(range(len(regions))); axh.set_yticklabels(regions)
    mean_all = np.nanmean(H.values)
    for ri in range(len(regions)):
        best = int(np.nanargmin(H.values[ri]))
        for ci in range(len(order)):
            v = H.values[ri, ci]
            axh.text(ci, ri, f"{v:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if v < mean_all else "black",
                     fontweight="bold" if ci == best else "normal")
        axh.add_patch(plt.Rectangle((best - .5, ri - .5), 1, 1, fill=False,
                                    edgecolor="red", lw=2.2))
    axh.add_patch(plt.Rectangle((-.5, regions.index(focus) - .5), len(order), 1, fill=False,
                                edgecolor="black", lw=1.6, ls=":"))   # focus-row marker
    axh.set_title("Distance from real graph to each family centroid\n"
                  "(standardized feature space; red box = closest; dotted = focus network)")
    fig.colorbar(im, ax=axh, shrink=.85, label="Euclidean distance", pad=0.02)
    # right: ONE dim-red embedding for the focus network
    ax = fig.add_subplot(gs[0, 1])
    e = emb[method]; cents = {}
    for f in fams:
        m = (sub["family"] == f).values
        _cov_ellipse(ax, e[m, 0], e[m, 1], color[f])
        ax.scatter(e[m, 0], e[m, 1], s=24, alpha=0.78, color=color[f], marker=_fam_marker(f),
                   edgecolors="white", linewidths=0.3, zorder=3)
        cents[f] = e[m].mean(0)
    m = (sub["family"] == "real").values; rp = e[m][0]
    nearest = min(fams, key=lambda f: np.linalg.norm(rp - cents[f]))
    ax.plot([rp[0], cents[nearest][0]], [rp[1], cents[nearest][1]], color="black",
            lw=1.0, ls="--", alpha=0.6, zorder=4)
    ax.scatter(rp[0], rp[1], s=320, marker="*", color="black", edgecolors="white",
               linewidths=1.3, zorder=6)
    ax.set_title(f"{focus} — {method}  (nearest: {_disp(nearest)})", fontsize=12, fontweight="bold")
    ax.tick_params(labelbottom=False, labelleft=False, length=0); ax.grid(alpha=.18, lw=.5)
    fig.legend(handles=_legend_handles(fams), loc="lower center", ncol=len(fams) + 1,
               frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Distance-to-real (left) and the {method} embedding for the {focus} "
                 f"network (right; family density regions + points, ★ = real graph)",
                 fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"dimred_distance.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"\ndistance-to-real: closest family by network -> {dict(n_closest)}  (focus={focus})")
    log("  mean distance-rank (1=closest): "
        + ", ".join(f"{f}={mean_rank[f]:.2f}" for f in order))


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

    if FEATURE_MODE == "vif":
        keep = _prune_colinear(F, THRESH); mapping = None
    else:
        keep, mapping = _select_balanced(F)
    (OUT / "kept_features.txt").write_text("\n".join(keep))
    # diagnostics: max pairwise |corr| and max VIF AMONG THE KEPT features
    from sklearn.preprocessing import StandardScaler
    Zk = pd.DataFrame(StandardScaler().fit_transform(F[keep].values), columns=keep)
    kept_corr = Zk.corr().abs()
    max_pair = (kept_corr.where(~np.eye(len(keep), dtype=bool)).max().max())
    max_vif = max(_vif(Zk).values())
    log(f"\nfeatures [{FEATURE_MODE}]: {F.shape[1]} candidates -> {len(keep)} kept; "
        f"max|corr|={max_pair:.2f} max VIF={max_vif:.2f}")
    if mapping:
        log("  " + ", ".join(f"{fam}={feat}" for fam, feat in mapping.items()))
    else:
        log("  " + ", ".join(keep))

    # colinearity heatmaps: candidates (before) and the kept set (after)
    fig, axx = plt.subplots(1, 2, figsize=(17, 8))
    for ax, (C, ttl) in zip(axx, [(F.corr(), f"All {F.shape[1]} candidates"),
                                  (F[keep].corr(), f"{len(keep)} kept (VIF<{VIF_THRESH})")]):
        im = ax.imshow(C.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(C))); ax.set_xticklabels(C.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(C))); ax.set_yticklabels(C.columns, fontsize=8)
        ax.set_title(ttl)
        for i in range(len(C)):
            for j in range(len(C)):
                if i != j and abs(C.values[i, j]) >= 0.5:
                    ax.text(j, i, f"{C.values[i, j]:.1f}", ha="center", va="center",
                            fontsize=6, color="black")
    fig.colorbar(im, ax=axx, shrink=.7, label="Pearson correlation")
    fig.suptitle("Feature colinearity: candidates vs. pruned set", fontsize=14)
    fig.savefig(OUT / "colinearity.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    _combined_figure(meta, F, keep)                # pooled 1x3 PCA/t-SNE/UMAP across networks
    region_embeds = _grid_figure(meta, F, keep)   # supplementary: all 3 methods x networks
    _main_figure(region_embeds, MAIN_METHOD)       # main figure: one method, 2 cols, clean
    _distance_figure(meta, F, keep, region_embeds)  # heatmap + the embeddings for one network
    log(f"\nWrote {OUT}/")


def main():
    log(f"TLG dim-red [{DATASET}]: networks={len(NET_ORDER)} families={FAMILIES} nreps={NREPS} "
        f"colinear_thresh={THRESH} seed={SEED}")
    records = generate_and_feature()
    dimred(records)


if __name__ == "__main__":
    main()
