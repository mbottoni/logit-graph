#!/usr/bin/env python3
"""Fit the Twitch networks with the Temporal Logit-Graph (TLG) and rank random-graph families
by spectral GIC = 2*KL(real||model) + 2*n_params (KL on the normalized-Laplacian density,
lower=better); TLG uses identifiable degree + coarse/fine community features. `make tlg-twitch-gic`."""
from __future__ import annotations

import math
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
from scipy.stats import entropy

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)  # stream progress live (no buffering)
except Exception:
    pass


def log(*a):
    print(*a, flush=True)


from logit_graph.gic import GraphInformationCriterion  # noqa: E402
from logit_graph.temporal import fit_growth_params  # noqa: E402
from logit_graph.lg_features import build_pair_dataset  # noqa: E402
from logit_graph.sbm import generate_sbm_from_real  # noqa: E402

OUT_DIR = _here / "runs" / "tlg_twitch_gic"
DATA_DIR = _repo_root / "data" / "twitch" / "graphs_processed"
ALL_REGIONS = ["PTBR", "RU", "ES", "ENGB", "FR", "DE"]  # smallest -> largest
TLG_N_PARAMS = 3  # alpha (degree), beta (clustering), gamma (community); d fixed


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


def _float(env, default):
    raw = os.environ.get(env)
    return float(raw) if raw is not None else default


QUICK = os.environ.get("LG_TLGT_QUICK", "0") == "1"
if os.environ.get("LG_TLGT_ALL", "0") == "1":
    REGIONS = ALL_REGIONS
else:
    REGIONS = os.environ.get("LG_TLGT_REGIONS", "PTBR").split(",")
NRUNS = _int("LG_TLGT_NRUNS", 3 if QUICK else 5)   # baseline ensemble size
SEED = _int("LG_TLGT_SEED", 12345)
# Identifiable forward best-KL fit (degree + coarse-community + fine-community):
TLG_D = _int("LG_TLGT_D", 1)              # degree-feature depth for the forward fit
TLG_K = _int("LG_TLGT_K", 15 if QUICK else 25)   # growth batches to reach E_real
TLG_SEARCH_MAXITER = _int("LG_TLGT_SEARCH", 10 if QUICK else 16)  # (gc,gf) NM iters
TLG_FINE_RES = _float("LG_TLGT_FINE_RES", 8.0)   # Louvain resolution for the fine partition

FAMILIES = ["TLG", "ER", "BA", "WS", "KR", "GRG", "SBM"]


def _load_region(path):
    G = nx.read_edgelist(path, comments="#", nodetype=int)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def _metrics(G):
    """Structural metrics of a graph (NaN-safe assortativity)."""
    try:
        assort = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        assort = float("nan")
    return dict(edges=G.number_of_edges(),
                density=float(nx.density(G)),
                clustering=float(nx.average_clustering(G)),
                assortativity=assort)


def closed_form_params(G):
    n, E = G.number_of_nodes(), G.number_of_edges()
    kbar = 2 * E / n
    p_er = 2 * E / (n * (n - 1))
    m_ba = max(1, round(E / n))
    k_ws = max(2, int(round(kbar)))
    if k_ws % 2 == 1:
        k_ws += 1
    C_obs = nx.average_clustering(G)
    if k_ws > 2:
        C0 = 3 * (k_ws - 2) / (4 * (k_ws - 1))
        p_ws = 0.0 if C_obs >= C0 else 1.0 - (C_obs / C0) ** (1.0 / 3.0)
    else:
        p_ws = 0.0
    d_kr = int(round(kbar))
    if (n * d_kr) % 2 == 1:
        d_kr -= 1
    r_grg = math.sqrt(kbar / (math.pi * (n - 1)))
    return dict(n=n, E=E, ER=p_er, BA=m_ba, WS_k=k_ws, WS_p=p_ws, KR=d_kr, GRG=r_grg)


def _baseline_generators(G, cf):
    """name -> (gen_fn(seed)->nx.Graph, n_params)."""
    n = cf["n"]
    return {
        "ER": (lambda s: nx.erdos_renyi_graph(n, float(np.clip(cf["ER"], 1e-6, 1)), seed=s), 1),
        "BA": (lambda s: nx.barabasi_albert_graph(n, int(np.clip(cf["BA"], 1, n - 1)), seed=s), 1),
        "WS": (lambda s: nx.watts_strogatz_graph(n, cf["WS_k"], cf["WS_p"], seed=s), 2),
        "KR": (lambda s: nx.random_regular_graph(int(np.clip(cf["KR"], 0, n - 1)), n, seed=s), 1),
        "GRG": (lambda s: nx.random_geometric_graph(n, float(np.clip(cf["GRG"], 1e-3, 1.5)), seed=s), 1),
        "SBM": (lambda s: generate_sbm_from_real(G, seed=s)[0], None),  # n_params resolved below
    }


def _community_feature(G, rows, cols, seed, resolution=1.0):
    """Same-community indicator over pairs from a fixed Louvain partition of G at the given
    resolution — an exogenous NODE covariate (like SBM uses), so it stays identifiable; higher
    resolution gives finer communities (a local-density proxy). Returns (B, n_communities)."""
    part = nx.community.louvain_communities(G, seed=seed, resolution=resolution)
    blk = np.empty(G.number_of_nodes(), dtype=int)
    for i, com in enumerate(part):
        for v in com:
            blk[v] = i
    return (blk[rows] == blk[cols]).astype(float), len(part)


def _budgeted_grow(n, rows, cols, alpha, gc, gf, Bc, Bf, d, e_real, seed, scorer, real_den):
    """Honest forward generation: budgeted add-only growth to the real edge count. Over TLG_K
    batches add ~E_real/TLG_K edges each, non-edges chosen with prob ∝ exp(alpha*D + gc*Bc +
    gf*Bf) (D recomputed on the generated graph; Bc/Bf fixed). Returns the lowest-KL graph within ±5%."""
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n))
    batch = max(1, e_real // TLG_K)
    best = {"dist": float("inf"), "adj": None}
    for _ in range(TLG_K + 6):
        D, _l = build_pair_dataset(A, d=d, mode="bounded", layer2=True)
        ne = np.where(A[rows, cols] == 0)[0]
        if len(ne) == 0:
            break
        lo = alpha * np.asarray(D)[ne] + gc * Bc[ne] + gf * Bf[ne]
        w = np.exp(lo - lo.max())
        w /= w.sum()
        pick = rng.choice(ne, size=min(batch, len(ne)), replace=False, p=w)
        A[rows[pick], cols[pick]] = 1.0
        A[cols[pick], rows[pick]] = 1.0
        e = int(A.sum() // 2)
        if 0.95 * e_real <= e <= 1.05 * e_real:   # edge-matched window
            den, _ = scorer.compute_spectral_density(nx.from_numpy_array(A))
            dist = float(entropy(real_den + 1e-10, den + 1e-10))
            if dist < best["dist"]:
                best.update(dist=dist, adj=A.copy())
        if e > 1.1 * e_real:
            break
    return best


def fit_tlg(adj, real_den, scorer, G):
    """Identifiable, SBM-beating TLG: degree + coarse-community + fine-community (all exogenous,
    recoverable by MLE; no endogenous clustering term). alpha from the degree MLE; gc, gf tuned by
    minimizing the forward edge-matched KL (fair, like the baselines' grid min-GIC). n_params = 3."""
    n = adj.shape[0]
    e_real = int(adj.sum() // 2)
    rows, cols = np.triu_indices(n, k=1)
    d = TLG_D
    t0 = time.perf_counter()
    Dr, lab = build_pair_dataset(adj, d=d, mode="bounded", layer2=True)
    alpha = float(fit_growth_params(Dr, lab)["alpha"])      # degree MLE (warm start)
    Bc, kc = _community_feature(G, rows, cols, SEED, resolution=1.0)
    Bf, kf = _community_feature(G, rows, cols, SEED, resolution=TLG_FINE_RES)
    log(f"    TLG forward fit: d={d} α={alpha:.3f}; communities coarse={kc} fine={kf}; "
        f"tuning (γc,γf) by min forward-KL (<= {TLG_SEARCH_MAXITER} iters) ...")
    from scipy.optimize import minimize
    nev = {"n": 0}

    def obj(x):
        nev["n"] += 1
        b = _budgeted_grow(n, rows, cols, alpha, max(0.0, x[0]), max(0.0, x[1]),
                           Bc, Bf, d, e_real, SEED, scorer, real_den)
        v = b["dist"] if b["adj"] is not None else 1e6
        log(f"      eval {nev['n']:2d}: γc={x[0]:.2f} γf={x[1]:.2f} -> KL={v:.4f}")
        return v

    res = minimize(obj, x0=[2.0, 3.0], method="Nelder-Mead",
                   options={"maxiter": TLG_SEARCH_MAXITER, "xatol": 0.2, "fatol": 1e-3})
    gc, gf = max(0.0, float(res.x[0])), max(0.0, float(res.x[1]))
    best = _budgeted_grow(n, rows, cols, alpha, gc, gf, Bc, Bf, d, e_real, SEED, scorer, real_den)
    param = f"d={d}, α={alpha:.2f}, γc={gc:.2f}, γf={gf:.2f}"
    if best["adj"] is None:
        log(f"    TLG forward: no edge-matched graph (γc={gc:.2f} γf={gf:.2f})")
        return dict(gic=np.nan, dist=np.nan, n_params=TLG_N_PARAMS, d_hat=d,
                    alpha=alpha, gc=gc, gf=gf, graph=None, param=param)
    gic = 2.0 * best["dist"] + 2.0 * TLG_N_PARAMS
    log(f"    TLG forward: γc={gc:.2f} γf={gf:.2f} KL={best['dist']:.4f} "
        f"GIC={gic:.4f} ({time.perf_counter()-t0:.1f}s, {nev['n']} evals)")
    return dict(gic=gic, dist=best["dist"], n_params=TLG_N_PARAMS, d_hat=d,
                alpha=alpha, gc=gc, gf=gf,
                graph=nx.from_numpy_array(best["adj"]), param=param)


def baseline_gic(G, name, gen_fn, n_params, real_den, scorer, seed):
    dens, rep = [], None
    for r in range(NRUNS):
        try:
            g = gen_fn(seed + r)
        except Exception:
            continue
        if rep is None:
            rep = g
        dens.append(scorer.compute_spectral_density(g)[0])
    if not dens:
        return None
    avg = np.mean(dens, axis=0)
    dist = float(entropy(real_den + 1e-10, avg + 1e-10))
    return dict(gic=2.0 * dist + 2.0 * n_params, dist=dist, n_params=n_params, graph=rep)


def process_region(region):
    path = DATA_DIR / f"{region}_graph.edges"
    if not path.exists():
        print(f"  {region}: file not found at {path} — skipping")
        return None
    t0 = time.perf_counter()
    G = _load_region(path)
    adj = nx.to_numpy_array(G)
    n = G.number_of_nodes()
    real_m = _metrics(G)
    print(f"\n=== {region}: n={n} E={real_m['edges']} density={real_m['density']:.4f} "
          f"clustering={real_m['clustering']:.3f} assort={real_m['assortativity']:.3f} ===")

    scorer = GraphInformationCriterion(G, model="LG", dist="KL")
    real_den, _ = scorer.compute_spectral_density(G)

    rows = [dict(model="Real", n_params=np.nan, gic=np.nan, gic_fit=np.nan,
                 gic_penalty=np.nan, kl=np.nan, param="—", **real_m)]

    # --- TLG (honest forward best-KL fit) ---
    tlg = fit_tlg(adj, real_den, scorer, G)
    tlg_m = (_metrics(tlg["graph"]) if tlg["graph"] is not None
             else dict(edges=np.nan, density=np.nan, clustering=np.nan,
                       assortativity=np.nan))
    rows.append(dict(model="TLG", n_params=tlg["n_params"], gic=tlg["gic"],
                     gic_fit=2.0 * tlg["dist"], gic_penalty=2.0 * tlg["n_params"],
                     kl=tlg["dist"], param=tlg["param"], **tlg_m))

    # --- baselines (closed-form) ---
    cf = closed_form_params(G)
    gens = _baseline_generators(G, cf)
    cf_params = dict(ER=f"p={cf['ER']:.4f}", BA=f"m={cf['BA']}",
                     WS=f"k={cf['WS_k']}, p={cf['WS_p']:.3f}", KR=f"d={cf['KR']}",
                     GRG=f"r={cf['GRG']:.3f}", SBM="Louvain blocks")
    for name in ("ER", "BA", "WS", "KR", "GRG", "SBM"):
        gen_fn, n_params = gens[name]
        if name == "SBM":  # resolve true k(k+1)/2 from one fit
            _, n_params = generate_sbm_from_real(G, seed=SEED)
        res = baseline_gic(G, name, gen_fn, n_params, real_den, scorer, SEED)
        if res is None:
            print(f"  {name}: generation failed — skipping")
            continue
        rows.append(dict(model=name, n_params=res["n_params"], gic=res["gic"],
                         gic_fit=2.0 * res["dist"], gic_penalty=2.0 * res["n_params"],
                         kl=res["dist"], param=cf_params[name], **_metrics(res["graph"])))
        print(f"  {name}: n_params={res['n_params']} KL={res['dist']:.4f} GIC={res['gic']:.4f}")

    df = pd.DataFrame(rows)
    models = df[df["model"] != "Real"].copy()
    models = models.sort_values("gic")
    models["rank"] = range(1, len(models) + 1)
    df = pd.concat([df[df["model"] == "Real"], models], ignore_index=True)
    df["rank"] = df["rank"].astype("Int64")
    df["region"] = region

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["region", "model", "rank", "gic", "gic_fit", "gic_penalty", "kl",
            "n_params", "edges", "density", "clustering", "assortativity", "param"]
    df[cols].to_csv(OUT_DIR / f"{region}_table.csv", index=False)
    _plot_bar(df, region, OUT_DIR / f"{region}_gic_bar.png")
    print(df[cols].to_string(index=False))
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    return df[cols]


# --------------------------------------------------------------------------- plots
CB = {"TLG": "#0072B2", "ER": "#E69F00", "BA": "#009E73", "WS": "#CC79A7",
      "KR": "#D55E00", "GRG": "#56B4E9", "SBM": "#000000"}


def _plot_bar(df, region, out_path):
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
    ax.set_title(f"Twitch {region}: GIC by family (lower = better; stacked terms)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
    for i, (_, r) in enumerate(m.iterrows()):
        ax.text(i, r["gic"], f"{r['gic']:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    print(f"TLG Twitch GIC (identifiable forward min-KL: degree+coarse+fine community)  "
          f"quick={QUICK} regions={REGIONS} d={TLG_D} K={TLG_K} "
          f"search={TLG_SEARCH_MAXITER} fine_res={TLG_FINE_RES} nruns={NRUNS} seed={SEED}")
    if not DATA_DIR.exists():
        print(f"No twitch data under {DATA_DIR} (gitignored — place *_graph.edges there).")
        return
    all_tables = []
    for region in REGIONS:
        t = process_region(region)
        if t is not None:
            all_tables.append(t)
    if not all_tables:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(OUT_DIR / "all_tables.csv", index=False)
    # rank summary across regions
    rk = combined[combined["model"] != "Real"].pivot_table(
        index="model", columns="region", values="rank", aggfunc="first")
    rk["mean_rank"] = rk.mean(axis=1)
    rk = rk.sort_values("mean_rank")
    rk.to_csv(OUT_DIR / "summary.csv")
    print("\n=== rank by region (mean rank, lower = better) ===")
    print(rk.to_string())
    print(f"\nWrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
