#!/usr/bin/env python3
"""Closed-form (moment-matched) baseline estimators vs fixed-interval grid"""
from __future__ import annotations

import math
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

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
for p in (_src, _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")

from logit_graph.gic import GraphInformationCriterion, _MODEL_N_PARAMS  # noqa: E402
from logit_graph.graph import GraphModel  # noqa: E402
from logit_graph.simulation import (  # noqa: E402
    _direct_er_at_sigma,
    _warm_start_er_p,
    estimate_sigma_only,
)


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


QUICK = os.environ.get("LG_TCF_QUICK", "0") == "1"
MIN_NODES = _int("LG_TCF_MIN_NODES", 50)
MAX_NODES = _int("LG_TCF_MAX_NODES", 150 if QUICK else 300)
MAX_NETS = _int("LG_TCF_MAX_NETS", 5 if QUICK else 30)
N_RUNS = _int("LG_TCF_N_RUNS", 3 if QUICK else 5)
GRID_POINTS = _int("LG_TCF_GRID_POINTS", 3 if QUICK else 5)
LG_D_LIST = [0, 1, 2]
LG_BURN_MIN = _int("LG_TCF_LG_BURN_MIN", 4000)
LG_BURN_PER_N = _int("LG_TCF_LG_BURN_PER_N", 25)
LG_STRIDE_MIN = _int("LG_TCF_LG_STRIDE_MIN", 600)
LG_STRIDE_PER_N = _int("LG_TCF_LG_STRIDE_PER_N", 6)
SEED = _int("LG_TCF_SEED", 12345)

GRID_INTERVALS = {"ER": (0.01, 0.25), "BA": (1, 8), "WS_k": (2, 10), "WS_p": (0.01, 0.5)}


def gic_of(real_nx, model_name, gen_fn, n_runs, seed):
    """GIC = 2*KL(real_density, mean_model_density) + 2*n_params."""
    n_params = _MODEL_N_PARAMS.get(model_name, 1)
    specs = []
    for r in range(n_runs):
        try:
            g = gen_fn(seed + r)
        except Exception:
            continue
        den, _ = GraphInformationCriterion(real_nx, model=model_name).compute_spectral_density(g)
        specs.append(den)
    if not specs:
        return np.nan, np.nan
    avg = np.mean(specs, axis=0)
    scorer = GraphInformationCriterion(real_nx, model=model_name, dist="KL")
    return float(scorer.calculate_gic(model_den=avg, n_params=n_params)), n_params


def _er(n, p):
    p = float(np.clip(p, 1e-6, 1.0))
    return lambda s: nx.erdos_renyi_graph(n, p, seed=s)


def _ba(n, m):
    m = int(np.clip(m, 1, n - 1))
    return lambda s: nx.barabasi_albert_graph(n, m, seed=s)


def _ws(n, k, p):
    k = int(np.clip(k, 2, n - 1))
    if k % 2 == 1:
        k += 1
    p = float(np.clip(p, 0.0, 1.0))
    return lambda s: nx.watts_strogatz_graph(n, k, p, seed=s)


def _kr(n, d):
    d = int(np.clip(d, 0, n - 1))
    if (n * d) % 2 == 1:
        d -= 1
    return lambda s: nx.random_regular_graph(d, n, seed=s)


def _grg(n, r):
    r = float(np.clip(r, 1e-3, 1.5))
    return lambda s: nx.random_geometric_graph(n, r, seed=s)


def closed_form_params(G):
    n = G.number_of_nodes()
    E = G.number_of_edges()
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
    r_grg = math.sqrt(kbar / (math.pi * (n - 1)))
    return dict(n=n, E=E, density=p_er, kbar=kbar,
                ER=p_er, BA=m_ba, WS_k=k_ws, WS_p=p_ws, KR=d_kr, GRG=r_grg,
                C_obs=C_obs)


def grid_best(real_nx, model_name, n, build_gen, lo, hi, n_runs, seed):
    best = (np.nan, None)
    for j, val in enumerate(np.linspace(lo, hi, GRID_POINTS)):
        g, _ = gic_of(real_nx, model_name, build_gen(val), n_runs, seed + 100 * j)
        if not np.isnan(g) and (best[1] is None or g < best[0]):
            best = (g, val)
    return best


def grid_best_ws(real_nx, n, n_runs, seed):
    klo, khi = GRID_INTERVALS["WS_k"]
    plo, phi = GRID_INTERVALS["WS_p"]
    best = (np.nan, None)
    j = 0
    for k in range(int(klo), int(khi) + 1, 2):
        for p in np.linspace(plo, phi, GRID_POINTS):
            g, _ = gic_of(real_nx, "WS", _ws(n, k, p), n_runs, seed + 100 * j)
            j += 1
            if not np.isnan(g) and (best[1] is None or g < best[0]):
                best = (g, (k, round(float(p), 3)))
    return best


def _lg_burn(n):
    return max(LG_BURN_MIN, LG_BURN_PER_N * n)


def _lg_stride(n):
    return max(LG_STRIDE_MIN, LG_STRIDE_PER_N * n)


def lg_gic_one_d(adj, d, seed):
    n = adj.shape[0]
    real_nx = nx.from_numpy_array(adj)
    scorer = GraphInformationCriterion(real_nx, model="LG", dist="KL")
    sigma, _ = estimate_sigma_only(adj, d=d, feature_mode="incremental")

    dens = []
    if d == 0:
        for r in range(N_RUNS):
            g = nx.from_numpy_array(_direct_er_at_sigma(n, sigma, seed=seed + r))
            dens.append(scorer.compute_spectral_density(g)[0])
    else:
        gm = GraphModel(n=n, d=d, sigma=sigma, er_p=_warm_start_er_p(sigma),
                        layer2=True, feature_mode="incremental", seed=seed)
        for _ in range(_lg_burn(n)):
            gm.add_remove_edge()
        for s in range(N_RUNS):
            for _ in range(_lg_stride(n)):
                gm.add_remove_edge()
            dens.append(scorer.compute_spectral_density(nx.from_numpy_array(gm.graph))[0])

    avg = np.mean(dens, axis=0)
    return float(scorer.calculate_gic(model_den=avg, n_params=1)), sigma


def lg_gic(adj, seed):
    best = (np.inf, None)
    for d in LG_D_LIST:
        try:
            g, _ = lg_gic_one_d(adj, d, seed + d)
            if g < best[0]:
                best = (g, d)
        except Exception as e:
            print(f"    LG d={d} failed: {e}")
    return best


def _ensure_data(data_dir):
    """Extract data/misc/twitter.tar.gz if the .edges files are not present."""
    if data_dir.exists() and any(data_dir.glob("*.edges")):
        return
    tarball = _repo_root / "data" / "misc" / "twitter.tar.gz"
    if not tarball.exists():
        return
    import tarfile
    print(f"extracting {tarball.relative_to(_repo_root)} ...")
    with tarfile.open(tarball) as tf:
        tf.extractall(_repo_root / "data" / "misc")


def _peek_size(path):
    """|V| of an .edges file via a single linear token scan (no nx parsing)."""
    nodes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                nodes.add(parts[0])
                nodes.add(parts[1])
    return len(nodes)


def _load_edges(path):
    """Undirected, self-loop-free largest connected component, relabeled 0..n-1."""
    G = nx.read_edgelist(path, nodetype=int)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def _sample_files():
    data_dir = _repo_root / "data" / "misc" / "twitter"
    _ensure_data(data_dir)
    allf = sorted(data_dir.glob("*.edges"))
    if not allf:
        return []
    inwin = [f for f in allf if MIN_NODES <= _peek_size(f) <= MAX_NODES]
    rng = random.Random(SEED)
    k = min(MAX_NETS, len(inwin))
    return sorted(rng.sample(inwin, k)), len(inwin), len(allf)


def main():
    out_dir = _here / "runs" / "twitter_closedform"
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled = _sample_files()
    if not sampled:
        print("No twitter .edges files found under data/misc/twitter/ — data is "
              "gitignored; place the SNAP twitter tarball/edges there first.")
        return
    files, n_in_window, n_total = sampled

    print(f"twitter closed-form experiment  seed={SEED}  quick={QUICK}  "
          f"window=[{MIN_NODES},{MAX_NODES}]  sampled={len(files)}/{n_in_window} "
          f"(of {n_total} ego nets)  n_runs={N_RUNS}  grid_points={GRID_POINTS}")

    rows = []
    for i, f in enumerate(files, 1):
        try:
            G = _load_edges(f)
        except Exception as e:
            print(f"  skip {f.stem}: {e}")
            continue
        n = G.number_of_nodes()
        if not (MIN_NODES <= n <= MAX_NODES):
            continue
        t0 = time.perf_counter()
        cf = closed_form_params(G)
        adj = nx.to_numpy_array(G)
        real_nx = nx.from_numpy_array(adj)
        name = f.stem
        print(f"\n[{i}] {name}  n={n}  E={cf['E']}  density={cf['density']:.3f}  "
              f"kbar={cf['kbar']:.1f}  C={cf['C_obs']:.3f}")

        lg_val, lg_d = lg_gic(adj, SEED)
        print(f"    LG     gic={lg_val:.3f} (d={lg_d})")

        er_grid = grid_best(real_nx, "ER", n,
                            lambda v: _er(n, v), *GRID_INTERVALS["ER"], N_RUNS, SEED)
        er_cf, _ = gic_of(real_nx, "ER", _er(n, cf["ER"]), N_RUNS, SEED)
        print(f"    ER     grid={er_grid[0]:.3f} (p={er_grid[1]:.3f})   "
              f"cf={er_cf:.3f} (p={cf['ER']:.3f})")

        ba_grid = grid_best(real_nx, "BA", n,
                            lambda v: _ba(n, v), *GRID_INTERVALS["BA"], N_RUNS, SEED)
        ba_cf, _ = gic_of(real_nx, "BA", _ba(n, cf["BA"]), N_RUNS, SEED)
        print(f"    BA     grid={ba_grid[0]:.3f} (m={ba_grid[1]:.1f})   "
              f"cf={ba_cf:.3f} (m={cf['BA']})")

        ws_grid = grid_best_ws(real_nx, n, N_RUNS, SEED)
        ws_cf, _ = gic_of(real_nx, "WS", _ws(n, cf["WS_k"], cf["WS_p"]), N_RUNS, SEED)
        print(f"    WS     grid={ws_grid[0]:.3f} (k,p={ws_grid[1]})   "
              f"cf={ws_cf:.3f} (k={cf['WS_k']},p={cf['WS_p']:.3f})")

        kr_cf, _ = gic_of(real_nx, "KR", _kr(n, cf["KR"]), N_RUNS, SEED)
        grg_cf, _ = gic_of(real_nx, "GRG", _grg(n, cf["GRG"]), N_RUNS, SEED)
        print(f"    KR cf={kr_cf:.3f} (d={cf['KR']})   GRG cf={grg_cf:.3f} (r={cf['GRG']:.3f})"
              f"   [{time.perf_counter() - t0:.0f}s]")

        rows.append(dict(
            name=name, n=n, E=cf["E"], density=cf["density"], kbar=cf["kbar"],
            LG=lg_val, LG_d=lg_d,
            ER_grid=er_grid[0], ER_cf=er_cf,
            BA_grid=ba_grid[0], BA_cf=ba_cf,
            WS_grid=ws_grid[0], WS_cf=ws_cf,
            KR_cf=kr_cf, GRG_cf=grg_cf,
        ))

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nNo ego nets scored.")
        return
    df.to_csv(out_dir / "results.csv", index=False)

    print("\n" + "=" * 78)
    print("AGGREGATE (mean GIC across ego nets, lower = better)")
    print("=" * 78)
    for fam in ("ER", "BA", "WS"):
        g = df[f"{fam}_grid"].mean()
        c = df[f"{fam}_cf"].mean()
        win = (df[f"{fam}_cf"] < df[f"{fam}_grid"]).mean()
        print(f"  {fam}:  grid={g:7.3f}   closed-form={c:7.3f}   "
              f"Δ(grid-cf)={g - c:+7.3f}   cf_wins={win:.0%}")
    print(f"  KR(cf)={df['KR_cf'].mean():.3f}   GRG(cf)={df['GRG_cf'].mean():.3f}   "
          f"LG={df['LG'].mean():.3f}")

    rank_cols = {"LG": "LG", "ER": "ER_cf", "BA": "BA_cf", "WS": "WS_cf",
                 "KR": "KR_cf", "GRG": "GRG_cf"}
    ranks = df[list(rank_cols.values())].rank(axis=1)
    ranks.columns = list(rank_cols.keys())
    print("\nMean rank (LG + closed-form baselines, lower = better):")
    print(ranks.mean().sort_values().to_string())
    print(f"\nWrote {out_dir / 'results.csv'}  ({len(df)} ego nets)")


if __name__ == "__main__":
    main()
