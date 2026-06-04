#!/usr/bin/env python3
"""Quick experiment: closed-form (moment-matched) baseline estimators vs the
current fixed-interval grid search, on gplus ego networks, alongside LG.

For each gplus ego network in a small size window we score, by spectral GIC
(2*KL + 2*n_params, KL on the 50-bin normalized-Laplacian density):

  * LG            -- the real pipeline fit (estimate sigma, generate, best-of)
  * ER/BA/WS      -- two ways each:
       grid : current method (fixed interval, grid_points pts, pick min GIC)
       cf   : closed-form moment estimate (no search)
  * KR/GRG        -- closed-form only (bonus families)

Closed-form estimators (n nodes, E edges, kbar = 2E/n avg degree):
  ER  p = 2E/(n(n-1))                      (exact MLE)
  BA  m = round(E/n)            in [1, n)   (edge count E = m(n-m))
  WS  k = 2*round(E/n) (even)   in [2, n)   (E = nk/2, rewiring conserves edges)
  WS  p = 1 - (C_obs/C0)^(1/3), C0 = 3(k-2)/(4(k-1))   (clustering moment)
  KR  d = round(kbar)          (nd even)   (E = nd/2)
  GRG r = sqrt(kbar / (pi*(n-1)))          (E[deg] ~ (n-1) pi r^2, 2-D)

Read-only w.r.t. the library: uses logit_graph's own GIC scorer and the same
GraphModelComparator LG path; writes only under runs/gplus_closedform/.

Env knobs:
  LG_CF_MIN_NODES (50)  LG_CF_MAX_NODES (120)  LG_CF_MAX_NETS (8)
  LG_CF_N_RUNS (3)      LG_CF_GRID_POINTS (3)  LG_CF_LG_ITER (1500)
"""
from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

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
from logit_graph.simulation import GraphModelComparator  # noqa: E402


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


MIN_NODES = _int("LG_CF_MIN_NODES", 50)
MAX_NODES = _int("LG_CF_MAX_NODES", 120)
MAX_NETS = _int("LG_CF_MAX_NETS", 8)
N_RUNS = _int("LG_CF_N_RUNS", 3)
GRID_POINTS = _int("LG_CF_GRID_POINTS", 3)
LG_ITER = _int("LG_CF_LG_ITER", 1500)
SEED = 12345

# Current fixed intervals (mirror simulation.py defaults).
GRID_INTERVALS = {"ER": (0.01, 0.25), "BA": (1, 8), "WS_k": (2, 10), "WS_p": (0.01, 0.5)}


# ---------------------------------------------------------------------------
# GIC scoring (apples-to-apples with the pipeline)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Closed-form estimators
# ---------------------------------------------------------------------------

def closed_form_params(G):
    n = G.number_of_nodes()
    E = G.number_of_edges()
    kbar = 2 * E / n
    p_er = 2 * E / (n * (n - 1))
    m_ba = max(1, round(E / n))
    k_ws = max(2, int(round(kbar)))
    if k_ws % 2 == 1:
        k_ws += 1
    # WS rewiring p from clustering moment.
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


# ---------------------------------------------------------------------------
# Current-style grid search (fixed interval, pick min GIC = best case for grid)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# LG via the real pipeline (best of d in {0,1})
# ---------------------------------------------------------------------------

def lg_gic(adj, seed):
    lg_params = dict(max_iterations=LG_ITER, patience=300, edge_delta=None,
                     min_gic_threshold=5, check_interval=50)
    cmp = GraphModelComparator(d_list=[0, 1], lg_params=lg_params, dist_type="KL",
                               verbose=False, random_state=seed)
    best = (np.inf, None)
    for d in (0, 1):
        try:
            _, sigma, gic_val, *_ = cmp._get_logit_graph_for_d(adj, d)
            if gic_val < best[0]:
                best = (gic_val, d)
        except Exception as e:
            print(f"    LG d={d} failed: {e}")
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = _repo_root / "data" / "misc" / "gplus"
    files = sorted(data_dir.glob("*.edges"))
    out_dir = _here / "runs" / "gplus_closedform"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"closed-form baseline experiment  window=[{MIN_NODES},{MAX_NODES}]  "
          f"max_nets={MAX_NETS}  n_runs={N_RUNS}  grid_points={GRID_POINTS}  lg_iter={LG_ITER}")

    rows = []
    picked = 0
    for f in files:
        if picked >= MAX_NETS:
            break
        G = nx.read_edgelist(f, create_using=nx.DiGraph).to_undirected()
        G = nx.convert_node_labels_to_integers(G)
        n = G.number_of_nodes()
        if not (MIN_NODES <= n <= MAX_NODES):
            continue
        picked += 1
        cf = closed_form_params(G)
        adj = nx.to_numpy_array(G)
        real_nx = nx.from_numpy_array(adj)
        name = f.stem[:10]
        print(f"\n[{picked}] {name}  n={n}  E={cf['E']}  density={cf['density']:.3f}  "
              f"kbar={cf['kbar']:.1f}  C={cf['C_obs']:.3f}")

        # LG
        lg_val, lg_d = lg_gic(adj, SEED)
        print(f"    LG     gic={lg_val:.3f} (d={lg_d})")

        # ER
        er_grid = grid_best(real_nx, "ER", n,
                            lambda v: _er(n, v), *GRID_INTERVALS["ER"], N_RUNS, SEED)
        er_cf, _ = gic_of(real_nx, "ER", _er(n, cf["ER"]), N_RUNS, SEED)
        print(f"    ER     grid={er_grid[0]:.3f} (p={er_grid[1]:.3f})   "
              f"cf={er_cf:.3f} (p={cf['ER']:.3f})")

        # BA
        ba_grid = grid_best(real_nx, "BA", n,
                            lambda v: _ba(n, v), *GRID_INTERVALS["BA"], N_RUNS, SEED)
        ba_cf, _ = gic_of(real_nx, "BA", _ba(n, cf["BA"]), N_RUNS, SEED)
        print(f"    BA     grid={ba_grid[0]:.3f} (m={ba_grid[1]:.1f})   "
              f"cf={ba_cf:.3f} (m={cf['BA']})")

        # WS
        ws_grid = grid_best_ws(real_nx, n, N_RUNS, SEED)
        ws_cf, _ = gic_of(real_nx, "WS", _ws(n, cf["WS_k"], cf["WS_p"]), N_RUNS, SEED)
        print(f"    WS     grid={ws_grid[0]:.3f} (k,p={ws_grid[1]})   "
              f"cf={ws_cf:.3f} (k={cf['WS_k']},p={cf['WS_p']:.3f})")

        # KR / GRG closed-form only
        kr_cf, _ = gic_of(real_nx, "KR", _kr(n, cf["KR"]), N_RUNS, SEED)
        grg_cf, _ = gic_of(real_nx, "GRG", _grg(n, cf["GRG"]), N_RUNS, SEED)
        print(f"    KR cf={kr_cf:.3f} (d={cf['KR']})   GRG cf={grg_cf:.3f} (r={cf['GRG']:.3f})")

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
        print("\nNo networks in window.")
        return
    df.to_csv(out_dir / "results.csv", index=False)

    print("\n" + "=" * 78)
    print("AGGREGATE (mean GIC across networks, lower = better)")
    print("=" * 78)
    for fam in ("ER", "BA", "WS"):
        g = df[f"{fam}_grid"].mean()
        c = df[f"{fam}_cf"].mean()
        win = (df[f"{fam}_cf"] < df[f"{fam}_grid"]).mean()
        print(f"  {fam}:  grid={g:7.3f}   closed-form={c:7.3f}   "
              f"Δ(grid-cf)={g - c:+7.3f}   cf_wins={win:.0%}")
    print(f"  KR(cf)={df['KR_cf'].mean():.3f}   GRG(cf)={df['GRG_cf'].mean():.3f}   "
          f"LG={df['LG'].mean():.3f}")

    # Ranking among LG + closed-form baselines (lower GIC = rank 1)
    rank_cols = {"LG": "LG", "ER": "ER_cf", "BA": "BA_cf", "WS": "WS_cf",
                 "KR": "KR_cf", "GRG": "GRG_cf"}
    ranks = df[list(rank_cols.values())].rank(axis=1)
    ranks.columns = list(rank_cols.keys())
    print("\nMean rank (LG + closed-form baselines, lower = better):")
    print(ranks.mean().sort_values().to_string())
    print(f"\nWrote {out_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
