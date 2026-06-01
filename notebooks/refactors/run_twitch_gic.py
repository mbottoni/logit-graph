#!/usr/bin/env python3
"""Fit LG + ER/WS/BA on every Twitch country network and rank by GIC.

Structurally identical to ``run_facebook_gic.py`` / ``run_arxiv_gic.py``
— per-graph single-pass fit using KPM spectral density on the SPARSE
normalised Laplacian, sampled logistic regression for σ/β, dense LG
MCMC at d=0 with a KPM-based convergence monitor. NO dense
``eigvalsh`` inside the convergence loop — that's the bottleneck the
existing notebook hit (5–15 hours per fit on the larger countries).
This script does each country in ~30–60s.

Loops over the 6 country graphs in ``data/twitch/graphs_processed/``
(PTBR, RU, ES, FR, ENGB, DE — n=1,912 to n=9,498) and emits both
per-country result rows and an aggregate leaderboard.

Env-var overrides:
  LG_TWITCH_MAX_NODES    cap on |V|        (default 10000 = all 6)
  LG_TWITCH_MIN_NODES    floor on |V|      (default 500)
  LG_TWITCH_MAX_ITER     LG MCMC iters     (default 30_000)
  LG_TWITCH_CHECK        spectral check    (default 3_000)
  LG_TWITCH_WARM_UP      iters before patience (default 6_000)
  LG_TWITCH_PATIENCE     no-improvement checks  (default 30)
  LG_TWITCH_SAMPLE_EDGES σ/β estimation sample  (default 30_000)
  LG_TWITCH_KPM_MOMENTS  Chebyshev moments      (default 120)
  LG_TWITCH_KPM_PROBES   random probes          (default 30)
  LG_TWITCH_USE_CACHE    reuse cached per-country results (default 1)
  LG_TWITCH_SEED         RNG seed                (default 42)
  LG_TWITCH_QUICK        smoke (skips DE/FR/ENGB, fewer iters)

  make gic-twitch         full preset (~3-5 min, all 6 countries)
  make gic-twitch-quick   smoke (~1 min, smaller countries only)
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Optional


COUNTRIES = ["PTBR", "RU", "ES", "FR", "ENGB", "DE"]  # ascending n


def _get_int(env: str, default: int) -> int:
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


def _get_optional_int(env: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(env)
    if raw is None:
        return default
    if raw.lower() in ("none", "", "null"):
        return None
    return int(raw)


def _fmt(s: float) -> str:
    if s < 60:
        return f"{s:5.1f}s"
    m, rem = divmod(s, 60)
    return f"{int(m):2d}m{int(rem):02d}s"


def _sparse_normalised_laplacian(G_or_adj):
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp

    if isinstance(G_or_adj, nx.Graph):
        A = nx.adjacency_matrix(G_or_adj).astype(np.float64)
    elif isinstance(G_or_adj, np.ndarray):
        A = sp.csr_matrix(G_or_adj)
    else:
        A = sp.csr_matrix(G_or_adj)
    n = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    return sp.eye(n, format="csr") - (D_inv_sqrt @ A @ D_inv_sqrt)


def fit_country(country: str, edge_path: Path, *, max_iter: int, check_interval: int,
                warm_up: int, patience: int, sample_edges: int, kpm_moments: int,
                kpm_probes: int, seed: int):
    """Single-country fit: returns dict with per-model GICs + LG params."""
    import numpy as np
    import networkx as nx
    import statsmodels.api as sm
    from logit_graph.gic import kpm_spectral_density
    from logit_graph.graph import GraphModel
    from scipy.stats import entropy as _entropy

    print(f"\n=== {country}: load + KPM real density")
    t0 = time.perf_counter()
    G_directed = nx.read_edgelist(edge_path, comments="#", nodetype=int)
    G = G_directed.to_undirected() if G_directed.is_directed() else G_directed
    G.remove_edges_from(nx.selfloop_edges(G))
    gcc_nodes = max(nx.connected_components(G), key=len)
    G_gcc = nx.convert_node_labels_to_integers(G.subgraph(gcc_nodes).copy())
    n = G_gcc.number_of_nodes()
    m = G_gcc.number_of_edges()
    real_density_val = nx.density(G_gcc)
    real_avg_deg = 2 * m / n
    del G_directed, G
    gc.collect()

    L_real = _sparse_normalised_laplacian(G_gcc)
    real_density, _ = kpm_spectral_density(
        L_real, n_bins=50, n_moments=kpm_moments,
        n_probes=kpm_probes, seed=seed,
    )
    print(f"  {country}: n={n:,} m={m:,} density={real_density_val:.5f}  "
          f"[{_fmt(time.perf_counter() - t0)}]")

    # 1. Estimate LG σ, β by sampled logistic regression
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    degree_dict = dict(G_gcc.degree())
    sum_degrees = np.array([degree_dict.get(v, 0) for v in range(n)], dtype=np.float64)
    edges_all = list(G_gcc.edges())
    n_sample = min(sample_edges, len(edges_all))
    edge_idx = rng.choice(len(edges_all), size=n_sample, replace=False)
    edge_pairs = [edges_all[i] for i in edge_idx]
    edge_set = set((min(a, b), max(a, b)) for a, b in edges_all)
    non_edge_pairs = []
    while len(non_edge_pairs) < n_sample:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair not in edge_set:
            non_edge_pairs.append(pair)
    pairs = edge_pairs + non_edge_pairs
    labels = [1] * len(edge_pairs) + [0] * len(non_edge_pairs)
    feats = np.array([sum_degrees[i] + sum_degrees[j] for i, j in pairs]).reshape(-1, 1)
    features = sm.add_constant(feats)
    result = sm.Logit(labels, features).fit_regularized(method="l1", alpha=0, disp=False)
    sigma = float(result.params[0])
    beta = float(result.params[1])
    d_lg = 0
    print(f"  σ={sigma:+.4f}  β={beta:+.4f}  d={d_lg}  "
          f"[{_fmt(time.perf_counter() - t0)}]")

    # 2. LG MCMC fit
    er_seed_p = 1.25 * (real_density_val / 2)
    gm = GraphModel(n=n, d=d_lg, sigma=sigma, beta=beta, er_p=er_seed_p, seed=seed)
    print(f"  LG init: edges={gm._edge_count:,}  dense adj={gm.graph.nbytes / 1e9:.2f} GB")
    best_graph = gm.graph.copy()
    best_diff = float("inf")
    best_iter = 0
    no_improve = 0
    t_start = time.perf_counter()

    for i in range(max_iter):
        gm.add_remove_edge()
        if i > 0 and i % check_interval == 0:
            L_cur = _sparse_normalised_laplacian(gm.graph)
            cur_density, _ = kpm_spectral_density(
                L_cur, n_bins=50, n_moments=kpm_moments,
                n_probes=kpm_probes, seed=seed + i,
            )
            diff = float(np.linalg.norm(cur_density - real_density))
            if diff < best_diff:
                best_diff = diff
                best_graph = gm.graph.copy()
                best_iter = i
                no_improve = 0
            elif i >= warm_up:
                no_improve += 1
            elapsed = time.perf_counter() - t_start
            print(
                f"    iter {i:>6,}/{max_iter:,}  L2={diff:.4f}  best={best_diff:.4f}  "
                f"edges={gm._edge_count:,}/{m:,}  pat={no_improve}/{patience}  "
                f"wall={_fmt(elapsed)}"
            )
            if i >= warm_up and no_improve >= patience:
                print(f"    converged (no improvement for {patience} checks)")
                break

    t_lg = time.perf_counter() - t_start
    print(f"  LG fit done in {_fmt(t_lg)}  best_iter={best_iter:,}  best_L2={best_diff:.4f}")

    # 3. GIC scoring
    eps = 1e-10
    def _gic(p):
        return 0.5 * (_entropy(real_density + eps, p + eps)
                      + _entropy(p + eps, real_density + eps))

    L_lg = _sparse_normalised_laplacian(best_graph)
    lg_density, _ = kpm_spectral_density(
        L_lg, n_bins=50, n_moments=kpm_moments,
        n_probes=kpm_probes, seed=seed + 99_991,
    )
    lg_gic = float(_gic(lg_density))
    results = {"LG": {"gic": lg_gic, "param": f"σ={sigma:.3f},β={beta:.3f},d={d_lg}",
                      "n": n, "m": int(best_graph.sum() // 2)}}
    print(f"  LG  GIC={lg_gic:.4f}")
    del gm
    gc.collect()

    baselines = [
        ("ER", "Erdos-Renyi",
         lambda: nx.erdos_renyi_graph(n, real_density_val, seed=seed)),
        ("WS", "Watts-Strogatz",
         lambda: nx.watts_strogatz_graph(n, max(2, int(round(real_avg_deg))), 0.1, seed=seed)),
        ("BA", "Barabasi-Albert",
         lambda: nx.barabasi_albert_graph(n, max(1, int(round(real_avg_deg / 2))), seed=seed)),
    ]
    for name, label, gen in baselines:
        G_m = gen()
        L_m = _sparse_normalised_laplacian(G_m)
        m_density, _ = kpm_spectral_density(
            L_m, n_bins=50, n_moments=kpm_moments,
            n_probes=kpm_probes, seed=seed + hash(name) % 1000,
        )
        gic_val = float(_gic(m_density))
        results[name] = {"gic": gic_val, "param": label,
                         "n": G_m.number_of_nodes(), "m": G_m.number_of_edges()}
        print(f"  {name:<2s}  GIC={gic_val:.4f}")
        del G_m, L_m
        gc.collect()

    return {
        "country": country, "n": n, "m": m, "density": real_density_val,
        "results": results,
        "lg_params": {"sigma": sigma, "beta": beta, "d": d_lg,
                      "best_iter": best_iter, "best_L2": best_diff, "fit_seconds": t_lg},
    }


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    warnings.filterwarnings("ignore")

    _here = Path(__file__).resolve().parent
    _repo_root = _here.parents[1]
    _src = _repo_root / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    import pandas as pd

    quick = os.environ.get("LG_TWITCH_QUICK", "0") == "1"
    max_nodes_default = 5000 if quick else 10000
    max_iter_default = 6_000 if quick else 30_000
    check_default = 1_500 if quick else 3_000
    warm_up_default = 3_000 if quick else 6_000
    patience_default = 20 if quick else 30
    sample_edges_default = 15_000 if quick else 30_000
    kpm_moments_default = 80 if quick else 120
    kpm_probes_default = 20 if quick else 30

    max_nodes = _get_optional_int("LG_TWITCH_MAX_NODES", max_nodes_default)
    min_nodes = _get_int("LG_TWITCH_MIN_NODES", 500)
    max_iter = _get_int("LG_TWITCH_MAX_ITER", max_iter_default)
    check_interval = _get_int("LG_TWITCH_CHECK", check_default)
    warm_up = _get_int("LG_TWITCH_WARM_UP", warm_up_default)
    patience = _get_int("LG_TWITCH_PATIENCE", patience_default)
    sample_edges = _get_int("LG_TWITCH_SAMPLE_EDGES", sample_edges_default)
    kpm_moments = _get_int("LG_TWITCH_KPM_MOMENTS", kpm_moments_default)
    kpm_probes = _get_int("LG_TWITCH_KPM_PROBES", kpm_probes_default)
    seed = _get_int("LG_TWITCH_SEED", 42)
    use_cache = os.environ.get("LG_TWITCH_USE_CACHE", "1") == "1"

    out_dir = _here / "runs" / "twitch_fast"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_sig = json.dumps(
        [max_iter, check_interval, sample_edges, kpm_moments, kpm_probes, seed],
        sort_keys=True,
    )

    print(
        f"Twitch country GIC ranking  max_nodes={max_nodes}  max_iter={max_iter:,}  "
        f"check={check_interval:,}  patience={patience}  sample_edges={sample_edges:,}  "
        f"kpm_moments={kpm_moments}  cache={use_cache}  quick={quick}"
    )

    twitch_root = _repo_root / "data" / "twitch" / "graphs_processed"
    if not twitch_root.is_dir():
        print(f"Twitch data not found at {twitch_root}; aborting.")
        return

    rows = []
    per_country_summary = []
    for country in COUNTRIES:
        edge_path = twitch_root / f"{country}_graph.edges"
        if not edge_path.is_file():
            print(f"  {country}: file missing at {edge_path}, skipping")
            continue
        nodes = set()
        with open(edge_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    nodes.add(parts[0])
                    nodes.add(parts[1])
        n_peek = len(nodes)
        if n_peek < min_nodes:
            print(f"  {country}: n={n_peek} below min_nodes={min_nodes}, skipping")
            continue
        if max_nodes is not None and n_peek > max_nodes:
            print(f"  {country}: n={n_peek} above max_nodes={max_nodes}, skipping")
            continue

        country_cache = out_dir / f"{country}.pkl"
        if use_cache and country_cache.is_file():
            cached = pickle.load(country_cache.open("rb"))
            if cached.get("config_signature") == cfg_sig:
                print(f"\n=== {country}: CACHE HIT (best={cached['best_model']}, GIC={cached['best_gic']:.3f})")
                rows.extend(cached["rows"])
                per_country_summary.append(cached["summary"])
                continue

        try:
            cdata = fit_country(country, edge_path,
                                max_iter=max_iter, check_interval=check_interval,
                                warm_up=warm_up, patience=patience,
                                sample_edges=sample_edges, kpm_moments=kpm_moments,
                                kpm_probes=kpm_probes, seed=seed)
        except Exception as exc:
            print(f"  {country}: FAILED — {exc}")
            import traceback
            traceback.print_exc()
            continue

        country_rows = []
        for model, data in cdata["results"].items():
            country_rows.append({
                "country": country, "n": cdata["n"], "m": cdata["m"],
                "model": model, "gic": data["gic"], "param": data["param"],
            })
        best_row = min(country_rows, key=lambda r: r["gic"])
        summary_row = {
            "country": country, "n": cdata["n"], "m": cdata["m"],
            "best_model": best_row["model"], "best_gic": best_row["gic"],
            "lg_sigma": cdata["lg_params"]["sigma"],
            "lg_beta": cdata["lg_params"]["beta"],
            "lg_fit_seconds": cdata["lg_params"]["fit_seconds"],
        }
        rows.extend(country_rows)
        per_country_summary.append(summary_row)
        with country_cache.open("wb") as f:
            pickle.dump({"rows": country_rows, "summary": summary_row,
                         "best_model": best_row["model"], "best_gic": best_row["gic"],
                         "config_signature": cfg_sig}, f)

    if not rows:
        print("\nNo countries processed.")
        return

    long_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(per_country_summary)
    long_df.to_csv(out_dir / "long.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    pivot = long_df.pivot_table(index="country", columns="model", values="gic").reindex(
        columns=["LG", "ER", "WS", "BA"]
    )
    rank = pivot.rank(axis=1, method="min")
    pivot.to_csv(out_dir / "gic_pivot.csv")
    rank.to_csv(out_dir / "gic_rank_pivot.csv")

    print(f"\n=== Per-country results ===")
    print(summary_df.to_string(index=False))
    print(f"\n=== GIC matrix ===")
    print(pivot.to_string())
    print(f"\n=== Mean GIC rank (lower = better fit) ===")
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    for i, (model, r) in enumerate(rank.mean().sort_values().items()):
        wins = (pivot.idxmin(axis=1) == model).sum()
        print(f"  {medals[i]} {model:<3s}  mean rank = {r:.3f}   wins = {wins}/{len(pivot)}")

    print(f"\nArtifacts: {out_dir}")


if __name__ == "__main__":
    main()
