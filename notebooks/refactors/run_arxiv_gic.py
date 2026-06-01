#!/usr/bin/env python3
"""Fit LG + ER/WS/BA on the FULL cit-HepTh arxiv citation network and rank by GIC.

Reproduces `notebooks/citation/0-citation.ipynb` as a CLI script — single
large network (n≈27,400, m≈350k), no subsampling. Memory + time are
managed by:
  - KPM spectral density on the sparse normalised Laplacian (no O(n³)
    eigendecomposition — uses logit_graph.gic.kpm_spectral_density)
  - LG σ/β estimation from a random ~50k edge / 50k non-edge sample
    (statsmodels logistic regression — avoids dense pair iteration)
  - LG MCMC stores ONE dense adjacency (~6 GB at n=27k) and a best-graph
    copy; baselines are generated and scored sparsely with NetworkX + KPM

Env-var overrides:
  LG_ARXIV_DATA          path to cit-HepTh.txt (default data/citation_networks/cit-HepTh.txt)
  LG_ARXIV_MAX_ITER      LG MCMC iterations          (default 50_000)
  LG_ARXIV_CHECK         spectral-density check interval (default 5_000)
  LG_ARXIV_WARM_UP       iterations before patience kicks in (default 10_000)
  LG_ARXIV_PATIENCE      stop after this many no-improvement checks (default 100)
  LG_ARXIV_SAMPLE_EDGES  σ/β estimation sample size  (default 50_000)
  LG_ARXIV_KPM_MOMENTS   Chebyshev moments           (default 150)
  LG_ARXIV_KPM_PROBES    random probes               (default 40)
  LG_ARXIV_USE_CACHE     reuse cached fit results    (default 1)
  LG_ARXIV_SEED          RNG seed                    (default 42)
  LG_ARXIV_QUICK         smoke (MAX_ITER=10k, fewer KPM moments)

  make gic-arxiv         full preset (~3-5 min on 4 cores)
  make gic-arxiv-quick   smoke (~30-60s)
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


def _get_int(env: str, default: int) -> int:
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


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
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))

    import numpy as np
    import networkx as nx
    import pandas as pd
    import statsmodels.api as sm
    from logit_graph.gic import kpm_spectral_density
    from logit_graph.graph import GraphModel

    quick = os.environ.get("LG_ARXIV_QUICK", "0") == "1"
    data_path = Path(os.environ.get(
        "LG_ARXIV_DATA",
        str(_repo_root / "data" / "citation_networks" / "cit-HepTh.txt"),
    ))
    max_iter = _get_int("LG_ARXIV_MAX_ITER", 10_000 if quick else 50_000)
    check_interval = _get_int("LG_ARXIV_CHECK", 1_000 if quick else 5_000)
    warm_up = _get_int("LG_ARXIV_WARM_UP", 2_000 if quick else 10_000)
    patience = _get_int("LG_ARXIV_PATIENCE", 30 if quick else 100)
    sample_edges = _get_int("LG_ARXIV_SAMPLE_EDGES", 20_000 if quick else 50_000)
    kpm_moments = _get_int("LG_ARXIV_KPM_MOMENTS", 80 if quick else 150)
    kpm_probes = _get_int("LG_ARXIV_KPM_PROBES", 20 if quick else 40)
    seed = _get_int("LG_ARXIV_SEED", 42)
    use_cache = os.environ.get("LG_ARXIV_USE_CACHE", "1") == "1"

    out_dir = _here / "runs" / "arxiv"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "arxiv_full_results.pkl"

    print(
        f"arxiv (cit-HepTh) full-graph GIC ranking  "
        f"max_iter={max_iter:,}  check={check_interval:,}  patience={patience}  "
        f"sample_edges={sample_edges:,}  kpm_moments={kpm_moments}  "
        f"kpm_probes={kpm_probes}  quick={quick}"
    )

    if use_cache and cache_path.is_file():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        cfg_key = _config_signature(max_iter, check_interval, sample_edges,
                                    kpm_moments, kpm_probes, seed)
        if cached.get("config_signature") == cfg_key:
            print(f"\nCache hit at {cache_path}; reprinting summary.")
            _print_summary(cached["summary_df"], cached["n"], cached["m"])
            return
        print("Cache present but config changed; recomputing.")

    # 1. Load + take giant connected component
    print(f"\n=== 1. Load full graph from {data_path}")
    t0 = time.perf_counter()
    G_directed = nx.read_edgelist(
        data_path, comments="#", delimiter="\t",
        create_using=nx.DiGraph(), nodetype=int,
    )
    G_full = G_directed.to_undirected()
    G_full.remove_edges_from(nx.selfloop_edges(G_full))
    gcc_nodes = max(nx.connected_components(G_full), key=len)
    G_gcc = nx.convert_node_labels_to_integers(G_full.subgraph(gcc_nodes).copy())
    n = G_gcc.number_of_nodes()
    m = G_gcc.number_of_edges()
    del G_directed, G_full
    gc.collect()
    print(f"  GCC: n={n:,}  m={m:,}  density={nx.density(G_gcc):.6f}  "
          f"[{_fmt(time.perf_counter() - t0)}]")

    # 2. KPM spectral density of the real graph (the target distribution)
    print(f"\n=== 2. KPM spectral density of real graph")
    t0 = time.perf_counter()
    L_real = _sparse_normalised_laplacian(G_gcc)
    real_density, _ = kpm_spectral_density(
        L_real, n_bins=50, n_moments=kpm_moments,
        n_probes=kpm_probes, seed=seed,
    )
    t_kpm = time.perf_counter() - t0
    print(f"  KPM done in {_fmt(t_kpm)}")

    # 3. AIC-select d ∈ {0, 1, 2} from sampled (edge, non-edge) logit fits
    print(f"\n=== 3. AIC-select d from σ, β logit on "
          f"{sample_edges:,} edges + {sample_edges:,} non-edges")
    from lg_aic_utils import aic_select_d, sample_pairs as _sample_pairs
    d_candidates = [int(x) for x in
                    os.environ.get("LG_ARXIV_D_CANDIDATES", "0,1,2").split(",")
                    if x.strip()]
    pairs, labels = _sample_pairs(G_gcc, sample_edges, seed)
    best_aic, aic_table = aic_select_d(G_gcc, pairs, labels, d_candidates)
    for r in aic_table:
        marker = "  ← AIC pick" if r["d"] == best_aic["d"] else ""
        print(f"    d={r['d']}  σ={r['sigma']:+.4f}  β={r['beta']:+.4f}  "
              f"loglik={r['loglik']:.1f}  AIC={r['aic']:.1f}  "
              f"({_fmt(r['seconds'])}){marker}")
    sigma = best_aic["sigma"]
    beta = best_aic["beta"]
    d_lg = best_aic["d"]
    print(f"  → d̂={d_lg}  σ={sigma:+.4f}  β={beta:+.4f}")

    # 4. LG MCMC fit on the full graph
    print(f"\n=== 4. LG MCMC fit  (max_iter={max_iter:,}, check every {check_interval:,})")
    er_seed_p = 1.25 * (nx.density(G_gcc) / 2)
    gm = GraphModel(n=n, d=d_lg, sigma=sigma, beta=beta, er_p=er_seed_p, seed=seed)
    print(f"  Initial: edges={gm._edge_count:,}  dense adj={gm.graph.nbytes / 1e9:.2f} GB")
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
                f"  iter {i:>7,}/{max_iter:,}  L2={diff:.4f}  best={best_diff:.4f}  "
                f"edges={gm._edge_count:,}/{m:,}  pat={no_improve}/{patience}  "
                f"wall={_fmt(elapsed)}"
            )
            if i >= warm_up and no_improve >= patience:
                print(f"  converged (no improvement for {patience} checks)")
                break

    t_fit = time.perf_counter() - t_start
    print(f"  LG fit done in {_fmt(t_fit)}  best_iter={best_iter:,}  "
          f"best_L2={best_diff:.4f}")

    # 5. Baselines + GIC
    print(f"\n=== 5. GIC for LG + ER / WS / BA")
    results: dict[str, dict] = {}
    real_density_val = nx.density(G_gcc)
    real_avg_deg = 2 * m / n
    eps = 1e-10

    def _gic(p):
        # Symmetric KL acts as our GIC stand-in (matches the notebook's compute_gic).
        from scipy.stats import entropy as _entropy
        return 0.5 * (_entropy(real_density + eps, p + eps)
                      + _entropy(p + eps, real_density + eps))

    # LG
    t0 = time.perf_counter()
    L_lg = _sparse_normalised_laplacian(best_graph)
    lg_density, _ = kpm_spectral_density(
        L_lg, n_bins=50, n_moments=kpm_moments,
        n_probes=kpm_probes, seed=seed + 99_991,
    )
    lg_gic = _gic(lg_density)
    results["LG"] = {"gic": float(lg_gic), "param": f"σ={sigma:.3f},β={beta:.3f},d={d_lg}",
                     "n": n, "m": int(best_graph.sum() // 2)}
    print(f"  LG  GIC={lg_gic:.4f}  ({_fmt(time.perf_counter() - t0)})")
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
        t0 = time.perf_counter()
        G_m = gen()
        L_m = _sparse_normalised_laplacian(G_m)
        m_density, _ = kpm_spectral_density(
            L_m, n_bins=50, n_moments=kpm_moments,
            n_probes=kpm_probes, seed=seed + hash(name) % 1000,
        )
        gic_val = float(_gic(m_density))
        results[name] = {"gic": gic_val, "param": label,
                         "n": G_m.number_of_nodes(), "m": G_m.number_of_edges()}
        print(f"  {name:<2s}  GIC={gic_val:.4f}  ({_fmt(time.perf_counter() - t0)})")
        del G_m, L_m
        gc.collect()

    # 6. Summary + cache
    summary_rows = [{"model": "Original", "gic": float("nan"), "param": "N/A",
                     "nodes": n, "edges": m, "density": real_density_val}]
    for name, data in results.items():
        summary_rows.append({
            "model": name, "gic": data["gic"], "param": data["param"],
            "nodes": data["n"], "edges": data["m"],
            "density": 2 * data["m"] / (data["n"] * (data["n"] - 1))
            if data["n"] > 1 else 0,
        })
    summary_df = pd.DataFrame(summary_rows)
    _print_summary(summary_df, n, m)

    summary_df.to_csv(out_dir / "summary.csv", index=False)
    cache_payload = {
        "summary_df": summary_df,
        "n": n,
        "m": m,
        "lg_params": {"sigma": sigma, "beta": beta, "d": d_lg,
                      "best_iter": best_iter, "best_L2": best_diff},
        "config_signature": _config_signature(max_iter, check_interval, sample_edges,
                                              kpm_moments, kpm_probes, seed),
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache_payload, f)
    print(f"\nArtifacts: {out_dir}")


def _config_signature(*items):
    return json.dumps(items, sort_keys=True)


def _print_summary(summary_df, n, m):
    import pandas as pd
    print(f"\n=== Summary  (cit-HepTh GCC: n={n:,}, m={m:,}) ===\n")
    print(summary_df.to_string(index=False))
    models_only = summary_df[summary_df["model"] != "Original"].sort_values("gic")
    print("\nRanking by GIC (lower = better):")
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    for rank, (_, row) in enumerate(models_only.iterrows(), 1):
        m_str = medals[rank - 1] if rank <= 4 else f"{rank}."
        print(f"  {m_str} {row['model']:>3s}  GIC = {row['gic']:.4f}")


if __name__ == "__main__":
    main()
