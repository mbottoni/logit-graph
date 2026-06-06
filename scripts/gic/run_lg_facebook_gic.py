#!/usr/bin/env python3
"""Fit LG + ER/WS/BA on the FULL MUSAE Facebook page-page network and rank by GIC.

Mirrors ``run_arxiv_gic.py`` but on the page-page Facebook graph shipped
in ``data/facebook_large/musae_facebook_edges.csv`` (n≈22,470, m≈171k).
No node subsampling — whole graph fitted via:
  - KPM spectral density on the sparse normalised Laplacian
  - sampled logistic regression for σ, β (~50k edges + 50k non-edges)
  - dense LG MCMC at d=0 (one adjacency in RAM, ~4 GB)
  - sparse baseline generation + KPM scoring

Env-var overrides:
  LG_FB_DATA          path to musae_facebook_edges.csv
  LG_FB_MAX_ITER      LG MCMC iterations         (default 50_000)
  LG_FB_CHECK         spectral check interval    (default 5_000)
  LG_FB_WARM_UP       iters before patience      (default 10_000)
  LG_FB_PATIENCE      no-improvement checks      (default 100)
  LG_FB_SAMPLE_EDGES  σ/β estimation sample      (default 50_000)
  LG_FB_KPM_MOMENTS   Chebyshev moments          (default 150)
  LG_FB_KPM_PROBES    random probes              (default 40)
  LG_FB_USE_CACHE     reuse cached results       (default 1)
  LG_FB_SEED          RNG seed                   (default 42)
  LG_FB_QUICK         smoke (MAX_ITER=10k, fewer KPM moments)

  make lg-gic-facebook        full preset (~2-3 min)
  make lg-gic-facebook-quick  smoke (~30-60s)
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

    quick = os.environ.get("LG_FB_QUICK", "0") == "1"
    data_path = Path(os.environ.get(
        "LG_FB_DATA",
        str(_repo_root / "data" / "facebook_large" / "musae_facebook_edges.csv"),
    ))
    max_iter = _get_int("LG_FB_MAX_ITER", 10_000 if quick else 50_000)
    check_interval = _get_int("LG_FB_CHECK", 1_000 if quick else 5_000)
    warm_up = _get_int("LG_FB_WARM_UP", 2_000 if quick else 10_000)
    patience = _get_int("LG_FB_PATIENCE", 30 if quick else 100)
    sample_edges = _get_int("LG_FB_SAMPLE_EDGES", 20_000 if quick else 50_000)
    kpm_moments = _get_int("LG_FB_KPM_MOMENTS", 80 if quick else 150)
    kpm_probes = _get_int("LG_FB_KPM_PROBES", 20 if quick else 40)
    seed = _get_int("LG_FB_SEED", 42)
    use_cache = os.environ.get("LG_FB_USE_CACHE", "1") == "1"

    out_dir = _here / "runs" / "facebook"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "facebook_full_results.pkl"

    print(
        f"MUSAE Facebook full-graph GIC ranking  "
        f"max_iter={max_iter:,}  check={check_interval:,}  patience={patience}  "
        f"sample_edges={sample_edges:,}  kpm_moments={kpm_moments}  "
        f"kpm_probes={kpm_probes}  quick={quick}"
    )

    cfg_sig = json.dumps(
        [max_iter, check_interval, sample_edges, kpm_moments, kpm_probes, seed],
        sort_keys=True,
    )
    if use_cache and cache_path.is_file():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("config_signature") == cfg_sig:
            print(f"\nCache hit at {cache_path}; reprinting summary.")
            _print_summary(cached["summary_df"], cached["n"], cached["m"])
            return
        print("Cache present but config changed; recomputing.")

    # 1. Load + GCC
    print(f"\n=== 1. Load full graph from {data_path}")
    t0 = time.perf_counter()
    df = pd.read_csv(data_path)
    G_full = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])
    G_full.remove_edges_from(nx.selfloop_edges(G_full))
    gcc_nodes = max(nx.connected_components(G_full), key=len)
    G_gcc = nx.convert_node_labels_to_integers(G_full.subgraph(gcc_nodes).copy())
    n = G_gcc.number_of_nodes()
    m = G_gcc.number_of_edges()
    del df, G_full
    gc.collect()
    print(f"  GCC: n={n:,}  m={m:,}  density={nx.density(G_gcc):.6f}  "
          f"[{_fmt(time.perf_counter() - t0)}]")

    # 2. KPM density of the real graph
    print(f"\n=== 2. KPM spectral density of real graph")
    t0 = time.perf_counter()
    L_real = _sparse_normalised_laplacian(G_gcc)
    real_density, _ = kpm_spectral_density(
        L_real, n_bins=50, n_moments=kpm_moments,
        n_probes=kpm_probes, seed=seed,
    )
    print(f"  KPM done in {_fmt(time.perf_counter() - t0)}")

    # 3. AIC-select d ∈ {0, 1, 2} from sampled (edge, non-edge) logit fits
    print(f"\n=== 3. AIC-select d from σ, β logit on "
          f"{sample_edges:,} edges + {sample_edges:,} non-edges")
    from lg_aic_utils import aic_select_d, sample_pairs as _sample_pairs
    d_candidates = [int(x) for x in
                    os.environ.get("LG_FB_D_CANDIDATES", "0,1,2").split(",")
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

    # 4. LG MCMC fit
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
            print(
                f"  iter {i:>7,}/{max_iter:,}  L2={diff:.4f}  best={best_diff:.4f}  "
                f"edges={gm._edge_count:,}/{m:,}  pat={no_improve}/{patience}  "
                f"wall={_fmt(time.perf_counter() - t_start)}"
            )
            if i >= warm_up and no_improve >= patience:
                print(f"  converged (no improvement for {patience} checks)")
                break

    print(f"  LG fit done in {_fmt(time.perf_counter() - t_start)}  "
          f"best_iter={best_iter:,}  best_L2={best_diff:.4f}")

    # 5. GIC for LG + ER / WS / BA
    print(f"\n=== 5. GIC for LG + ER / WS / BA")
    results: dict[str, dict] = {}
    real_density_val = nx.density(G_gcc)
    real_avg_deg = 2 * m / n
    eps = 1e-10

    def _gic(p, n_params):
        # GIC = 2*spectral_distance + 2*n_params (AIC-style complexity penalty);
        # spectral_distance is the symmetric KL between the real and model ESDs.
        from scipy.stats import entropy as _entropy
        dist = 0.5 * (_entropy(real_density + eps, p + eps)
                      + _entropy(p + eps, real_density + eps))
        return 2.0 * dist + 2.0 * n_params

    t0 = time.perf_counter()
    L_lg = _sparse_normalised_laplacian(best_graph)
    lg_density, _ = kpm_spectral_density(
        L_lg, n_bins=50, n_moments=kpm_moments,
        n_probes=kpm_probes, seed=seed + 99_991,
    )
    lg_gic = float(_gic(lg_density, n_params=1))  # LG penalized on sigma only
    results["LG"] = {"gic": lg_gic, "param": f"σ={sigma:.3f},β={beta:.3f},d={d_lg}",
                     "n": n, "m": int(best_graph.sum() // 2)}
    print(f"  LG  GIC={lg_gic:.4f}  ({_fmt(time.perf_counter() - t0)})")
    del gm
    gc.collect()

    from logit_graph.sbm import generate_sbm_from_real, fit_sbm_from_graph
    # Free-parameter counts for the GIC penalty: ER/BA = 1 scalar, WS = 2 (k, p),
    # SBM = k(k+1)/2 block-edge probabilities (k = number of Louvain communities).
    _sbm_sizes, _, _ = fit_sbm_from_graph(G_gcc, seed=seed)
    _sbm_k = len(_sbm_sizes)
    sbm_n_params = _sbm_k * (_sbm_k + 1) // 2
    baselines = [
        ("ER", "Erdos-Renyi", 1,
         lambda: nx.erdos_renyi_graph(n, real_density_val, seed=seed)),
        ("WS", "Watts-Strogatz", 2,
         lambda: nx.watts_strogatz_graph(n, max(2, int(round(real_avg_deg))), 0.1, seed=seed)),
        ("BA", "Barabasi-Albert", 1,
         lambda: nx.barabasi_albert_graph(n, max(1, int(round(real_avg_deg / 2))), seed=seed)),
        ("SBM", f"Louvain SBM (k={_sbm_k})", sbm_n_params,
         lambda: generate_sbm_from_real(G_gcc, seed=seed)[0]),
    ]
    for name, label, n_params, gen in baselines:
        t0 = time.perf_counter()
        G_m = gen()
        L_m = _sparse_normalised_laplacian(G_m)
        m_density, _ = kpm_spectral_density(
            L_m, n_bins=50, n_moments=kpm_moments,
            n_probes=kpm_probes, seed=seed + hash(name) % 1000,
        )
        gic_val = float(_gic(m_density, n_params))
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
    with open(cache_path, "wb") as f:
        pickle.dump({
            "summary_df": summary_df,
            "n": n, "m": m,
            "lg_params": {"sigma": sigma, "beta": beta, "d": d_lg,
                          "best_iter": best_iter, "best_L2": best_diff},
            "config_signature": cfg_sig,
        }, f)
    print(f"\nArtifacts: {out_dir}")


def _print_summary(summary_df, n, m):
    print(f"\n=== Summary  (MUSAE Facebook GCC: n={n:,}, m={m:,}) ===\n")
    print(summary_df.to_string(index=False))
    models_only = summary_df[summary_df["model"] != "Original"].sort_values("gic")
    print("\nRanking by GIC (lower = better):")
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    for rank, (_, row) in enumerate(models_only.iterrows(), 1):
        m_str = medals[rank - 1] if rank <= 4 else f"{rank}."
        print(f"  {m_str} {row['model']:>3s}  GIC = {row['gic']:.4f}")


if __name__ == "__main__":
    main()
