#!/usr/bin/env python3
"""Benchmark Gibbs paths: legacy -> nbrs -> adj-only -> full fast (Numba CSR).
Usage: uv run python scripts/benchmarks/benchmark_gibbs.py [--n 150 --iters 10000 --d 2]."""
from __future__ import annotations

import argparse
import time

import numpy as np

from logit_graph.graph import GraphModel
from logit_graph.lg_features_fast import FastGibbsGraph, run_gibbs_numba
from logit_graph.experiments.sweeps import simulate_graph


def _warmup() -> None:
    rows = [np.zeros(0, dtype=np.int32) for _ in range(20)]
    degs = np.zeros(20, dtype=np.int32)
    draws = np.zeros((10, 3), dtype=np.float64)
    run_gibbs_numba(rows, degs, draws, 1, 2, -3.0, 1.0, 1.0, 2.0, 20)


def _run_legacy(n: int, d: int, n_iter: int, seed: int) -> float:
    gm = GraphModel(
        n=n, d=d, sigma=-3.0, er_p=0.12, layer2=True,
        feature_mode="incremental", seed=seed,
    )
    t0 = time.perf_counter()
    for _ in range(n_iter):
        gm._add_remove_edge_legacy()
    return time.perf_counter() - t0


def _run_nbrs_dense(n: int, d: int, n_iter: int, seed: int) -> float:
    gm = GraphModel(
        n=n, d=d, sigma=-3.0, er_p=0.12, layer2=True,
        feature_mode="incremental", seed=seed,
    )
    t0 = time.perf_counter()
    for _ in range(n_iter):
        gm.add_remove_edge()
    return time.perf_counter() - t0


def _run_adj_only(n: int, d: int, n_iter: int, seed: int) -> float:
    gm = GraphModel(
        n=n, d=d, sigma=-3.0, er_p=0.12, layer2=True,
        feature_mode="incremental", seed=seed,
    )
    t0 = time.perf_counter()
    for _ in range(n_iter):
        gm.add_remove_edge_adj_only()
    gm.materialize_adjacency()
    return time.perf_counter() - t0


def _run_fast_all(n: int, d: int, n_iter: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    gm = GraphModel(
        n=n, d=d, sigma=-3.0, er_p=0.12, layer2=True,
        feature_mode="incremental", seed=seed,
    )
    fg = FastGibbsGraph(
        n, d, gm.sigma, er_p=gm.er_p, rng=rng,
        feature_mode="incremental", alpha=gm.alpha, beta=gm.beta,
        adj=gm.graph.copy(),
    )
    t0 = time.perf_counter()
    fg.run_steps(n_iter, rng)
    _ = fg.to_adjacency()
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--d", type=int, default=1)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    _warmup()

    paths = [
        ("1. Legacy (matrix copy)", _run_legacy),
        ("2. Nbrs + dense sync", _run_nbrs_dense),
        ("3. Nbrs adj-only", _run_adj_only),
        ("4. Fast (Numba fused, skip-L2)", _run_fast_all),
    ]

    print(f"Benchmark n={args.n}, d={args.d}, iters={args.iters}, seed={args.seed}")
    print("-" * 62)

    baseline: float | None = None
    for label, fn in paths:
        t = fn(args.n, args.d, args.iters, args.seed)
        if baseline is None:
            baseline = t
        rel = baseline / t if t > 0 else float("inf")
        print(f"{label:28s} {t:7.3f}s  ({t / args.iters * 1e6:5.1f} µs/step)  {rel:4.1f}x vs legacy")

    print("-" * 62)
    t0 = time.perf_counter()
    simulate_graph(
        args.n, args.d, sigma=-3.0, n_iter=args.iters,
        feature_mode="incremental", seed=args.seed,
    )
    print(f"{'simulate_graph (fast_mode)':28s} {time.perf_counter() - t0:7.3f}s")


if __name__ == "__main__":
    main()
