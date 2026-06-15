"""Shared helpers for AIC-based d selection in the single-graph GIC drivers (run_lg_arxiv/
facebook/twitch_gic.py), so LG's neighbourhood radius d is chosen by the data (AIC on a
logistic-regression fit): sample_pairs -> aic_select_d -> run LG MCMC at best d/sigma/beta."""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np


def compute_d_features(G_gcc, pairs, d: int) -> np.ndarray:
    """For each pair (i, j) return |N^d(i) ∩ N^d(j)|: d=0 → zeros (only σ); d=1 → shared
    1-neighbours; d≥2 → shared d-balls (BFS to depth d, exclusive of the centre)."""
    if d == 0:
        return np.zeros(len(pairs))
    unique = set()
    for i, j in pairs:
        unique.add(i)
        unique.add(j)
    neigh = {}
    for v in unique:
        if d == 1:
            neigh[v] = frozenset(G_gcc.neighbors(v))
        else:
            ball = {v}
            for _ in range(d):
                ball = ball | {nb for u in ball for nb in G_gcc.neighbors(u)}
            ball.discard(v)
            neigh[v] = frozenset(ball)
    return np.array(
        [len(neigh[i] & neigh[j]) for i, j in pairs],
        dtype=float,
    )


def sample_pairs(G_gcc, sample_edges: int, seed: int):
    """Random sample of ``sample_edges`` real edges + matching non-edges."""
    n = G_gcc.number_of_nodes()
    rng = np.random.default_rng(seed)
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
    return pairs, labels


def aic_select_d(G_gcc, pairs, labels, d_candidates: Iterable[int]):
    """Pick the LG neighbourhood radius d by AIC (=2k−2·loglik, k=1 if d=0 else 2) of the σ, β
    logit fit; returns (best_dict, table). For d=0, downstream σ is logit(true density) (not the
    balanced-sample intercept); β is exact (slope invariant under case-control sampling)."""
    import statsmodels.api as sm
    import networkx as nx

    density = nx.density(G_gcc)
    if density <= 0:
        density = 1e-9
    elif density >= 1:
        density = 1 - 1e-9
    sigma_d0_mcmc = float(np.log(density / (1.0 - density)))

    results = []
    for d in d_candidates:
        t0 = time.perf_counter()
        feats = compute_d_features(G_gcc, pairs, d)
        if d == 0:
            X = np.ones((len(pairs), 1))
            result = sm.Logit(labels, X).fit_regularized(method="l1", alpha=0, disp=False)
            sigma_mcmc = sigma_d0_mcmc
            beta = 0.0
            k = 1
        else:
            X = sm.add_constant(feats.reshape(-1, 1))
            result = sm.Logit(labels, X).fit_regularized(method="l1", alpha=0, disp=False)
            sigma_mcmc = float(result.params[0])
            beta = float(result.params[1])
            k = 2
        loglik = float(result.llf)
        aic = 2 * k - 2 * loglik
        results.append({
            "d": d, "sigma": sigma_mcmc, "beta": beta,
            "loglik": loglik, "aic": aic,
            "seconds": time.perf_counter() - t0,
        })
    return min(results, key=lambda r: r["aic"]), results
