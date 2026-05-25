"""Neighborhood features for the logistic random graph model.

Supports Layer-2 conditioning (compute features on the graph with pair
(i, j) removed) and several feature modes used in paper experiments.
"""
from __future__ import annotations

import math
from typing import Literal, Optional, Union

import numpy as np
import networkx as nx

from .degrees_counts import get_sum_degrees

FeatureMode = Literal["paper_raw", "bounded", "incremental", "common_dhop"]

ALPHA_GWESP_DEFAULT = 2.0

FEATURE_MODES: tuple[str, ...] = (
    "paper_raw",
    "bounded",
    "incremental",
    "common_dhop",
)


def recommended_iterations(n: int, cap: Optional[int] = None) -> int:
    """Gibbs steps so each edge is resampled O(1) times."""
    base = max(20_000, int(5 * n * (n - 1)))
    if cap is not None:
        return min(base, cap)
    return base


def sum_degree(graph: np.ndarray, vertex: int, d: int) -> float:
    """Sum of degrees in the d-hop ball of ``vertex`` (paper S_i^{(d)})."""
    return float(get_sum_degrees(graph, vertex=vertex, d=d))


def _ball_vertices(graph: np.ndarray, v: int, d: int) -> set[int]:
    if d == 0:
        return {v}
    visited: set[int] = {v}
    current = [v]
    for _ in range(d):
        nxt: list[int] = []
        for u in current:
            for nv in np.nonzero(graph[u])[0]:
                nv = int(nv)
                if nv not in visited:
                    visited.add(nv)
                    nxt.append(nv)
        current = nxt
        if not current:
            break
    return visited


def common_dhop_count(graph: np.ndarray, i: int, j: int, d: int) -> int:
    """|B_d(i) cap B_d(j) \\ {i,j}|."""
    if d == 0:
        return 0
    bi = _ball_vertices(graph, i, d)
    bj = _ball_vertices(graph, j, d)
    inter = bi & bj
    inter.discard(i)
    inter.discard(j)
    return len(inter)


def incremental_h(
    graph: np.ndarray,
    i: int,
    j: int,
    d: int,
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
) -> float:
    if d == 0:
        return 0.0
    if d == 1:
        c = common_dhop_count(graph, i, j, 1)
        if c <= 0:
            return 0.0
        return alpha_gwesp * (1.0 - (1.0 - 1.0 / alpha_gwesp) ** c)
    delta = common_dhop_count(graph, i, j, d) - common_dhop_count(graph, i, j, d - 1)
    return math.log(1.0 + max(0, delta))


def pair_feature(
    graph: np.ndarray,
    i: int,
    j: int,
    d: int,
    mode: FeatureMode = "bounded",
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
) -> float:
    """Pair feature x_ij used as offset (beta fixed at 1 in paper estimator)."""
    if mode == "paper_raw":
        return sum_degree(graph, i, d) + sum_degree(graph, j, d)
    if mode == "bounded":
        si = sum_degree(graph, i, d)
        sj = sum_degree(graph, j, d)
        return math.log(1.0 + si) + math.log(1.0 + sj)
    if mode == "incremental":
        return incremental_h(graph, i, j, d, alpha_gwesp=alpha_gwesp)
    if mode == "common_dhop":
        return math.log(1.0 + common_dhop_count(graph, i, j, d))
    raise ValueError(f"Unknown feature mode: {mode!r}")


def _adj_from_input(graph: Union[np.ndarray, nx.Graph]) -> np.ndarray:
    if isinstance(graph, nx.Graph):
        return nx.to_numpy_array(graph)
    return np.asarray(graph, dtype=float)


def pair_feature_layer2(
    graph: Union[np.ndarray, nx.Graph],
    i: int,
    j: int,
    d: int,
    mode: FeatureMode = "bounded",
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
) -> float:
    """Layer-2 feature: computed on the graph with edge (i,j) removed."""
    adj = _adj_from_input(graph).copy()
    had = adj[i, j] > 0
    if had:
        adj[i, j] = adj[j, i] = 0.0
    return pair_feature(adj, i, j, d, mode=mode, alpha_gwesp=alpha_gwesp)


def precompute_vertex_sums(graph: np.ndarray, d: int) -> np.ndarray:
    """S_v^{(d)} for all vertices — reused for paper_raw / bounded modes."""
    n = graph.shape[0]
    out = np.zeros(n, dtype=float)
    for v in range(n):
        out[v] = sum_degree(graph, v, d)
    return out


def build_pair_dataset(
    graph: Union[np.ndarray, nx.Graph],
    d: int,
    mode: FeatureMode = "bounded",
    layer2: bool = True,
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
    max_pairs: Optional[int] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build offset vector and binary labels for all upper-triangle pairs.

    Returns
    -------
    offsets : (m,) feature values (used as fixed beta=1 offset)
    labels : (m,) 0/1 edge indicators
    """
    adj = _adj_from_input(graph)
    n = adj.shape[0]
    rng = np.random.default_rng(seed)

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))

    if max_pairs is not None and len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(k)] for k in idx]

    use_vertex_cache = layer2 and mode in ("paper_raw", "bounded")
    vertex_sums: Optional[np.ndarray] = None
    if use_vertex_cache:
        vertex_sums = precompute_vertex_sums(adj, d)

    offsets = np.empty(len(pairs), dtype=float)
    labels = np.empty(len(pairs), dtype=int)

    for k, (i, j) in enumerate(pairs):
        labels[k] = 1 if adj[i, j] > 0 else 0
        if layer2:
            if use_vertex_cache and vertex_sums is not None:
                adj_l2 = adj
                si = float(vertex_sums[i])
                sj = float(vertex_sums[j])
                if adj[i, j] > 0:
                    si -= 1.0
                    sj -= 1.0
                if mode == "paper_raw":
                    offsets[k] = si + sj
                else:
                    offsets[k] = math.log(1.0 + si) + math.log(1.0 + sj)
            else:
                offsets[k] = pair_feature_layer2(
                    adj, i, j, d, mode=mode, alpha_gwesp=alpha_gwesp,
                )
        else:
            offsets[k] = pair_feature(
                adj, i, j, d, mode=mode, alpha_gwesp=alpha_gwesp,
            )

    return offsets, labels
