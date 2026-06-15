"""Stochastic Block Model baseline fitted to an observed graph: detect Louvain
communities as blocks, set per-block-pair edge probabilities from observed counts, then
draw via ``nx.stochastic_block_model``. Has ``k·(k+1)/2`` params (reported for AIC)."""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np


def fit_sbm_from_graph(
    G_real: nx.Graph,
    seed: Optional[int] = None,
) -> tuple[list[int], np.ndarray, list[list[int]]]:
    """Louvain communities + per-block-pair edge probabilities. Returns
    ``(sizes, p_matrix, comm_nodes)``: community sizes in detection order, the k×k
    symmetric within/between edge-probability matrix, and sorted node lists per community."""
    comms = list(nx.community.louvain_communities(G_real, seed=seed))
    sizes = [len(c) for c in comms]
    k = len(comms)

    comm_nodes = [sorted(c) for c in comms]
    p_matrix = np.zeros((k, k))
    for ci in range(k):
        for cj in range(ci, k):
            ni, nj = comm_nodes[ci], comm_nodes[cj]
            if ci == cj:
                possible = len(ni) * (len(ni) - 1) / 2
                actual = sum(
                    1 for u in ni for v in ni
                    if u < v and G_real.has_edge(u, v)
                )
            else:
                possible = len(ni) * len(nj)
                actual = sum(
                    1 for u in ni for v in nj if G_real.has_edge(u, v)
                )
            p = actual / possible if possible > 0 else 0.0
            p_matrix[ci, cj] = p
            p_matrix[cj, ci] = p
    return sizes, p_matrix, comm_nodes


def generate_sbm_from_real(
    G_real: nx.Graph,
    seed: Optional[int] = None,
) -> tuple[nx.Graph, int]:
    """Fit an SBM to ``G_real`` and draw one sample. Returns ``(G_sbm, n_params)`` with
    ``n_params = k·(k+1)/2``; the graph is a fresh ``nx.Graph`` re-labeled to nodes
    ``0..n-1`` so it is directly comparable with the other baselines."""
    sizes, p_matrix, _ = fit_sbm_from_graph(G_real, seed=seed)
    sbm = nx.stochastic_block_model(sizes, p_matrix.tolist(), seed=seed)
    G_out = nx.Graph()
    G_out.add_nodes_from(range(sum(sizes)))
    G_out.add_edges_from(sbm.edges())
    k = len(sizes)
    return G_out, k * (k + 1) // 2
