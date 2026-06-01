"""Stochastic Block Model baseline fitted to an observed graph.

Single source of truth used by both the production codepath
(``model_selection._generate_graph`` and
``simulation.GraphModelComparator._fit_other_models``) and the
single-graph driver scripts in ``notebooks/refactors/``.

The fit follows the recipe in ``notebooks/more_baselines/02_new_baselines_facebook.ipynb``
and the corresponding twitch notebook:

  1. Detect communities on ``G_real`` with Louvain.
  2. Treat the communities as SBM blocks; compute the per-block-pair edge
     probability ``p_ij = actual_edges_between_blocks / possible_pairs``
     (within-block uses ``len(B)·(len(B)-1)/2`` possible pairs).
  3. Generate an SBM sample with ``nx.stochastic_block_model``.

The model has ``k·(k+1)/2`` free probability parameters where ``k`` is the
number of Louvain communities; this is reported as ``n_params`` for the
caller's AIC bookkeeping. The current pipeline scores SBM by spectral
GIC like the other baselines, so the parameter count is informational.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np


def fit_sbm_from_graph(
    G_real: nx.Graph,
    seed: Optional[int] = None,
) -> tuple[list[int], np.ndarray, list[list[int]]]:
    """Louvain communities + per-block-pair edge probabilities.

    Returns ``(sizes, p_matrix, comm_nodes)`` where:
      - ``sizes`` is the list of community sizes in detection order;
      - ``p_matrix`` is a ``k × k`` symmetric matrix of within / between
        block edge probabilities;
      - ``comm_nodes`` is the list of node-index lists per community
        (sorted within each community), useful if the caller needs to
        recover the partition.
    """
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
    """Fit an SBM to ``G_real`` and draw one sample.

    Returns ``(G_sbm, n_params)`` with ``n_params = k·(k+1)/2``.
    The returned graph is a fresh ``nx.Graph`` re-labeled to integer
    nodes ``0..n-1`` so it is directly comparable with other baselines.
    """
    sizes, p_matrix, _ = fit_sbm_from_graph(G_real, seed=seed)
    sbm = nx.stochastic_block_model(sizes, p_matrix.tolist(), seed=seed)
    G_out = nx.Graph()
    G_out.add_nodes_from(range(sum(sizes)))
    G_out.add_edges_from(sbm.edges())
    k = len(sizes)
    return G_out, k * (k + 1) // 2
