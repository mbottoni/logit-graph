"""Cache invariants, adjacency-only path, and convergence helpers for GraphModel.

Init/spectrum/edge-count basics are in test_graph_model{,_deeper}.py. This file
targets the incremental-cache bookkeeping (_sync_edge / add_remove_edge_adj_only
/ materialize_adjacency), _get_sum_degrees_fast correctness, and the
convergence-helper truth tables.
"""
import numpy as np
import networkx as nx
import pytest

from logit_graph.graph import GraphModel
from logit_graph.degrees_counts import get_sum_degrees


def _nbrs_of(adj, i):
    return {int(j) for j in np.nonzero(adj[i])[0]}


def _assert_cache_consistent_with_nbrs(gm):
    """Degrees, edge_count, and neighbor sets agree among themselves."""
    for i in range(gm.n):
        assert gm._degrees[i] == len(gm._nbrs[i])
    assert gm._edge_count == int(sum(gm._degrees) // 2)


def _assert_dense_matches_nbrs(gm):
    for i in range(gm.n):
        assert gm._nbrs[i] == _nbrs_of(gm.graph, i)
    np.testing.assert_array_equal(gm._degrees, gm.graph.sum(axis=1))
    assert gm._edge_count == int(np.triu(gm.graph).sum())


# -------------------------------------------------------------------
# _init_cache
# -------------------------------------------------------------------

def test_init_cache_matches_graph():
    gm = GraphModel(n=20, d=1, sigma=-2.0, er_p=0.2, seed=0)
    _assert_dense_matches_nbrs(gm)


# -------------------------------------------------------------------
# add_remove_edge keeps the dense graph and cache in sync
# -------------------------------------------------------------------

def test_add_remove_edge_keeps_cache_in_sync():
    gm = GraphModel(n=20, d=1, sigma=-2.0, er_p=0.2,
                    feature_mode="incremental", seed=1)
    for _ in range(500):
        gm.add_remove_edge()
    # Dense graph is kept current in the normal path; cache must match it.
    _assert_dense_matches_nbrs(gm)
    _assert_cache_consistent_with_nbrs(gm)
    # Graph stays symmetric / self-loop free.
    np.testing.assert_array_equal(gm.graph, gm.graph.T)
    assert np.all(np.diag(gm.graph) == 0)


# -------------------------------------------------------------------
# add_remove_edge_adj_only: dense graph frozen, cache live, restore flag
# -------------------------------------------------------------------

def test_adj_only_restores_flag_and_keeps_cache_consistent():
    gm = GraphModel(n=18, d=1, sigma=-2.0, er_p=0.2,
                    feature_mode="incremental", seed=2)
    for _ in range(400):
        gm.add_remove_edge_adj_only()
        # The flag is restored after every call.
        assert gm._adj_only is False
    # The neighbor-list cache is internally consistent even though the dense
    # matrix was not synced during adj-only stepping.
    _assert_cache_consistent_with_nbrs(gm)


def test_materialize_adjacency_rebuilds_dense_from_nbrs():
    gm = GraphModel(n=18, d=1, sigma=-2.0, er_p=0.2,
                    feature_mode="incremental", seed=3)
    for _ in range(400):
        gm.add_remove_edge_adj_only()
    gm.materialize_adjacency()
    assert gm._adj_only is False
    # After materialization the dense matrix agrees with the live cache.
    _assert_dense_matches_nbrs(gm)
    np.testing.assert_array_equal(gm.graph, gm.graph.T)


# -------------------------------------------------------------------
# _get_sum_degrees_fast matches the brute-force d-hop sum
# -------------------------------------------------------------------

@pytest.mark.parametrize("d", [0, 1, 2])
def test_get_sum_degrees_fast_matches_reference(d):
    gm = GraphModel(n=16, d=d, sigma=-2.0, er_p=0.3, seed=4)
    for v in range(gm.n):
        expected = get_sum_degrees(gm.graph, v, d)
        assert gm._get_sum_degrees_fast(v) == pytest.approx(expected)


# -------------------------------------------------------------------
# _has_edge in both modes
# -------------------------------------------------------------------

def test_has_edge_dense_and_adj_only_modes_agree():
    gm = GraphModel(n=12, d=1, sigma=-2.0, er_p=0.3, seed=5)
    for i in range(gm.n):
        for j in range(i + 1, gm.n):
            dense = gm._has_edge(i, j)
            gm._adj_only = True
            membership = gm._has_edge(i, j)
            gm._adj_only = False
            assert dense == membership


# -------------------------------------------------------------------
# d > 2 on a small graph does not error and keeps cache consistent
# -------------------------------------------------------------------

def test_high_d_on_small_graph_runs():
    gm = GraphModel(n=8, d=3, sigma=-1.5, er_p=0.4,
                    feature_mode="incremental", seed=6)
    for _ in range(200):
        gm.add_remove_edge()
    _assert_dense_matches_nbrs(gm)


# -------------------------------------------------------------------
# Convergence helper truth tables
# -------------------------------------------------------------------

def _gm():
    return GraphModel(n=6, d=0, sigma=-2.0, er_p=0.2, seed=7)


def test_edge_convergence_true_when_stable():
    g = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.3, seed=0))
    graphs = [g.copy() for _ in range(5)]
    assert _gm().check_convergence_number_of_edges(graphs, threshold_edges=0, stability_window=5)


def test_edge_convergence_false_on_jump():
    sparse = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.1, seed=0))
    dense = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.9, seed=1))
    graphs = [sparse, sparse, dense]  # big edge jump in the window
    assert not _gm().check_convergence_number_of_edges(graphs, threshold_edges=2, stability_window=3)


def test_spectrum_convergence_true_for_identical_graphs():
    g = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.3, seed=2))
    graphs = [g.copy() for _ in range(4)]
    assert _gm().check_convergence_spectrum(graphs, threshold_spectrum=1e-9, stability_window=4)


def test_spectrum_convergence_false_for_changing_graphs():
    g1 = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.2, seed=3))
    g2 = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.8, seed=4))
    graphs = [g1, g2]
    assert not _gm().check_convergence_spectrum(graphs, threshold_spectrum=0.1, stability_window=2)
