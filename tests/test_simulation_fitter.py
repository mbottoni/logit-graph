"""Tests for LogitGraphFitter and GraphModelComparator._get_logit_graph_for_d.

The simulation helpers (clean_and_convert_param, _warm_start_er_p,
_direct_er_at_sigma, estimate_sigma_only/many, calculate_graph_attributes) are
covered in test_simulation_api.py. The two fitter classes had no tests; this
file exercises the d=0 fast path, directed/isolated-node handling, the metadata
contract, one small d=1 Gibbs run, and seeded determinism.
"""
import numpy as np
import networkx as nx
import pytest

from logit_graph.simulation import LogitGraphFitter, GraphModelComparator


def _er_graph(n, p, seed):
    return nx.erdos_renyi_graph(n, p, seed=seed)


# -------------------------------------------------------------------
# LogitGraphFitter.fit — d=0 fast path
# -------------------------------------------------------------------

def test_fit_d0_succeeds_and_populates_metadata():
    g = _er_graph(30, 0.2, seed=0)
    fitter = LogitGraphFitter(d=0, verbose=False)
    out = fitter.fit(g)

    assert out is fitter  # fit returns self (sklearn-style)
    assert fitter.metadata["fit_success"] is True
    assert isinstance(fitter.fitted_graph, nx.Graph)
    assert fitter.fitted_graph.number_of_nodes() == g.number_of_nodes()

    for key in ("sigma", "gic_value", "best_iteration", "fitted_nodes",
                "fitted_edges", "spectrum_diffs", "edge_diffs", "gic_values"):
        assert key in fitter.metadata
    assert np.isfinite(fitter.metadata["sigma"])
    assert np.isfinite(fitter.metadata["gic_value"])


def test_fit_records_original_node_and_edge_counts():
    g = _er_graph(25, 0.25, seed=1)
    fitter = LogitGraphFitter(d=0, verbose=False).fit(g)
    assert fitter.metadata["original_nodes"] == g.number_of_nodes()
    assert fitter.metadata["original_edges"] == g.number_of_edges()


# -------------------------------------------------------------------
# Directed input is converted to undirected
# -------------------------------------------------------------------

def test_fit_accepts_directed_graph():
    dg = nx.gnp_random_graph(25, 0.2, seed=2, directed=True)
    fitter = LogitGraphFitter(d=0, verbose=False).fit(dg)
    assert fitter.metadata["fit_success"] is True
    assert fitter.fitted_graph.number_of_nodes() == dg.number_of_nodes()
    # original_edges is counted on the undirected view (≤ directed edge count).
    assert fitter.metadata["original_edges"] == dg.to_undirected().number_of_edges()


# -------------------------------------------------------------------
# Isolated nodes are preserved
# -------------------------------------------------------------------

def test_fit_preserves_isolated_nodes():
    g = nx.Graph()
    g.add_nodes_from(range(15))
    # edges only among the first 10 nodes; nodes 10..14 are isolated
    g.add_edges_from(nx.erdos_renyi_graph(10, 0.4, seed=3).edges())
    fitter = LogitGraphFitter(d=0, verbose=False).fit(g)
    assert fitter.metadata["fit_success"] is True
    assert fitter.fitted_graph.number_of_nodes() == 15


# -------------------------------------------------------------------
# d>=1 Gibbs path runs end-to-end on a small graph
# -------------------------------------------------------------------

def test_fit_d1_small_runs_to_completion():
    g = _er_graph(20, 0.25, seed=4)
    fitter = LogitGraphFitter(
        d=1, n_iteration=300, warm_up=50, patience=3,
        check_interval=50, min_gic_threshold=100.0, verbose=False,
    )
    fitter.fit(g)
    assert fitter.metadata["fit_success"] is True
    assert fitter.fitted_graph.number_of_nodes() == 20
    assert np.isfinite(fitter.metadata["gic_value"])


# -------------------------------------------------------------------
# GraphModelComparator._get_logit_graph_for_d (d=0 fast path)
# -------------------------------------------------------------------

def _comparator(random_state=42):
    return GraphModelComparator(
        d_list=[0], lg_params={}, verbose=False, random_state=random_state,
    )


def test_get_logit_graph_for_d0_returns_expected_shape():
    adj = nx.to_numpy_array(_er_graph(25, 0.2, seed=5))
    cmp = _comparator()
    out = cmp._get_logit_graph_for_d(adj, 0)
    best_arr, sigma, gic_value, best_iter, gic_values, spec_diffs, edge_diffs = out
    assert best_arr.shape == (25, 25)
    np.testing.assert_array_equal(best_arr, best_arr.T)  # symmetric sample
    assert np.isfinite(sigma)
    assert np.isfinite(gic_value)
    assert best_iter == 0
    assert gic_values == [gic_value]


def test_get_logit_graph_for_d0_is_deterministic_under_random_state():
    adj = nx.to_numpy_array(_er_graph(25, 0.2, seed=6))
    a = _comparator(random_state=7)._get_logit_graph_for_d(adj, 0)
    b = _comparator(random_state=7)._get_logit_graph_for_d(adj, 0)
    # Same seed ⇒ identical sampled graph and sigma.
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]
