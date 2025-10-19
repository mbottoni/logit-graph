import numpy as np
import networkx as nx

from logit_graph.graph import GraphModel


def test_generate_small_er_graph_returns_symmetric_adj():
    gm = GraphModel(n=10, d=1, sigma=0.0, er_p=0.2)
    adj = gm.generate_small_er_graph(10, 0.2)
    assert isinstance(adj, np.ndarray)
    assert adj.shape == (10, 10)
    assert np.allclose(adj, adj.T)


def test_calculate_spectrum_sorted_nonnegative():
    gm = GraphModel(n=8, d=1, sigma=0.0, er_p=0.2)
    spec = gm.calculate_spectrum(gm.graph)
    assert (np.diff(spec) >= -1e-12).all()  # sorted non-decreasing
    assert (spec >= -1e-9).all()


def test_add_remove_edge_changes_edge_count():
    gm = GraphModel(n=12, d=1, sigma=0.0, er_p=0.1)
    before = np.sum(np.triu(gm.graph))
    for _ in range(10):
        gm.add_remove_edge()
    after = np.sum(np.triu(gm.graph))
    # Probabilistic, but across 10 steps should change at least sometimes
    assert before != after or np.sum(gm.graph) in [0, gm.graph.size]


def test_convergence_checks_do_not_crash():
    gm = GraphModel(n=8, d=1, sigma=0.0, er_p=0.2)
    graphs = [gm.graph.copy()]
    for _ in range(5):
        gm.add_remove_edge()
        graphs.append(gm.graph.copy())

    _ = gm.check_convergence_number_of_edges(graphs, threshold_edges=100, stability_window=3)
    _ = gm.check_convergence_spectrum(graphs, threshold_spectrum=1e6, stability_window=3)


