"""Deeper coverage of the GraphModel generator + utility methods.

Existing tests/test_graph_model.py covers smoke + convergence integration.
This file targets the constructor, edge probability, spectrum, and core
invariants of generated graphs.
"""
import math

import numpy as np
import networkx as nx
import pytest

from logit_graph.graph import GraphModel


# -------------------------------------------------------------------
# Construction & init
# -------------------------------------------------------------------

def test_default_init_creates_er_graph_at_er_p():
    n, er_p = 80, 0.1
    m = GraphModel(n=n, d=0, sigma=-2.0, er_p=er_p, seed=42)
    # Initial graph density is close to er_p
    density = m.graph.sum() / (n * (n - 1))
    assert abs(density - er_p) < 0.05
    assert m.graph.shape == (n, n)


def test_init_with_explicit_init_graph_uses_it():
    n = 10
    G = nx.path_graph(n)  # path → n-1 edges
    m = GraphModel(n=n, d=1, sigma=-2.0, init_graph=G, seed=0)
    assert int(m.graph.sum() / 2) == n - 1


def test_init_creates_symmetric_no_self_loops():
    m = GraphModel(n=30, d=1, sigma=-2.0, er_p=0.15, seed=1)
    np.testing.assert_array_equal(m.graph, m.graph.T)
    assert np.all(np.diag(m.graph) == 0)


def test_default_feature_mode_is_incremental():
    m = GraphModel(n=10, d=1, sigma=-2.0, seed=0)
    assert m.feature_mode == "incremental"


def test_default_alpha_beta_layer2():
    m = GraphModel(n=10, d=1, sigma=-2.0, seed=0)
    assert m.alpha == 1
    assert m.beta == 1
    assert m.layer2 is True


def test_seed_determinism_initial_graph():
    m1 = GraphModel(n=30, d=1, sigma=-2.0, er_p=0.2, seed=99)
    m2 = GraphModel(n=30, d=1, sigma=-2.0, er_p=0.2, seed=99)
    np.testing.assert_array_equal(m1.graph, m2.graph)


def test_different_seeds_produce_different_graphs():
    m1 = GraphModel(n=30, d=1, sigma=-2.0, er_p=0.2, seed=1)
    m2 = GraphModel(n=30, d=1, sigma=-2.0, er_p=0.2, seed=2)
    assert not np.array_equal(m1.graph, m2.graph)


# -------------------------------------------------------------------
# logistic_regression / edge probability
# -------------------------------------------------------------------

def test_logistic_regression_returns_expit():
    m = GraphModel(n=5, d=0, sigma=-1.0, seed=0)
    # The method just calls expit on its argument
    from scipy.special import expit
    for x in (-5.0, -1.0, 0.0, 1.0, 5.0):
        assert math.isclose(m.logistic_regression(x), float(expit(x)))


def test_logistic_regression_is_in_unit_interval():
    m = GraphModel(n=5, d=0, sigma=-1.0, seed=0)
    for x in (-100.0, -1.0, 0.0, 1.0, 100.0):
        p = m.logistic_regression(x)
        assert 0.0 <= p <= 1.0


# -------------------------------------------------------------------
# generate_empty_graph / generate_small_er_graph
# -------------------------------------------------------------------

def test_generate_empty_graph_has_no_edges():
    m = GraphModel(n=10, d=0, sigma=-2.0, seed=0)
    G = m.generate_empty_graph(15)
    assert G.shape == (15, 15)
    assert G.sum() == 0


def test_generate_small_er_graph_density_close_to_p():
    m = GraphModel(n=10, d=0, sigma=-2.0, er_p=0.05, seed=42)
    # Generate a larger ER directly via the instance method
    G = m.generate_small_er_graph(80, p=0.3)
    density = G.sum() / (80 * 79)
    assert abs(density - 0.3) < 0.06
    # Symmetric, no self-loops
    np.testing.assert_array_equal(G, G.T)
    assert np.all(np.diag(G) == 0)


# -------------------------------------------------------------------
# calculate_spectrum
# -------------------------------------------------------------------

def test_calculate_spectrum_sorted_and_non_negative():
    rng = np.random.default_rng(0)
    n = 12
    upper = rng.random((n, n)) < 0.3
    upper = np.triu(upper, k=1)
    adj = (upper | upper.T).astype(float)
    eig = GraphModel.calculate_spectrum(adj)
    assert eig.shape == (n,)
    assert np.all(np.diff(eig) >= -1e-12)  # sorted ascending
    # Laplacian eigenvalues are non-negative (allow tiny FP noise)
    assert np.all(eig >= -1e-9)


def test_calculate_spectrum_zero_eigenvalue_for_connected_or_disconnected():
    # Any undirected graph's Laplacian has 0 as an eigenvalue with
    # multiplicity = number of connected components.
    n = 8
    # Two disconnected triangles + 2 isolated → 4 components
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)]
    adj = np.zeros((n, n), dtype=float)
    for i, j in edges:
        adj[i, j] = adj[j, i] = 1.0
    eig = GraphModel.calculate_spectrum(adj)
    # 4 components → 4 zero eigenvalues
    near_zero = np.sum(np.abs(eig) < 1e-9)
    assert near_zero == 4
