"""Unit tests for the SBM baseline (`logit_graph.sbm`).

Covers the two public functions used by the GIC model-ranking pipeline:
``fit_sbm_from_graph`` (Louvain partition + per-block-pair edge probabilities)
and ``generate_sbm_from_real`` (fit + draw one sample, with parameter count).
"""
import networkx as nx
import numpy as np
import pytest

from logit_graph.sbm import fit_sbm_from_graph, generate_sbm_from_real

SEED = 42


def _planted_two_block(seed=SEED):
    """Two dense blocks (size 15 each) with very few cross edges."""
    rng = np.random.default_rng(seed)
    g = nx.Graph()
    g.add_nodes_from(range(30))
    block_a, block_b = range(0, 15), range(15, 30)
    for block in (block_a, block_b):
        for u in block:
            for v in block:
                if u < v and rng.random() < 0.8:
                    g.add_edge(u, v)
    for u in block_a:
        for v in block_b:
            if rng.random() < 0.01:
                g.add_edge(u, v)
    return g


# -------------------------------------------------------------------
# fit_sbm_from_graph — structural invariants
# -------------------------------------------------------------------

def test_fit_partition_covers_all_nodes_exactly_once():
    g = _planted_two_block()
    sizes, p_matrix, comm_nodes = fit_sbm_from_graph(g, seed=SEED)

    assert sum(sizes) == g.number_of_nodes()
    # comm_nodes is a partition: disjoint and covers every node.
    all_nodes = [u for comm in comm_nodes for u in comm]
    assert sorted(all_nodes) == sorted(g.nodes())
    assert len(all_nodes) == len(set(all_nodes))


def test_fit_p_matrix_is_square_symmetric_and_valid_probabilities():
    g = _planted_two_block()
    sizes, p_matrix, comm_nodes = fit_sbm_from_graph(g, seed=SEED)
    k = len(sizes)

    assert p_matrix.shape == (k, k)
    assert np.allclose(p_matrix, p_matrix.T)
    assert np.all(np.isfinite(p_matrix))
    assert np.all(p_matrix >= 0.0)
    assert np.all(p_matrix <= 1.0)


def test_fit_sizes_match_comm_nodes_lengths():
    g = _planted_two_block()
    sizes, _, comm_nodes = fit_sbm_from_graph(g, seed=SEED)
    assert sizes == [len(c) for c in comm_nodes]


# -------------------------------------------------------------------
# fit_sbm_from_graph — recovers planted assortative structure
# -------------------------------------------------------------------

def test_fit_recovers_assortative_structure():
    g = _planted_two_block()
    sizes, p_matrix, _ = fit_sbm_from_graph(g, seed=SEED)
    k = len(sizes)

    assert k >= 2  # the two dense blocks are detected as separate communities

    diag = np.diag(p_matrix)
    off = p_matrix[~np.eye(k, dtype=bool)]
    # Within-block density should dominate between-block density.
    assert diag.mean() > off.mean()


# -------------------------------------------------------------------
# fit_sbm_from_graph — determinism
# -------------------------------------------------------------------

def test_fit_is_deterministic_under_fixed_seed():
    g = _planted_two_block()
    s1, p1, c1 = fit_sbm_from_graph(g, seed=SEED)
    s2, p2, c2 = fit_sbm_from_graph(g, seed=SEED)
    assert s1 == s2
    assert c1 == c2
    np.testing.assert_array_equal(p1, p2)


# -------------------------------------------------------------------
# fit_sbm_from_graph — degenerate graphs
# -------------------------------------------------------------------

def test_fit_complete_graph_within_block_probability_is_one():
    g = nx.complete_graph(12)
    sizes, p_matrix, _ = fit_sbm_from_graph(g, seed=SEED)
    # A clique is one community; its within-block edge probability is 1.
    assert sum(sizes) == 12
    assert np.all(np.isfinite(p_matrix))
    assert np.isclose(np.diag(p_matrix).max(), 1.0)


def test_fit_singleton_community_does_not_divide_by_zero():
    # An isolated node forms a size-1 community ⇒ possible pairs = 0.
    g = nx.complete_graph(10)
    g.add_node(10)  # isolated
    sizes, p_matrix, _ = fit_sbm_from_graph(g, seed=SEED)
    assert sum(sizes) == 11
    # No NaN/inf from the possible==0 guard.
    assert np.all(np.isfinite(p_matrix))


# -------------------------------------------------------------------
# generate_sbm_from_real
# -------------------------------------------------------------------

def test_generate_returns_graph_with_relabeled_nodes_and_param_count():
    g = _planted_two_block()
    sizes, _, _ = fit_sbm_from_graph(g, seed=SEED)
    k = len(sizes)

    g_sbm, n_params = generate_sbm_from_real(g, seed=SEED)

    assert isinstance(g_sbm, nx.Graph)
    assert g_sbm.number_of_nodes() == g.number_of_nodes()
    # Relabeled to contiguous integers 0..n-1.
    assert set(g_sbm.nodes()) == set(range(g.number_of_nodes()))
    # k(k+1)/2 free probability parameters.
    assert n_params == k * (k + 1) // 2


def test_generate_is_deterministic_under_fixed_seed():
    g = _planted_two_block()
    g1, n1 = generate_sbm_from_real(g, seed=SEED)
    g2, n2 = generate_sbm_from_real(g, seed=SEED)
    assert n1 == n2
    assert set(g1.edges()) == set(g2.edges())


def test_generate_param_count_one_for_single_community():
    # A clique collapses to a single block ⇒ k=1 ⇒ n_params = 1.
    g = nx.complete_graph(12)
    _, n_params = generate_sbm_from_real(g, seed=SEED)
    assert n_params == 1
