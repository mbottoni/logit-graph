"""Tests for the Graph Information Criterion (cross-family model selection)."""
import math

import networkx as nx
import numpy as np
import pytest

from logit_graph.gic import GraphInformationCriterion


def _er(n, p, seed):
    return nx.erdos_renyi_graph(n, p, seed=seed)


# -------------------------------------------------------------------
# compute_spectral_density
# -------------------------------------------------------------------

def test_spectral_density_returns_50_bins_in_0_2():
    g = _er(30, 0.2, seed=0)
    gic = GraphInformationCriterion(g, model="ER", p=0.2)
    hist, edges = gic.compute_spectral_density(g)
    assert hist.shape == (50,)
    assert edges.shape == (51,)
    # Bin edges span [0, 2] (range of normalized Laplacian eigenvalues)
    assert math.isclose(edges[0], 0.0)
    assert math.isclose(edges[-1], 2.0)


def test_spectral_density_is_normalized():
    g = _er(40, 0.25, seed=1)
    gic = GraphInformationCriterion(g, model="ER", p=0.25)
    hist, edges = gic.compute_spectral_density(g)
    bin_width = edges[1] - edges[0]
    # density=True ⇒ integrates to 1 over the bin range
    assert math.isclose(hist.sum() * bin_width, 1.0, abs_tol=1e-9)


# -------------------------------------------------------------------
# calculate_spectral_distance — KL vs same graph is near zero
# -------------------------------------------------------------------

def test_kl_distance_to_self_near_zero():
    g = _er(40, 0.2, seed=2)
    gic = GraphInformationCriterion(g, model="ER", p=0.2)
    own_density, _ = gic.compute_spectral_density(g)
    d = gic.calculate_spectral_distance(model_den=own_density)
    assert d < 1e-8


def test_kl_distance_to_very_different_graph_is_large():
    # Sparse ER vs dense complete-ish graph
    g_sparse = _er(40, 0.05, seed=3)
    g_dense = _er(40, 0.9, seed=4)
    gic = GraphInformationCriterion(g_sparse, model="ER", p=0.05)
    dense_density, _ = gic.compute_spectral_density(g_dense)
    sparse_density, _ = gic.compute_spectral_density(g_sparse)
    d_diff = gic.calculate_spectral_distance(model_den=dense_density)
    d_same = gic.calculate_spectral_distance(model_den=sparse_density)
    assert d_diff > d_same


# -------------------------------------------------------------------
# Distance metrics: KL, L1, L2 all non-negative
# -------------------------------------------------------------------

@pytest.mark.parametrize("dist", ["KL", "L1", "L2"])
def test_distance_metrics_non_negative(dist):
    g = _er(30, 0.2, seed=5)
    gic = GraphInformationCriterion(g, model="ER", p=0.2, dist=dist)
    hist, _ = gic.compute_spectral_density(g)
    d = gic.calculate_spectral_distance(model_den=hist)
    assert d >= 0.0


def test_unsupported_distance_raises():
    g = _er(20, 0.2, seed=6)
    gic = GraphInformationCriterion(g, model="ER", p=0.2, dist="cosine")
    with pytest.raises(ValueError):
        gic.calculate_spectral_distance(model_den=np.ones(50) / 50)


# -------------------------------------------------------------------
# calculate_gic — formula and penalty
# -------------------------------------------------------------------

def test_gic_equals_two_dist_plus_two_n_params():
    g = _er(30, 0.2, seed=7)
    gic = GraphInformationCriterion(g, model="ER", p=0.2)
    own_density, _ = gic.compute_spectral_density(g)
    dist = gic.calculate_spectral_distance(model_den=own_density)
    gic_val = gic.calculate_gic(model_den=own_density)
    # ER has 1 parameter (the edge probability)
    assert math.isclose(gic_val, 2.0 * dist + 2.0 * 1)


def test_gic_explicit_n_params_overrides_default():
    g = _er(30, 0.2, seed=8)
    gic = GraphInformationCriterion(g, model="ER", p=0.2)
    own_density, _ = gic.compute_spectral_density(g)
    gic_1 = gic.calculate_gic(model_den=own_density, n_params=1)
    gic_3 = gic.calculate_gic(model_den=own_density, n_params=3)
    # Difference is 2 * (3 - 1) = 4
    assert math.isclose(gic_3 - gic_1, 4.0)


# -------------------------------------------------------------------
# generate_model_graph dispatch
# -------------------------------------------------------------------

def test_generate_model_graph_er():
    g = _er(20, 0.3, seed=9)
    gic = GraphInformationCriterion(g, model="ER", p=0.3)
    out = gic.generate_model_graph()
    assert out.number_of_nodes() == 20


def test_generate_model_graph_unknown_model_raises():
    g = _er(15, 0.2, seed=10)
    gic = GraphInformationCriterion(g, model=123, p=0.2)  # invalid type
    with pytest.raises(ValueError):
        gic.generate_model_graph()
