"""High-level public API: helpers + the Fitter/Simulation/Comparator classes."""
import math

import numpy as np
import networkx as nx
import pytest

from logit_graph.simulation import (
    _direct_er_at_sigma,
    _warm_start_er_p,
    calculate_graph_attributes,
    clean_and_convert_param,
    estimate_sigma_many,
    estimate_sigma_only,
)


def _er_graph(n, p, seed):
    return nx.erdos_renyi_graph(n, p, seed=seed)


# -------------------------------------------------------------------
# clean_and_convert_param
# -------------------------------------------------------------------

def test_clean_and_convert_int_returns_same():
    assert clean_and_convert_param(3) == 3


def test_clean_and_convert_float_returns_same():
    assert clean_and_convert_param(0.42) == 0.42


def test_clean_and_convert_string_extracts_numeric():
    assert clean_and_convert_param("p=0.25") == 0.25


def test_clean_and_convert_invalid_returns_nan():
    out = clean_and_convert_param("not a number")
    assert math.isnan(out)


# -------------------------------------------------------------------
# _warm_start_er_p — clamped to [lo, hi]
# -------------------------------------------------------------------

def test_warm_start_er_p_clamps_below():
    # expit(-10) ≈ 4.5e-5, clamped to default lo=0.02
    assert _warm_start_er_p(-10.0) == pytest.approx(0.02)


def test_warm_start_er_p_clamps_above():
    # expit(5) ≈ 0.993, clamped to default hi=0.5
    assert _warm_start_er_p(5.0) == pytest.approx(0.5)


def test_warm_start_er_p_returns_expit_in_range():
    # expit(-2) ≈ 0.119, within [0.02, 0.5]
    p = _warm_start_er_p(-2.0)
    assert 0.02 <= p <= 0.5
    # Should match expit value
    from scipy.special import expit
    assert math.isclose(p, float(expit(-2.0)))


# -------------------------------------------------------------------
# _direct_er_at_sigma
# -------------------------------------------------------------------

def test_direct_er_at_sigma_returns_symmetric_zero_diagonal():
    adj = _direct_er_at_sigma(n=20, sigma=-1.0, seed=0)
    assert adj.shape == (20, 20)
    np.testing.assert_array_equal(adj, adj.T)
    assert np.all(np.diag(adj) == 0)


def test_direct_er_at_sigma_density_close_to_expit():
    n, sigma = 200, -2.0
    adj = _direct_er_at_sigma(n=n, sigma=sigma, seed=42)
    from scipy.special import expit
    density = adj.sum() / (n * (n - 1))
    assert abs(density - float(expit(sigma))) < 0.02


def test_direct_er_at_sigma_deterministic_with_seed():
    a1 = _direct_er_at_sigma(n=30, sigma=-1.5, seed=7)
    a2 = _direct_er_at_sigma(n=30, sigma=-1.5, seed=7)
    np.testing.assert_array_equal(a1, a2)


# -------------------------------------------------------------------
# estimate_sigma_only / estimate_sigma_many
# -------------------------------------------------------------------

def test_estimate_sigma_only_returns_float_and_fit_result():
    G = _er_graph(40, 0.15, seed=1)
    sigma, result = estimate_sigma_only(G, d=1)
    assert isinstance(sigma, float)
    assert math.isfinite(sigma)
    assert result is not None  # statsmodels result object


def test_estimate_sigma_only_accepts_legacy_max_edges_kwargs():
    # The legacy max_edges / max_non_edges params should not error
    G = _er_graph(30, 0.2, seed=2)
    sigma, _ = estimate_sigma_only(G, d=0, max_edges=50, max_non_edges=50)
    assert math.isfinite(sigma)


def test_estimate_sigma_many_returns_list_of_floats():
    G = _er_graph(30, 0.2, seed=3)
    sigmas = estimate_sigma_many(G, d=0, n_repeats=4, max_pairs=80, seed=1)
    assert isinstance(sigmas, list)
    assert len(sigmas) == 4
    assert all(math.isfinite(s) for s in sigmas)


def test_estimate_sigma_many_without_max_pairs_is_deterministic():
    # When max_pairs is None, all pairs are enumerated → every repeat identical
    G = _er_graph(20, 0.25, seed=4)
    sigmas = estimate_sigma_many(G, d=0, n_repeats=3, seed=42)
    assert sigmas[0] == sigmas[1] == sigmas[2]


# -------------------------------------------------------------------
# calculate_graph_attributes
# -------------------------------------------------------------------

def test_calculate_attributes_none_graph_returns_nan_dict():
    out = calculate_graph_attributes(None)
    for k in ("nodes", "edges", "density", "avg_clustering"):
        assert math.isnan(out[k])


def test_calculate_attributes_returns_expected_keys_and_counts():
    G = _er_graph(20, 0.3, seed=5)
    out = calculate_graph_attributes(G)
    assert out["nodes"] == 20
    assert out["edges"] == G.number_of_edges()
    assert "density" in out
    assert "avg_clustering" in out
    assert "num_components" in out


def test_calculate_attributes_empty_graph_returns_nan_dict():
    out = calculate_graph_attributes(nx.Graph())
    assert math.isnan(out["nodes"])
