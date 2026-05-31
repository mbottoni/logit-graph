"""API coverage for LogitRegEstimator (the public estimator class)."""
import math

import numpy as np
import networkx as nx
import pytest

from logit_graph.logit_estimator import LogitRegEstimator


def _er(n, p, seed):
    return nx.to_numpy_array(nx.erdos_renyi_graph(n, p, seed=seed))


# -------------------------------------------------------------------
# Construction
# -------------------------------------------------------------------

def test_accepts_numpy_adjacency():
    adj = _er(20, 0.2, seed=0)
    est = LogitRegEstimator(adj, d=1)
    assert est.n == 20
    assert est.adj is adj


def test_accepts_networkx_graph():
    G = nx.erdos_renyi_graph(15, 0.2, seed=1)
    est = LogitRegEstimator(G, d=1)
    assert est.n == 15
    assert isinstance(est.adj, np.ndarray)
    assert est.adj.shape == (15, 15)


def test_invalid_graph_type_raises():
    with pytest.raises(ValueError):
        LogitRegEstimator("not a graph", d=1)


def test_default_feature_mode_is_incremental():
    est = LogitRegEstimator(_er(10, 0.3, seed=2), d=1)
    assert est.feature_mode == "incremental"


# -------------------------------------------------------------------
# get_features_labels
# -------------------------------------------------------------------

def test_get_features_labels_shapes():
    adj = _er(12, 0.25, seed=3)
    est = LogitRegEstimator(adj, d=1, feature_mode="bounded")
    features, labels = est.get_features_labels()
    # n*(n-1)/2 = 66 pairs
    assert features.shape == (66, 2)  # statsmodels adds constant column
    assert len(labels) == 66


def test_get_features_labels_label_count_matches_edges():
    adj = _er(10, 0.4, seed=4)
    est = LogitRegEstimator(adj, d=1, feature_mode="bounded")
    _, labels = est.get_features_labels()
    assert sum(labels) == int(adj.sum() / 2)


# -------------------------------------------------------------------
# compute_aic — return shape and basic correctness
# -------------------------------------------------------------------

def test_compute_aic_returns_finite_with_expected_keys():
    adj = _er(20, 0.2, seed=5)
    est = LogitRegEstimator(adj, d=1)
    stats = est.compute_aic()
    # Keys returned by aic_from_offset_fit + d_est, n_obs
    for key in ("aic", "ll", "k", "sigma_hat", "d_est", "n_obs"):
        assert key in stats
    assert math.isfinite(stats["aic"])
    assert math.isfinite(stats["sigma_hat"])
    assert stats["k"] == 1.0
    assert stats["n_obs"] == 20 * 19 / 2


def test_compute_aic_d_est_override_changes_d():
    adj = _er(20, 0.2, seed=6)
    est = LogitRegEstimator(adj, d=1)
    s1 = est.compute_aic(d_est=1)
    s2 = est.compute_aic(d_est=2)
    assert s1["d_est"] == 1.0
    assert s2["d_est"] == 2.0


def test_compute_aic_extra_penalty_increases_aic():
    adj = _er(20, 0.2, seed=7)
    est = LogitRegEstimator(adj, d=1)
    no_pen = est.compute_aic(extra_penalty=0.0)
    with_pen = est.compute_aic(extra_penalty=5.0)
    assert with_pen["aic"] == no_pen["aic"] + 5.0
    # Penalty does not affect sigma_hat or ll
    assert with_pen["sigma_hat"] == no_pen["sigma_hat"]
    assert with_pen["ll"] == no_pen["ll"]


# -------------------------------------------------------------------
# select_d
# -------------------------------------------------------------------

def test_select_d_returns_d_in_candidates():
    adj = _er(25, 0.15, seed=8)
    est = LogitRegEstimator(adj, d=1)
    best, stats = est.select_d(d_candidates=[0, 1, 2])
    assert best in (0, 1, 2)
    assert set(stats.keys()) == {0, 1, 2}


def test_select_d_picks_argmin_aic():
    adj = _er(25, 0.15, seed=9)
    est = LogitRegEstimator(adj, d=1)
    best, stats = est.select_d(d_candidates=[0, 1, 2, 3])
    chosen_aic = stats[best]["aic"]
    for d, s in stats.items():
        assert chosen_aic <= s["aic"]


# -------------------------------------------------------------------
# Sigma recovery on synthetic ER (sigma ≡ logit(density) when d=0)
# -------------------------------------------------------------------

def test_sigma_hat_close_to_logit_density_on_ER_with_d0():
    n, p = 100, 0.10
    adj = _er(n, p, seed=10)
    est = LogitRegEstimator(adj, d=0)
    stats = est.compute_aic(d_est=0)
    # For d=0 the feature mode contributes nothing → sigma_hat ≈ logit(empirical density)
    emp_density = adj.sum() / (n * (n - 1))
    expected = math.log(emp_density / (1 - emp_density))
    assert abs(stats["sigma_hat"] - expected) < 0.05
