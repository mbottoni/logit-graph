"""Tests for the dyadic-cluster-robust SE of the offset-logit sigma estimate."""
import math

import numpy as np
import networkx as nx
import pytest

from logit_graph.graph import GraphModel
from logit_graph.robust_se import (
    dyadic_robust_sigma_se,
    fit_sigma,
    fit_sigma_with_robust_se,
    select_d_aic,
)


def _er(n, p, seed):
    return nx.to_numpy_array(nx.erdos_renyi_graph(n, p, seed=seed))


# -------------------------------------------------------------------
# At d=0 the dyads are independent (ER) -> robust reduces to naive
# -------------------------------------------------------------------

def test_robust_reduces_to_naive_at_d0():
    adj = _er(400, 0.05, seed=0)
    sigma, se_r, se_n = fit_sigma_with_robust_se(adj, d=0)
    # One realization: B fluctuates around A, so within a few percent.
    assert math.isclose(se_r, se_n, rel_tol=0.15)


def test_naive_se_equals_inverse_sqrt_fisher_info():
    from scipy.special import expit
    adj = _er(300, 0.08, seed=1)
    sigma, _ll, offsets, labels = fit_sigma(adj, d=0)
    p = expit(sigma + offsets)
    A = np.sum(p * (1 - p))
    _, se_n = dyadic_robust_sigma_se(offsets, labels, sigma, adj.shape[0])
    assert math.isclose(se_n, 1.0 / math.sqrt(A), rel_tol=1e-9)


# -------------------------------------------------------------------
# d=0 robust SE matches the Monte-Carlo sampling SD of sigma_hat
# -------------------------------------------------------------------

def test_robust_se_matches_montecarlo_d0():
    n, p, M = 300, 0.03, 150
    rng = np.random.default_rng(7)
    adj = _er(n, p, seed=3)
    _, se_r, _ = fit_sigma_with_robust_se(adj, d=0)
    mc = []
    for _ in range(M):
        g = _er(n, p, seed=int(rng.integers(1 << 30)))
        s, _, _, _ = fit_sigma(g, d=0)
        mc.append(s)
    mc_sd = float(np.std(mc, ddof=1))
    assert abs(se_r - mc_sd) / mc_sd < 0.3


# -------------------------------------------------------------------
# At d=1 (dependent dyads) the robust SE inflates beyond the naive SE
# -------------------------------------------------------------------

def test_robust_exceeds_naive_under_dependence_d1():
    n, sigma = 120, -1.0
    gm = GraphModel(n=n, d=1, sigma=sigma, er_p=0.2, layer2=True,
                    feature_mode="incremental", seed=11)
    for _ in range(25 * n):
        gm.add_remove_edge()
    _, se_r, se_n = fit_sigma_with_robust_se(gm.graph, d=1)
    assert se_r > se_n


# -------------------------------------------------------------------
# select_d_aic
# -------------------------------------------------------------------

def test_select_d_aic_returns_candidate_and_aligned_outputs():
    adj = _er(60, 0.2, seed=5)
    d_hat, sigma_hat, offsets, labels, aic_by_d = select_d_aic(adj, [0, 1])
    assert d_hat in (0, 1)
    assert set(aic_by_d.keys()) == {0, 1}
    assert all(np.isfinite(v) for v in aic_by_d.values())
    n = adj.shape[0]
    assert offsets.shape[0] == n * (n - 1) // 2
    assert len(labels) == n * (n - 1) // 2
    # The returned offsets correspond to d_hat: re-fitting gives the same sigma.
    s2, _, _, _ = fit_sigma(adj, d_hat)
    assert math.isclose(sigma_hat, s2, rel_tol=1e-9, abs_tol=1e-9)
