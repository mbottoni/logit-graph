"""Edge cases for LogitRegEstimator beyond the API surface in
test_logit_estimator_api.py (construction, compute_aic, select_d, recovery).

Adds: layer2 toggle effect on offsets, max_pairs subsampling + determinism,
the statsmodels fit fallback, extra_penalty_per_d selection pressure, and
estimate_parameters under fix_beta True/False.
"""
import numpy as np
import networkx as nx

from logit_graph.logit_estimator import LogitRegEstimator


def _er(n, p, seed):
    return nx.to_numpy_array(nx.erdos_renyi_graph(n, p, seed=seed))


# -------------------------------------------------------------------
# get_features_labels — layer2 toggle and subsampling
# -------------------------------------------------------------------

def test_layer2_toggle_changes_offsets_in_bounded_mode():
    # In "bounded" mode the offset is log(1+S_i)+log(1+S_j); removing edge
    # (i,j) drops each endpoint's degree by 1, so layer2 changes the offsets.
    # (Note: "incremental" mode at d=1 is invariant to edge removal because the
    # common-neighbor count excludes i and j.)
    adj = _er(25, 0.3, seed=0)
    est = LogitRegEstimator(adj, d=1, feature_mode="bounded")
    feat_l2, _ = est.get_features_labels(layer2=True)
    feat_no, _ = est.get_features_labels(layer2=False)
    assert not np.allclose(feat_l2[:, 1], feat_no[:, 1])


def test_max_pairs_subsamples_and_is_deterministic():
    adj = _er(20, 0.25, seed=1)
    est = LogitRegEstimator(adj, d=1)
    f1, l1 = est.get_features_labels(max_pairs=30, seed=123)
    f2, l2 = est.get_features_labels(max_pairs=30, seed=123)
    assert f1.shape[0] == 30
    assert len(l1) == 30
    # Same seed ⇒ identical subsample.
    np.testing.assert_array_equal(f1, f2)
    assert l1 == l2


# -------------------------------------------------------------------
# _fit_offset_logit — statsmodels path returns a finite-llf result
# -------------------------------------------------------------------

def test_fit_offset_logit_returns_finite_llf():
    adj = _er(30, 0.2, seed=2)
    est = LogitRegEstimator(adj, d=1)
    offsets, labels = est.get_features_labels()
    result = est._fit_offset_logit(offsets[:, 1], labels)
    assert np.isfinite(result.llf)


# -------------------------------------------------------------------
# select_d — extra_penalty_per_d pressure
# -------------------------------------------------------------------

def test_large_extra_penalty_per_d_selects_zero():
    adj = _er(25, 0.2, seed=3)
    est = LogitRegEstimator(adj, d=1)
    # A huge per-d penalty makes d=0 (penalty 0) strictly cheapest.
    best, stats = est.select_d(d_candidates=[0, 1, 2, 3], extra_penalty_per_d=1e6)
    assert best == 0
    # Penalty is additive: stats[d].aic grows monotonically with d here.
    assert stats[1]["aic"] < stats[2]["aic"] < stats[3]["aic"]


# -------------------------------------------------------------------
# estimate_parameters — fix_beta True/False
# -------------------------------------------------------------------

def test_estimate_parameters_fix_beta_true_single_sigma():
    adj = _er(20, 0.25, seed=4)
    est = LogitRegEstimator(adj, d=1)
    result, params, features = est.estimate_parameters(fix_beta=True)
    arr = np.asarray(params)
    # Offset logit estimates only the intercept sigma.
    assert arr.shape[0] == 1
    assert np.all(np.isfinite(arr))


def test_estimate_parameters_fix_beta_false_two_params():
    adj = _er(20, 0.25, seed=5)
    est = LogitRegEstimator(adj, d=1)
    result, params, features = est.estimate_parameters(fix_beta=False, l1_wt=1, alpha=0.0)
    arr = np.asarray(params)
    # const + degree-sum coefficient.
    assert arr.shape[0] == 2
    assert np.all(np.isfinite(arr))
