import numpy as np
import networkx as nx

from logit_graph.logit_estimator import LogitRegEstimator


def small_adj(n=8, p=0.3, seed=0):
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return nx.to_numpy_array(G)


def test_get_features_labels_shapes_and_values():
    adj = small_adj(10, 0.2, seed=1)
    est = LogitRegEstimator(adj, d=1, verbose=True)
    X, y = est.get_features_labels()
    # Intercept + 1 symmetric feature (S_i + S_j)
    assert X.shape[1] == 2
    # Labels length equals number of edges + sampled non-edges
    assert len(y) == X.shape[0]
    # Values finite
    assert np.isfinite(X).all()


def test_estimate_parameters_runs_and_returns_result():
    adj = small_adj(12, 0.25, seed=2)
    est = LogitRegEstimator(adj, d=1)
    X, y = est.get_features_labels()
    result, params, feats = est.estimate_parameters(l1_wt=1, alpha=0, features=X, labels=y)
    # Expect intercept (sigma) + 1 coefficient (beta)
    assert len(params) == 2
    assert feats.shape == X.shape


