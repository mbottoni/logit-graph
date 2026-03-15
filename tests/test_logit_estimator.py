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
    # Default fix_beta=True: only sigma is estimated (beta fixed at 1)
    result, params, feats = est.estimate_parameters(features=X, labels=y)
    assert len(params) == 1
    assert feats.shape == X.shape
    # Legacy fix_beta=False: both sigma and beta estimated
    result2, params2, _ = est.estimate_parameters(features=X, labels=y, fix_beta=False)
    assert len(params2) == 2


