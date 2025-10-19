import numpy as np
import networkx as nx

from logit_graph.simulation import estimate_sigma_only, _build_features_labels_sampled


def test_build_features_labels_sampled_balanced_sizes():
    G = nx.erdos_renyi_graph(20, 0.2, seed=1)
    X, y = _build_features_labels_sampled(G, d=1, max_edges=30, max_non_edges=30, random_state=123)
    assert X.shape[0] == len(y)
    assert X.shape[1] == 3
    # Expect balanced labels
    assert sum(y) == 30


def test_estimate_sigma_only_returns_float_and_model():
    G = nx.erdos_renyi_graph(25, 0.2, seed=0)
    sigma, model = estimate_sigma_only(G, d=1, max_edges=50, max_non_edges=50, l1_wt=1, alpha=0, random_state=42)
    assert isinstance(sigma, float)
    assert hasattr(model, 'params')

