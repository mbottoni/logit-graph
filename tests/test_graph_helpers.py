import numpy as np
import networkx as nx

from logit_graph.simulation import estimate_sigma_only
from logit_graph.lg_features import build_pair_dataset


def test_build_pair_dataset_shapes_and_label_set():
    G = nx.erdos_renyi_graph(20, 0.2, seed=1)
    adj = nx.to_numpy_array(G)
    offsets, labels = build_pair_dataset(
        adj, d=1, mode="incremental", layer2=True, max_pairs=60, seed=123,
    )
    assert offsets.shape == labels.shape
    assert set(np.unique(labels).tolist()).issubset({0, 1})


def test_estimate_sigma_only_returns_float_and_model():
    G = nx.erdos_renyi_graph(25, 0.2, seed=0)
    sigma, model = estimate_sigma_only(
        G, d=1, max_pairs=100, random_state=42,
    )
    assert isinstance(sigma, float)
    assert hasattr(model, "params")


def test_estimate_sigma_only_accepts_legacy_kwargs():
    """The old `max_edges` / `max_non_edges` / `l1_wt` / `alpha` parameters
    are deprecated but still accepted for backwards compatibility."""
    G = nx.erdos_renyi_graph(25, 0.2, seed=0)
    sigma, _ = estimate_sigma_only(
        G, d=1, max_edges=50, max_non_edges=50,
        l1_wt=1, alpha=0, random_state=42,
    )
    assert isinstance(sigma, float)
