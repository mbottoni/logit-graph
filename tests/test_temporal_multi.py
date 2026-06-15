"""Tests for the multi-feature (unified) temporal Logit-Graph: degree + fixed exogenous covariates."""
import networkx as nx
import numpy as np

from logit_graph.temporal_multi import (
    community_feature,
    fit_multi_params,
    grow_graph_multi,
    latent_feature,
    multi_design_from_snapshots,
)

NAMES = ["coarse", "fine", "latent"]


def _synthetic_features(n, seed):
    """Two same-community indicators + a latent inner-product feature, fixed over the trajectory."""
    rng = np.random.default_rng(seed)
    rows, cols = np.triu_indices(n, k=1)
    coarse, fine = rng.integers(0, 4, n), rng.integers(0, 16, n)
    z = rng.normal(size=(n, 4))
    Bc = (coarse[rows] == coarse[cols]).astype(float)
    Bf = (fine[rows] == fine[cols]).astype(float)
    L = (z[rows] * z[cols]).sum(1)
    L = (L - L.mean()) / (L.std() + 1e-9)
    return np.column_stack([Bc, Bf, L])


def _fit_at(n, truth, seed):
    F = _synthetic_features(n, seed)
    res = grow_graph_multi(n, 0, truth["sigma"], truth["alpha"], F,
                           [truth["coarse"], truth["fine"], truth["latent"]],
                           n_steps=6, feature_names=NAMES, allow_removal=True, seed=seed)
    return fit_multi_params(res.X, res.y, NAMES)


def test_recovers_all_five_coefficients():
    """Generate at a known (sigma, alpha, gc, gf, lam) and recover every coefficient in a loose band."""
    truth = dict(sigma=-2.0, alpha=0.05, coarse=0.8, fine=0.5, latent=0.4)
    ests = [_fit_at(220, truth, seed=100 + r) for r in range(6)]
    sigma = np.mean([e["sigma"] for e in ests])
    alpha = np.mean([e["alpha"] for e in ests])
    gc = np.mean([e["coefs"]["coarse"] for e in ests])
    gf = np.mean([e["coefs"]["fine"] for e in ests])
    lam = np.mean([e["coefs"]["latent"] for e in ests])
    assert abs(sigma - truth["sigma"]) < 0.2
    assert abs(alpha - truth["alpha"]) < 0.02
    assert abs(gc - truth["coarse"]) < 0.15
    assert abs(gf - truth["fine"]) < 0.15
    assert abs(lam - truth["latent"]) < 0.15
    assert ests[0]["n_params"] == 5  # sigma, alpha + 3 features


def test_alpha_consistency_shrinks_with_n():
    """The spread of alpha_hat should shrink as n grows (consistency of the pooled MLE)."""
    truth = dict(sigma=-2.0, alpha=0.05, coarse=0.8, fine=0.5, latent=0.4)
    sd_small = np.std([_fit_at(80, truth, 200 + r)["alpha"] for r in range(6)], ddof=1)
    sd_large = np.std([_fit_at(240, truth, 300 + r)["alpha"] for r in range(6)], ddof=1)
    assert sd_large < sd_small


def test_design_matches_snapshot_rebuild():
    """multi_design_from_snapshots reproduces grow_graph_multi's recorded [D, F...] design."""
    F = _synthetic_features(120, 7)
    res = grow_graph_multi(120, 0, -2.0, 0.05, F, [0.8, 0.5, 0.4],
                           n_steps=4, feature_names=NAMES, allow_removal=True, seed=7)
    X2, y2 = multi_design_from_snapshots(res.snapshots, 0, F, allow_removal=True)
    assert res.X.shape[1] == 4  # D + 3 features
    assert np.allclose(res.X, X2)
    assert np.array_equal(res.y, y2)


def test_feature_builders_shape_and_exogeneity():
    """community_feature / latent_feature return one value per upper-triangle dyad, fixed for a graph."""
    G = nx.gnp_random_graph(60, 0.1, seed=3)
    m = 60 * 59 // 2
    Bc = community_feature(G, seed=0)
    L = latent_feature(G, k=4, kind="dot")
    assert Bc.shape == (m,) and L.shape == (m,)
    assert set(np.unique(Bc)).issubset({0.0, 1.0})       # same-community indicator
    assert np.allclose(community_feature(G, seed=0), Bc)  # deterministic / exogenous
    assert abs(L.mean()) < 1e-6 and abs(L.std() - 1.0) < 1e-3  # standardized
