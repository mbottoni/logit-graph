"""Algorithm equivalence tests for fast Gibbs / feature paths."""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest
from scipy.special import expit

from logit_graph.graph import GraphModel
from logit_graph.lg_features import (
    FEATURE_MODES,
    pair_feature,
    pair_feature_layer2,
)
from logit_graph.lg_features_fast import (
    FastGibbsGraph,
    make_gibbs_draws,
    nbrs_from_adj,
    pair_feature_layer2_csr_py,
    pair_feature_layer2_nbrs,
    pair_feature_nbrs,
)
from logit_graph.experiments.sweeps import simulate_graph


def _random_adj(n: int, p: float, seed: int) -> np.ndarray:
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return nx.to_numpy_array(G)


@pytest.mark.parametrize("mode", FEATURE_MODES)
@pytest.mark.parametrize("d", [0, 1, 2, 3])
def test_pair_feature_nbrs_matches_matrix(mode: str, d: int):
    adj = _random_adj(12, 0.25, seed=7 + d)
    nbrs = nbrs_from_adj(adj)
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            expected = pair_feature(adj, i, j, d, mode=mode)
            got = pair_feature_nbrs(nbrs, i, j, d, mode=mode)
            assert got == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.mark.parametrize("mode", FEATURE_MODES)
@pytest.mark.parametrize("d", [0, 1, 2, 3])
def test_layer2_toggle_matches_copy(mode: str, d: int):
    adj = _random_adj(14, 0.3, seed=11 + d)
    nbrs = nbrs_from_adj(adj)
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            expected = pair_feature_layer2(adj, i, j, d, mode=mode)
            got_nbrs = pair_feature_layer2_nbrs(nbrs, i, j, d, mode=mode)
            got_csr = pair_feature_layer2_csr_py(nbrs, i, j, d, mode=mode)
            assert got_nbrs == pytest.approx(expected, rel=0, abs=1e-12)
            assert got_csr == pytest.approx(expected, rel=0, abs=1e-12)


def _run_gibbs(gm: GraphModel, n_iter: int, *, legacy: bool) -> np.ndarray:
    step = gm._add_remove_edge_legacy if legacy else gm.add_remove_edge
    for _ in range(n_iter):
        step()
    return gm.graph.copy()


def _run_nbrs_from_draws(gm: GraphModel, draws: np.ndarray) -> None:
    n = gm.n
    for t in range(draws.shape[0]):
        i = int(draws[t, 0])
        j = int(draws[t, 1])
        if j >= i:
            j += 1
        u = float(draws[t, 2])
        feat = pair_feature_layer2_nbrs(
            gm._nbrs, i, j, gm.d, mode=gm.feature_mode,
        )
        logit = gm.sigma + gm.alpha * gm.beta * feat
        new_val = float(u < expit(logit))
        old_val = gm._has_edge(i, j)
        gm._sync_edge(i, j, new_val, old_val)


@pytest.mark.parametrize("d", [0, 1, 2])
@pytest.mark.parametrize("feature_mode", ["incremental", "bounded"])
def test_gibbs_bitwise_reproducibility(d: int, feature_mode: str):
    n, n_iter, seed = 30, 800, 12345
    kwargs = dict(
        n=n, d=d, sigma=-2.0, er_p=0.12, layer2=True,
        feature_mode=feature_mode, seed=seed,
    )
    gm_fast = GraphModel(**kwargs)
    adj_fast = _run_gibbs(gm_fast, n_iter, legacy=False)
    gm_legacy = GraphModel(**kwargs)
    adj_legacy = _run_gibbs(gm_legacy, n_iter, legacy=True)
    np.testing.assert_array_equal(adj_fast, adj_legacy)


def test_fast_gibbs_matches_nbrs():
    n, d, n_iter, seed = 40, 1, 3000, 99
    gm = GraphModel(
        n=n, d=d, sigma=-3.0, er_p=0.12, layer2=True,
        feature_mode="incremental", seed=seed,
    )
    adj0 = gm.graph.copy()
    draws = make_gibbs_draws(n, n_iter, gm._rng)

    gm_nbrs = GraphModel(
        n=n, d=d, sigma=-3.0, er_p=0.12, layer2=True,
        feature_mode="incremental", seed=seed,
    )
    _run_nbrs_from_draws(gm_nbrs, draws)

    fg = FastGibbsGraph(
        n, d, gm.sigma, er_p=gm.er_p, rng=gm._rng,
        feature_mode="incremental", alpha=gm.alpha, beta=gm.beta,
        adj=adj0,
    )
    fg.run_from_draws(draws)
    np.testing.assert_array_equal(gm_nbrs.graph, fg.to_adjacency())


def test_simulate_graph_deterministic():
    n, d, nit = 40, 1, 2000
    a = simulate_graph(
        n, d, sigma=-3.0, n_iter=nit, feature_mode="incremental", seed=99,
    )
    b = simulate_graph(
        n, d, sigma=-3.0, n_iter=nit, feature_mode="incremental", seed=99,
    )
    np.testing.assert_array_equal(a, b)
