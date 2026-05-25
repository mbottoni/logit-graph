"""Tests for fast scalar offset-logit fit and multi-d pair datasets."""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest
import statsmodels.api as sm

from logit_graph.lg_features import build_pair_dataset
from logit_graph.lg_features_fast import build_multi_d_pair_datasets_fast
from logit_graph.offset_logit import fit_offset_logit_fast
from logit_graph.experiments.sweeps import select_d_ensemble, simulate_graph


def _er_adj(n: int, p: float, seed: int) -> np.ndarray:
    return nx.to_numpy_array(nx.erdos_renyi_graph(n, p, seed=seed))


def test_fit_offset_logit_matches_statsmodels():
    adj = _er_adj(40, 0.15, seed=3)
    for d in [0, 1, 2, 3]:
        offsets, labels = build_pair_dataset(adj, d=d, mode="incremental", layer2=True)
        sigma_fast, ll_fast = fit_offset_logit_fast(offsets, labels)
        if d == 0:
            p_hat = np.clip(labels.mean(), 1e-15, 1.0 - 1e-15)
            sigma_closed = float(np.log(p_hat / (1.0 - p_hat)))
            assert sigma_fast == pytest.approx(sigma_closed, rel=1e-5, abs=1e-4)
            continue
        y = labels.astype(int)
        x = np.ones((len(y), 1), dtype=float)
        sm_res = sm.Logit(y, x, offset=offsets).fit(method="lbfgs", disp=False, maxiter=200)
        assert sigma_fast == pytest.approx(float(sm_res.params[0]), rel=1e-5, abs=1e-4)
        assert ll_fast == pytest.approx(float(sm_res.llf), rel=1e-5, abs=1e-3)


def test_multi_d_offsets_match_single_d():
    adj = _er_adj(30, 0.2, seed=11)
    d_values = [0, 1, 2, 3]
    labels_multi, offsets_multi = build_multi_d_pair_datasets_fast(
        adj, d_values, mode="incremental",
    )
    for d in d_values:
        offsets_single, labels_single = build_pair_dataset(
            adj, d=d, mode="incremental", layer2=True,
        )
        np.testing.assert_array_equal(labels_multi, labels_single)
        np.testing.assert_allclose(offsets_multi[d], offsets_single, rtol=0, atol=1e-12)


def test_select_d_ensemble_unchanged_smoke():
    n, nit = 50, 3000
    graphs = [
        simulate_graph(n, 2, sigma=-3.0, n_iter=nit, feature_mode="incremental", seed=10 + m)
        for m in range(3)
    ]
    hat, stats = select_d_ensemble(
        graphs, [0, 1, 2, 3], "incremental", extra_penalty_per_d=3.0,
    )
    assert hat in (0, 1, 2, 3)
    for d in (0, 1, 2, 3):
        assert np.isfinite(stats[d]["aic"])
        assert np.isfinite(stats[d]["sigma_hat"])
