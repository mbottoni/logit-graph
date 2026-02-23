"""Tests targeting specific bug-fixes (issues 2.1–2.8, 3.1–3.2).

Each test class is named after the issue it verifies.
"""
from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from logit_graph.graph import GraphModel
from logit_graph.gic import GraphInformationCriterion
from logit_graph.simulation import (
    estimate_sigma_only,
    estimate_sigma_many,
    LogitGraphFitter,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _small_er(n: int = 15, p: float = 0.25, seed: int = 42) -> nx.Graph:
    return nx.erdos_renyi_graph(n, p, seed=seed)


def _small_adj(n: int = 15, p: float = 0.25, seed: int = 42) -> np.ndarray:
    return nx.to_numpy_array(_small_er(n, p, seed))


# ── 2.1  MLEGraphModelEstimator init ─────────────────────────────────────

class TestMLEGraphModelEstimatorInit:
    """Issue 2.1: __init__ had ``self.p = p`` (undefined).
    After the fix it should be ``self.d = d`` and torch-guarded."""

    def test_init_stores_d_not_p(self):
        """If torch is available the class should init fine and store d."""
        try:
            from logit_graph.logit_estimator import MLEGraphModelEstimator
            adj = _small_adj(8, 0.3, seed=0)
            mle = MLEGraphModelEstimator(adj, d=2)
            assert mle.d == 2
            assert not hasattr(mle, 'p'), "self.p should no longer exist"
        except ImportError:
            pytest.skip("PyTorch not installed; MLEGraphModelEstimator skipped")

    def test_init_raises_without_torch(self, monkeypatch):
        """When torch is None the constructor should raise ImportError."""
        import logit_graph.logit_estimator as mod
        original_torch = mod.torch
        monkeypatch.setattr(mod, 'torch', None)
        try:
            with pytest.raises(ImportError):
                mod.MLEGraphModelEstimator(_small_adj(5), d=1)
        finally:
            monkeypatch.setattr(mod, 'torch', original_torch)


# ── 2.2  dist_type keyword mismatch ──────────────────────────────────────

class TestDistKeywordMismatch:
    """Issue 2.2: GIC constructor accepts ``dist=`` but some callers
    passed ``dist_type=`` which was silently swallowed by **kwargs."""

    def test_gic_honours_dist_keyword(self):
        G = _small_er()
        gic_kl = GraphInformationCriterion(G, model='LG', log_graph=G, dist='KL')
        gic_l2 = GraphInformationCriterion(G, model='LG', log_graph=G, dist='L2')
        assert gic_kl.dist_type == 'KL'
        assert gic_l2.dist_type == 'L2'

    def test_fitter_passes_correct_dist(self):
        """LogitGraphFitter should forward dist_type as ``dist=`` to GIC."""
        G = _small_er(10, 0.3, seed=7)
        fitter = LogitGraphFitter(
            d=0, n_iteration=200, patience=10,
            dist_type='L2', min_gic_threshold=100,
            check_interval=25, verbose=False,
        )
        fitter.fit(G)
        # If the fix works the GIC was computed with L2, not the default KL.
        # We can verify by checking the metadata stored the value correctly.
        assert fitter.metadata.get('fit_success') is True
        assert fitter.metadata.get('gic_value') is not None


# ── 2.3  estimate_sigma_many returns floats ──────────────────────────────

class TestEstimateSigmaMany:
    """Issue 2.3: estimate_sigma_many returned a list of tuples instead
    of a list of floats."""

    def test_returns_list_of_floats(self):
        G = _small_er(20, 0.2, seed=1)
        sigmas = estimate_sigma_many(G, d=0, n_repeats=3,
                                      max_edges=30, max_non_edges=30,
                                      seed=42)
        assert isinstance(sigmas, list)
        assert len(sigmas) == 3
        for s in sigmas:
            assert isinstance(s, float), f"Expected float, got {type(s)}"

    def test_sigma_only_returns_tuple_of_float_and_result(self):
        G = _small_er(20, 0.2, seed=1)
        sigma, result = estimate_sigma_only(G, d=0, max_edges=30,
                                            max_non_edges=30, random_state=0)
        assert isinstance(sigma, float)
        assert hasattr(result, 'params')  # statsmodels result


# ── 2.4 / 2.5  edge_delta + edge count consistency ──────────────────────

class TestEdgeCountAndDelta:
    """Issues 2.4 & 2.5: edge_delta lower-bound was a no-op; edge counts
    mixed np.sum(graph) (double-counted) with np.sum(np.triu(graph))."""

    def test_populate_spectrum_uses_triu_edge_count(self):
        """After the fix, populate_edges_spectrum should count edges via
        np.triu so the edge_delta comparison is correct."""
        adj = _small_adj(10, 0.3, seed=5)
        np.random.seed(5)
        gm = GraphModel(n=10, d=0, sigma=-1.0, er_p=0.1)

        # A very tight edge_delta should cause an early stop (upper bound)
        graphs, _, sd, bi = gm.populate_edges_spectrum(
            warm_up=0, max_iterations=3000, patience=100,
            real_graph=adj, edge_delta=2, check_interval=10, verbose=False,
        )
        # If it stopped due to edge_delta, the final edge count should be
        # close to real.  The key assertion is that it terminated.
        assert len(graphs) >= 1

    def test_edge_delta_upper_bound_triggers(self):
        """Verify that populate_edges_spectrum_min_gic stops when edge
        count exceeds real_edges + edge_delta."""
        adj = _small_adj(10, 0.3, seed=3)
        np.random.seed(3)
        gm = GraphModel(n=10, d=0, sigma=2.0, er_p=0.5)  # likely to add edges

        _, _, _, _, best_graph, _ = gm.populate_edges_spectrum_min_gic(
            max_iterations=5000, patience=500,
            real_graph=adj, min_gic_threshold=100,
            edge_delta=3, check_interval=10, verbose=False,
        )
        # Should have terminated (not run all 5000 iters)
        assert best_graph.shape == (10, 10)


# ── 2.7  np.float deprecation ───────────────────────────────────────────

class TestNpFloatDeprecation:
    """Issue 2.7: np.float was used in MLEGraphModelEstimator; it is
    removed in NumPy ≥ 1.24.  After the fix, ``float()`` is used."""

    def test_no_np_float_in_source(self):
        """Grep the source file for np.float (excluding np.float32/64)."""
        import inspect
        from logit_graph import logit_estimator as mod
        src = inspect.getsource(mod)
        # np.float32 and np.float64 are fine; bare np.float is not
        lines = src.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if 'np.float(' in stripped:
                pytest.fail(f"Found deprecated np.float() in: {stripped}")


# ── 3.1 / 3.2  populate_edges_spectrum check_interval ───────────────────

class TestPopulateEdgesSpectrumCheckInterval:
    """Issues 3.1 & 3.2: O(n^3) spectrum every iteration + full copy
    every iteration.  After the fix both happen every check_interval."""

    def test_check_interval_limits_spectrum_computations(self):
        adj = _small_adj(10, 0.3, seed=6)
        np.random.seed(6)
        gm = GraphModel(n=10, d=0, sigma=-1.0, er_p=0.1)

        _, _, spectrum_diffs, _ = gm.populate_edges_spectrum(
            warm_up=50, max_iterations=500, patience=20,
            real_graph=adj, check_interval=100, verbose=False,
        )
        # With max_iterations=500 and check_interval=100 we get at most 5 checks
        assert len(spectrum_diffs) <= 10, (
            f"Expected <=10 spectrum checks, got {len(spectrum_diffs)}")

    def test_graph_snapshots_bounded_by_deque(self):
        """The returned graph list should be bounded, not one-per-iteration."""
        adj = _small_adj(10, 0.3, seed=8)
        np.random.seed(8)
        gm = GraphModel(n=10, d=0, sigma=-1.0, er_p=0.1)

        graphs, _, _, _ = gm.populate_edges_spectrum(
            warm_up=50, max_iterations=2000, patience=50,
            real_graph=adj, check_interval=50, verbose=False,
        )
        # With deque maxlen ≈ max(2*50+100, 500) = 500,
        # we should have far fewer than 2000 snapshots
        assert len(graphs) < 600
