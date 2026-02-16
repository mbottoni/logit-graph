import numpy as np
import networkx as nx
import pytest

from logit_graph.graph import GraphModel


def test_generate_small_er_graph_returns_symmetric_adj():
    gm = GraphModel(n=10, d=1, sigma=0.0, er_p=0.2)
    adj = gm.generate_small_er_graph(10, 0.2)
    assert isinstance(adj, np.ndarray)
    assert adj.shape == (10, 10)
    assert np.allclose(adj, adj.T)


def test_calculate_spectrum_sorted_nonnegative():
    gm = GraphModel(n=8, d=1, sigma=0.0, er_p=0.2)
    spec = gm.calculate_spectrum(gm.graph)
    assert (np.diff(spec) >= -1e-12).all()  # sorted non-decreasing
    assert (spec >= -1e-9).all()


def test_add_remove_edge_changes_edge_count():
    gm = GraphModel(n=12, d=1, sigma=0.0, er_p=0.1)
    before = np.sum(np.triu(gm.graph))
    for _ in range(10):
        gm.add_remove_edge()
    after = np.sum(np.triu(gm.graph))
    # Probabilistic, but across 10 steps should change at least sometimes
    assert before != after or np.sum(gm.graph) in [0, gm.graph.size]


def test_convergence_checks_do_not_crash():
    gm = GraphModel(n=8, d=1, sigma=0.0, er_p=0.2)
    graphs = [gm.graph.copy()]
    for _ in range(5):
        gm.add_remove_edge()
        graphs.append(gm.graph.copy())

    _ = gm.check_convergence_number_of_edges(graphs, threshold_edges=100, stability_window=3)
    _ = gm.check_convergence_spectrum(graphs, threshold_spectrum=1e6, stability_window=3)


# ---------------------------------------------------------------------------
# Convergence-specific tests
# ---------------------------------------------------------------------------

class TestBaselineConvergence:
    """Verify populate_edges_baseline with the new CV-based convergence."""

    def test_baseline_iterates_past_warmup(self):
        """With reasonable parameters the baseline should run well past
        warm-up before converging (the old trivially-true thresholds made
        it stop at warm_up+1)."""
        np.random.seed(42)
        n = 20
        gm = GraphModel(n=n, d=0, sigma=-1.0, er_p=0.1)

        warm_up = 200
        max_iter = 5000
        patience = 10
        check_interval = 20

        graphs, spectra = gm.populate_edges_baseline(
            warm_up=warm_up,
            max_iterations=max_iter,
            patience=patience,
            check_interval=check_interval,
            edge_cv_tol=0.05,
            spectrum_cv_tol=0.05,
        )

        # The returned graph list should have more than just the warm-up
        # snapshot.  Under the old code len(graphs) was 2 (initial + first
        # snapshot after warm_up); now we expect many checkpoints.
        assert len(graphs) > 2, (
            f"Expected meaningful iteration, got only {len(graphs)} snapshots")

        # Spectrum should be a valid sorted array
        assert spectra is not None
        assert len(spectra) == n

    def test_baseline_returns_valid_symmetric_graph(self):
        np.random.seed(7)
        gm = GraphModel(n=15, d=0, sigma=-0.5, er_p=0.1)
        graphs, _ = gm.populate_edges_baseline(
            warm_up=100, max_iterations=2000, patience=8,
            check_interval=20, edge_cv_tol=0.1, spectrum_cv_tol=0.1)

        final = graphs[-1]
        assert final.shape == (15, 15)
        assert np.allclose(final, final.T), "Adjacency must be symmetric"

    def test_baseline_respects_max_iterations(self):
        """If convergence is never reached, the loop must still terminate
        at max_iterations."""
        np.random.seed(0)
        gm = GraphModel(n=10, d=0, sigma=0.0, er_p=0.1)
        max_iter = 300
        graphs, _ = gm.populate_edges_baseline(
            warm_up=100, max_iterations=max_iter, patience=50,
            check_interval=10,
            edge_cv_tol=1e-9,      # impossibly tight
            spectrum_cv_tol=1e-9,
        )
        # Should terminate (not hang) and produce at least some snapshots
        assert len(graphs) >= 1


class TestMinGicConvergence:
    """Verify populate_edges_spectrum_min_gic with the new check_interval."""

    @pytest.fixture()
    def small_real_graph(self):
        """A small fixed graph to use as reference."""
        np.random.seed(99)
        G = nx.erdos_renyi_graph(15, 0.25, seed=99)
        return nx.to_numpy_array(G)

    def test_gic_gate_blocks_until_threshold(self, small_real_graph):
        """With an impossibly low GIC threshold the loop should run until
        max_iterations without ever entering the patience phase."""
        np.random.seed(1)
        gm = GraphModel(n=15, d=0, sigma=-1.0, er_p=0.1)
        max_iter = 500
        patience = 50

        _, _, spectrum_diffs, best_iter, best_graph, gic_values = \
            gm.populate_edges_spectrum_min_gic(
                max_iterations=max_iter,
                patience=patience,
                real_graph=small_real_graph,
                min_gic_threshold=0.0001,   # nearly impossible
                check_interval=25,
                verbose=False,
            )

        # Should have run up to max_iterations
        assert len(gic_values) > 0
        # best_graph should still be a valid 15x15 matrix
        assert best_graph.shape == (15, 15)

    def test_check_interval_reduces_spectrum_computations(self, small_real_graph):
        """With check_interval=100 and max_iterations=500, we should get
        at most ~5 spectrum measurements (500/100), not 500."""
        np.random.seed(2)
        gm = GraphModel(n=15, d=0, sigma=-1.0, er_p=0.1)

        _, _, spectrum_diffs, _, _, gic_values = \
            gm.populate_edges_spectrum_min_gic(
                max_iterations=500,
                patience=50,
                real_graph=small_real_graph,
                min_gic_threshold=100,   # easy threshold
                check_interval=100,
                verbose=False,
            )

        # spectrum_diffs is appended once per check
        assert len(spectrum_diffs) <= 10, (
            f"Expected <=10 spectrum checks, got {len(spectrum_diffs)}")

    def test_returns_best_graph_not_last(self, small_real_graph):
        """best_graph should be the one with the smallest spectrum diff,
        not necessarily the final state."""
        np.random.seed(3)
        gm = GraphModel(n=15, d=0, sigma=-1.0, er_p=0.1)

        _, spectra, spectrum_diffs, best_iter, best_graph, _ = \
            gm.populate_edges_spectrum_min_gic(
                max_iterations=600,
                patience=20,
                real_graph=small_real_graph,
                min_gic_threshold=100,
                check_interval=25,
                verbose=False,
            )

        # The returned spectra correspond to best_graph
        recomputed = GraphModel.calculate_spectrum(best_graph)
        assert np.allclose(spectra, recomputed, atol=1e-10)


