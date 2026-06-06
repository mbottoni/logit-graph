"""Tests for the temporal / growth Logit-Graph (degree-only: generation + estimation)."""
import numpy as np

from logit_graph.temporal import (
    grow_graph,
    fit_growth_params,
    fit_growth_from_result,
    growth_design_from_snapshots,
)


def _density(adj):
    n = adj.shape[0]
    return adj.sum() / (n * (n - 1))


def test_grow_graph_density_grows_smoothly():
    """Density should increase step-by-step and be consistent across seeds
    (no equilibrium-style bimodality / degeneracy)."""
    traces = []
    for s in range(4):
        res = grow_graph(120, d=1, sigma=-4.0, alpha=0.05, n_steps=3, seed=s)
        tr = [_density(g) for g in res.snapshots]
        traces.append(tr)
        # monotone non-decreasing (edges only added)
        assert all(tr[k + 1] >= tr[k] - 1e-12 for k in range(len(tr) - 1))
    finals = np.array([t[-1] for t in traces])
    # seed-consistent (no bistable empty-vs-dense split): spread is small
    assert finals.std() < 0.15
    assert 0.0 < finals.mean() < 0.9


def test_design_matches_snapshot_rebuild():
    """growth_design_from_snapshots reproduces grow_graph's recorded design."""
    res = grow_graph(60, d=1, sigma=-3.5, alpha=0.1, n_steps=3, seed=1)
    X2, y2 = growth_design_from_snapshots(res.snapshots, d=1)
    assert res.X.shape == X2.shape
    assert res.X.shape[1] == 1  # degree-only design
    assert np.allclose(res.X, X2)
    assert np.array_equal(res.y, y2)


def test_fit_returns_two_finite_params():
    res = grow_graph(100, d=1, sigma=-4.0, alpha=0.05, n_steps=3, seed=2)
    out = fit_growth_from_result(res)
    assert out["n_params"] == 2
    for k in ("sigma", "alpha", "se_sigma", "se_alpha", "ll", "aic"):
        assert np.isfinite(out[k]), k


def test_recovery_and_consistency():
    """The degree slope alpha (and sigma) recover, and the estimate tightens as n
    grows — the property the equilibrium model lacks."""
    sigma, alpha = -4.0, 0.05

    def fit_at(n, seed):
        res = grow_graph(n, d=1, sigma=sigma, alpha=alpha, n_steps=3,
                         seed=seed, store_snapshots=False)
        return fit_growth_from_result(res)

    small = [fit_at(80, s) for s in range(5)]
    large = [fit_at(160, s) for s in range(5)]
    a_small = np.array([o["alpha"] for o in small])
    a_large = np.array([o["alpha"] for o in large])

    # recovered in a loose band at the larger n
    assert abs(a_large.mean() - alpha) < 0.04
    assert abs(np.mean([o["sigma"] for o in large]) - sigma) < 0.7
    # consistency: spread shrinks with n
    assert a_large.std() < a_small.std()


def test_removal_design_matches_snapshot_rebuild():
    """With allow_removal, the design is over ALL dyads and matches the rebuild."""
    res = grow_graph(60, d=1, sigma=-2.0, alpha=0.05, n_steps=3, seed=1,
                     allow_removal=True)
    X2, y2 = growth_design_from_snapshots(res.snapshots, d=1, allow_removal=True)
    n_pairs = 60 * 59 // 2
    # all dyads contribute every step (3 steps) -> no at-risk filtering
    assert res.X.shape == (3 * n_pairs, 1)
    assert res.X.shape == X2.shape
    assert np.allclose(res.X, X2)
    assert np.array_equal(res.y, y2)


def test_removal_reaches_moderate_density_not_saturation():
    """allow_removal turns growth into an ergodic chain: edges dissolve, so the
    process settles at a moderate density instead of drifting to saturation."""
    grow = grow_graph(150, d=0, sigma=-2.0, alpha=0.05, n_steps=2, seed=3)
    stat = grow_graph(150, d=0, sigma=-2.0, alpha=0.05, n_steps=8, seed=3,
                      allow_removal=True)
    # add-only keeps climbing; add+remove balances well below saturation
    assert _density(stat.adj) < _density(grow.adj)
    assert 0.01 < _density(stat.adj) < 0.5


def test_removal_recovers_params():
    """Estimation stays consistent under removal (predictors remain predetermined)."""
    sigma, alpha = -2.0, 0.05

    def fit_at(n, seed):
        res = grow_graph(n, d=0, sigma=sigma, alpha=alpha, n_steps=5, seed=seed,
                         store_snapshots=False, allow_removal=True)
        return fit_growth_from_result(res)

    small = np.array([fit_at(80, s)["alpha"] for s in range(5)])
    large = np.array([fit_at(200, s)["alpha"] for s in range(5)])
    assert abs(large.mean() - alpha) < 0.03
    assert abs(np.mean([fit_at(200, s)["sigma"] for s in range(5)]) - sigma) < 0.5
    assert large.std() < small.std()


def test_equilibrium_path_untouched():
    """Sanity: the equilibrium API still imports and runs (additive change)."""
    from logit_graph import simulate_graph
    adj = simulate_graph(40, 0, sigma=-2.0, n_iter=0, seed=0)
    assert adj.shape == (40, 40)
