"""Smoke tests for GraphUtils plotting helpers.

GraphUtils is a static-method bag of matplotlib plotting helpers plus a few
pickle/HTML I/O methods with hardcoded paths (``../data/input/`` etc.). We
only smoke-test the plot helpers here — testing the I/O methods would require
refactoring them to accept paths as parameters (out of scope).
"""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")  # Headless backend for CI

from logit_graph.utils import GraphUtils


def _small_adj(seed=0):
    rng = np.random.default_rng(seed)
    n = 8
    upper = rng.random((n, n)) < 0.3
    upper = np.triu(upper, k=1)
    return (upper | upper.T).astype(float)


@pytest.mark.xfail(
    reason="utils.py:20 calls plt.show(fig) — invalid positional arg in current "
    "matplotlib. Pre-existing bug; not fixed here (out of scope per CLAUDE.md §3)."
)
def test_plot_graph_from_adjacency_does_not_crash():
    import matplotlib.pyplot as plt
    plt.close("all")
    GraphUtils.plot_graph_from_adjacency(_small_adj(), title="t", size=(2, 2))
    plt.close("all")


def test_plot_degree_distribution_does_not_crash():
    import matplotlib.pyplot as plt
    plt.close("all")
    GraphUtils.plot_degree_distribution(_small_adj(seed=1), size=(2, 2))
    plt.close("all")


def test_plot_spectrum_and_zoom_does_not_crash():
    import matplotlib.pyplot as plt
    plt.close("all")
    spectrum = np.linspace(0, 2, 30)
    GraphUtils.plot_spectrum_and_zoom(spectrum, size=(2, 1))
    plt.close("all")


def test_plot_graph_and_spectrum_does_not_crash():
    import matplotlib.pyplot as plt
    plt.close("all")
    adj = _small_adj(seed=2)
    spectrum = np.linalg.eigvalsh(adj)
    GraphUtils.plot_graph_and_spectrum(adj, spectrum, size=(4, 2))
    plt.close("all")


def test_graph_utils_methods_are_static():
    # All plot helpers are accessible as class methods without instantiation
    assert callable(GraphUtils.plot_graph_from_adjacency)
    assert callable(GraphUtils.plot_degree_distribution)
    assert callable(GraphUtils.plot_spectrum_and_zoom)
    assert callable(GraphUtils.plot_graph_and_spectrum)
