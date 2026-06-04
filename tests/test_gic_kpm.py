"""Tests for the KPM (Kernel Polynomial Method) spectral-density path of GIC.

The dense ``eigvalsh`` path is covered in ``test_gic_core.py``; this file
exercises the large-n approximation (``kpm_spectral_density``), the routing in
``compute_spectral_density``, the Jackson kernel, degenerate graphs, and the
per-model parameter count.
"""
import math

import networkx as nx
import numpy as np
import pytest

from logit_graph import gic
from logit_graph.gic import (
    GraphInformationCriterion,
    _jackson_kernel,
    kpm_spectral_density,
)

SEED = 0


def _connected_er(n, p, seed):
    """An ER graph resampled until connected (so every node has degree>0)."""
    s = seed
    while True:
        g = nx.erdos_renyi_graph(n, p, seed=s)
        if nx.is_connected(g):
            return g
        s += 1


# -------------------------------------------------------------------
# kpm_spectral_density — output shape / normalization / sign
# -------------------------------------------------------------------

def test_kpm_returns_50_bins_and_edges_span_0_2():
    g = _connected_er(80, 0.1, SEED)
    lap = nx.normalized_laplacian_matrix(g)
    density, edges = kpm_spectral_density(lap, n_bins=50)
    assert density.shape == (50,)
    assert edges.shape == (51,)
    assert math.isclose(edges[0], 0.0)
    assert math.isclose(edges[-1], 2.0)


def test_kpm_density_is_non_negative():
    g = _connected_er(80, 0.1, SEED)
    lap = nx.normalized_laplacian_matrix(g)
    density, _ = kpm_spectral_density(lap, n_bins=50)
    assert np.all(density >= 0.0)
    assert np.all(np.isfinite(density))


def test_kpm_density_integrates_to_one():
    g = _connected_er(80, 0.12, SEED)
    lap = nx.normalized_laplacian_matrix(g)
    density, edges = kpm_spectral_density(lap, n_bins=50)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # KPM normalizes via the trapezoid rule over the bin centers.
    assert math.isclose(float(np.trapezoid(density, centers)), 1.0, rel_tol=1e-6)


# -------------------------------------------------------------------
# kpm_spectral_density — determinism
# -------------------------------------------------------------------

def test_kpm_is_deterministic_under_fixed_seed():
    g = _connected_er(80, 0.1, SEED)
    lap = nx.normalized_laplacian_matrix(g)
    d1, _ = kpm_spectral_density(lap, seed=123)
    d2, _ = kpm_spectral_density(lap, seed=123)
    np.testing.assert_array_equal(d1, d2)


# -------------------------------------------------------------------
# kpm_spectral_density — accuracy vs the exact spectrum
# -------------------------------------------------------------------

def test_kpm_recovers_mean_eigenvalue():
    # For a connected graph trace(L_norm) = n, so the mean eigenvalue is 1.
    g = _connected_er(120, 0.15, SEED)
    lap = nx.normalized_laplacian_matrix(g)

    exact_mean = float(np.linalg.eigvalsh(lap.todense()).mean())
    density, edges = kpm_spectral_density(lap, n_bins=50)
    centers = 0.5 * (edges[:-1] + edges[1:])
    kpm_mean = float(np.trapezoid(centers * density, centers))

    assert math.isclose(exact_mean, 1.0, abs_tol=0.05)
    assert abs(kpm_mean - exact_mean) < 0.15


# -------------------------------------------------------------------
# _jackson_kernel
# -------------------------------------------------------------------

@pytest.mark.parametrize("M", [10, 30, 60])
def test_jackson_kernel_shape_and_leading_coefficient(M):
    g = _jackson_kernel(M)
    assert g.shape == (M,)
    assert np.all(np.isfinite(g))
    # g[0] == 1 exactly by construction; it is also the maximum coefficient.
    assert math.isclose(g[0], 1.0, abs_tol=1e-12)
    assert g.argmax() == 0
    # Damping pulls higher-order coefficients toward zero.
    assert g[-1] < g[0]


# -------------------------------------------------------------------
# compute_spectral_density — routing to the KPM path above threshold
# -------------------------------------------------------------------

def test_compute_spectral_density_uses_kpm_above_threshold(monkeypatch):
    monkeypatch.setattr(gic, "KPM_THRESHOLD", 50)
    g = _connected_er(80, 0.1, SEED)  # n=80 > 50 ⇒ KPM path

    gc = GraphInformationCriterion(g, model="ER", p=0.1)
    hist, edges = gc.compute_spectral_density(g)

    expected, exp_edges = kpm_spectral_density(
        nx.normalized_laplacian_matrix(g), n_bins=50
    )
    np.testing.assert_array_equal(hist, expected)
    np.testing.assert_array_equal(edges, exp_edges)


def test_compute_spectral_density_uses_dense_below_threshold(monkeypatch):
    monkeypatch.setattr(gic, "KPM_THRESHOLD", 500)
    g = _connected_er(40, 0.2, SEED)  # n=40 < 500 ⇒ dense histogram path
    gc = GraphInformationCriterion(g, model="ER", p=0.2)
    hist, edges = gc.compute_spectral_density(g)
    bin_width = edges[1] - edges[0]
    # Dense path uses np.histogram(density=True): sum*binwidth == 1.
    assert math.isclose(hist.sum() * bin_width, 1.0, abs_tol=1e-9)


# -------------------------------------------------------------------
# Degenerate graphs — isolated / zero-degree nodes
# -------------------------------------------------------------------

def test_dense_density_handles_isolated_nodes(monkeypatch):
    monkeypatch.setattr(gic, "KPM_THRESHOLD", 500)
    g = nx.erdos_renyi_graph(30, 0.2, seed=SEED)
    g.add_node(99)  # isolated ⇒ zero-degree row in the normalized Laplacian
    gc = GraphInformationCriterion(g, model="ER", p=0.2)
    hist, edges = gc.compute_spectral_density(g)
    assert np.all(np.isfinite(hist))
    bin_width = edges[1] - edges[0]
    assert math.isclose(hist.sum() * bin_width, 1.0, abs_tol=1e-9)


def test_kpm_density_handles_isolated_nodes():
    g = nx.erdos_renyi_graph(80, 0.1, seed=SEED)
    g.add_node(999)  # isolated
    lap = nx.normalized_laplacian_matrix(g)
    density, _ = kpm_spectral_density(lap, n_bins=50)
    assert np.all(np.isfinite(density))
    assert np.all(density >= 0.0)


# -------------------------------------------------------------------
# _get_n_params — per-model parameter count
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,expected",
    [("ER", 1), ("BA", 1), ("GRG", 1), ("KR", 1), ("LG", 1), ("WS", 2), ("SBM", 1)],
)
def test_get_n_params_per_model(model, expected):
    g = _connected_er(20, 0.2, SEED)
    gc = GraphInformationCriterion(g, model=model, p=0.2)
    assert gc._get_n_params() == expected


def test_get_n_params_callable_model_defaults_to_one():
    g = _connected_er(20, 0.2, SEED)
    gc = GraphInformationCriterion(g, model=lambda n, p: nx.erdos_renyi_graph(n, p), p=0.2)
    assert gc._get_n_params() == 1
