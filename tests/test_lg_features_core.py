"""Correctness tests for the lg_features primitives (Python paths only)."""
import math

import numpy as np
import networkx as nx
import pytest

from logit_graph.lg_features import (
    ALPHA_GWESP_DEFAULT,
    build_pair_dataset,
    common_dhop_count,
    incremental_h,
    pair_feature,
    pair_feature_layer2,
    precompute_vertex_sums,
    recommended_iterations,
    sum_degree,
)


def _adj(n, edges):
    g = np.zeros((n, n), dtype=float)
    for i, j in edges:
        g[i, j] = g[j, i] = 1.0
    return g


# -------------------------------------------------------------------
# recommended_iterations
# -------------------------------------------------------------------

def test_recommended_iterations_floor_when_small_n():
    # For very small n the 20_000 floor wins over 5*n*(n-1)
    assert recommended_iterations(10) == 20_000


def test_recommended_iterations_scales_quadratically_for_large_n():
    assert recommended_iterations(100) == 5 * 100 * 99
    assert recommended_iterations(500) == 5 * 500 * 499


def test_recommended_iterations_cap_clips_below_floor():
    # If user caps at 5_000, result is 5_000 (cap takes precedence over the 20k floor)
    assert recommended_iterations(100, cap=5_000) == 5_000


def test_recommended_iterations_cap_above_formula_returns_formula():
    assert recommended_iterations(100, cap=10_000_000) == 5 * 100 * 99


# -------------------------------------------------------------------
# common_dhop_count
# -------------------------------------------------------------------

def test_common_dhop_count_d0_is_zero():
    adj = _adj(3, [(0, 1), (1, 2), (0, 2)])
    assert common_dhop_count(adj, 0, 1, d=0) == 0


def test_common_dhop_count_triangle_d1():
    # Triangle 0-1-2: (0,1) share neighbor 2
    adj = _adj(3, [(0, 1), (1, 2), (0, 2)])
    assert common_dhop_count(adj, 0, 1, d=1) == 1


def test_common_dhop_count_no_shared_neighbors():
    # Two disjoint edges → (0,2) share no neighbor
    adj = _adj(4, [(0, 1), (2, 3)])
    assert common_dhop_count(adj, 0, 2, d=1) == 0


# -------------------------------------------------------------------
# incremental_h
# -------------------------------------------------------------------

def test_incremental_h_d0_returns_zero():
    adj = _adj(3, [(0, 1), (1, 2)])
    assert incremental_h(adj, 0, 1, d=0) == 0.0


def test_incremental_h_d1_matches_gwesp_formula():
    # Triangle: (0,1) have 1 common neighbor; check GWESP closed-form
    adj = _adj(3, [(0, 1), (1, 2), (0, 2)])
    alpha = ALPHA_GWESP_DEFAULT
    c = 1
    expected = alpha * (1.0 - (1.0 - 1.0 / alpha) ** c)
    assert math.isclose(incremental_h(adj, 0, 1, d=1), expected)


def test_incremental_h_no_common_neighbor_is_zero():
    adj = _adj(4, [(0, 1), (2, 3)])
    assert incremental_h(adj, 0, 2, d=1) == 0.0


# -------------------------------------------------------------------
# pair_feature modes
# -------------------------------------------------------------------

def test_pair_feature_paper_raw_sums_d_hop_degrees():
    # Path 0-1-2: vertex 0 has d=1 ball {0,1}; degrees 1+2=3.
    # vertex 2 has d=1 ball {2,1}; degrees 1+2=3.
    adj = _adj(3, [(0, 1), (1, 2)])
    assert pair_feature(adj, 0, 2, d=1, mode="paper_raw") == 6.0


def test_pair_feature_bounded_is_log1p_of_sums():
    adj = _adj(3, [(0, 1), (1, 2)])
    f = pair_feature(adj, 0, 2, d=1, mode="bounded")
    assert math.isclose(f, 2 * math.log(4.0))


def test_pair_feature_common_dhop_log_form():
    # Triangle: common neighbor count = 1; mode returns log(1 + 1)
    adj = _adj(3, [(0, 1), (1, 2), (0, 2)])
    assert math.isclose(
        pair_feature(adj, 0, 1, d=1, mode="common_dhop"),
        math.log(2.0),
    )


def test_pair_feature_unknown_mode_raises():
    adj = _adj(3, [(0, 1)])
    with pytest.raises(ValueError):
        pair_feature(adj, 0, 1, d=1, mode="bogus")


# -------------------------------------------------------------------
# pair_feature_layer2
# -------------------------------------------------------------------

def test_pair_feature_layer2_drops_edge_when_present():
    # Triangle; remove edge (0,1) → degrees of 0 and 1 drop
    adj = _adj(3, [(0, 1), (1, 2), (0, 2)])
    f_full = pair_feature(adj, 0, 1, d=1, mode="paper_raw")
    f_l2 = pair_feature_layer2(adj, 0, 1, d=1, mode="paper_raw")
    assert f_l2 < f_full


def test_pair_feature_layer2_unchanged_when_edge_absent():
    # No edge (0,2)
    adj = _adj(3, [(0, 1), (1, 2)])
    f_full = pair_feature(adj, 0, 2, d=1, mode="paper_raw")
    f_l2 = pair_feature_layer2(adj, 0, 2, d=1, mode="paper_raw")
    assert math.isclose(f_full, f_l2)


# -------------------------------------------------------------------
# precompute_vertex_sums
# -------------------------------------------------------------------

def test_precompute_vertex_sums_matches_per_vertex_calls():
    rng = np.random.default_rng(0)
    n = 8
    upper = rng.random((n, n)) < 0.3
    upper = np.triu(upper, k=1)
    adj = (upper | upper.T).astype(float)
    for d in [1, 2]:
        sums = precompute_vertex_sums(adj, d)
        for v in range(n):
            assert math.isclose(sums[v], sum_degree(adj, v, d))


# -------------------------------------------------------------------
# build_pair_dataset
# -------------------------------------------------------------------

def test_build_pair_dataset_shapes_and_label_invariant():
    adj = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.3, seed=1))
    offsets, labels = build_pair_dataset(adj, d=1, mode="bounded", layer2=True)
    # n choose 2 = 45 pairs
    assert offsets.shape == (45,)
    assert labels.shape == (45,)
    # Labels are binary and their sum equals the edge count
    assert set(np.unique(labels).tolist()).issubset({0, 1})
    assert int(labels.sum()) == int(adj.sum() / 2)


def test_build_pair_dataset_deterministic():
    adj = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.3, seed=42))
    o1, l1 = build_pair_dataset(adj, d=1, mode="bounded", layer2=True)
    o2, l2 = build_pair_dataset(adj, d=1, mode="bounded", layer2=True)
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(l1, l2)
