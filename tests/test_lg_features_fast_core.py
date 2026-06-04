"""Unit tests for the low-level CSR/sorted primitives in lg_features_fast.

High-level feature/Gibbs equivalence is covered in test_gibbs_equivalence.py.
This file targets the building blocks: the Numba sorted-row helpers, the
adjacency<->neighbor<->CSR round-trips, the Gibbs draw generator, and
FastGibbsGraph determinism / bookkeeping.
"""
import numpy as np
import networkx as nx
import pytest

from logit_graph.lg_features_fast import (
    FastGibbsGraph,
    _sorted_add,
    _sorted_add_buf,
    _sorted_has,
    _sorted_has_buf,
    _sorted_remove,
    _sorted_remove_buf,
    adj_from_rows,
    density_from_rows,
    make_gibbs_draws,
    nbrs_from_adj,
    nbrs_to_csr,
    rows_from_adj,
)


def _sorted_row(values):
    return np.array(sorted(values), dtype=np.int32)


# -------------------------------------------------------------------
# _sorted_has — binary-search membership matches Python `in`
# -------------------------------------------------------------------

def test_sorted_has_matches_python_membership():
    present = {1, 4, 7, 9, 12, 20}
    row = _sorted_row(present)
    for x in range(25):
        assert _sorted_has(row, x) == (x in present)


def test_sorted_has_empty_row():
    row = np.empty(0, dtype=np.int32)
    assert not _sorted_has(row, 0)


# -------------------------------------------------------------------
# _sorted_add — insert (contract assumes x absent) keeps sorted
# -------------------------------------------------------------------

def test_sorted_add_keeps_sorted_and_grows_by_one():
    base = {2, 5, 8, 11}
    for x in (0, 6, 9, 99):  # all absent
        row = _sorted_row(base)
        out = _sorted_add(row, x)
        assert out.shape[0] == row.shape[0] + 1
        assert list(out) == sorted(base | {x})
        assert np.all(np.diff(out) > 0)  # strictly increasing


# -------------------------------------------------------------------
# _sorted_remove — drop element, no-op if absent
# -------------------------------------------------------------------

def test_sorted_remove_present_element():
    row = _sorted_row({3, 6, 9, 14})
    out = _sorted_remove(row, 9)
    assert list(out) == [3, 6, 14]


# Note: the array-variant _sorted_remove pre-sizes its output to n-1, so it
# assumes x is present (its only call sites guarantee this). The buffer variant
# below handles an absent element as a well-defined no-op.


# -------------------------------------------------------------------
# Buffer variants — in-place mutation with row_lens
# -------------------------------------------------------------------

def _make_buf(values, max_deg=16):
    row_buf = np.zeros((1, max_deg), dtype=np.int32)
    vals = sorted(values)
    row_buf[0, : len(vals)] = vals
    row_lens = np.array([len(vals)], dtype=np.int32)
    return row_buf, row_lens


def test_sorted_add_buf_inserts_and_keeps_sorted():
    row_buf, row_lens = _make_buf({2, 5, 8})
    _sorted_add_buf(row_buf, row_lens, 0, 6)
    ln = int(row_lens[0])
    assert ln == 4
    assert list(row_buf[0, :ln]) == [2, 5, 6, 8]


def test_sorted_add_buf_is_idempotent_for_existing_element():
    row_buf, row_lens = _make_buf({2, 5, 8})
    _sorted_add_buf(row_buf, row_lens, 0, 5)  # already present
    ln = int(row_lens[0])
    assert ln == 3
    assert list(row_buf[0, :ln]) == [2, 5, 8]


def test_sorted_remove_buf_removes_element():
    row_buf, row_lens = _make_buf({2, 5, 8})
    _sorted_remove_buf(row_buf, row_lens, 0, 5)
    ln = int(row_lens[0])
    assert ln == 2
    assert list(row_buf[0, :ln]) == [2, 8]


def test_sorted_remove_buf_absent_is_noop():
    row_buf, row_lens = _make_buf({2, 5, 8})
    _sorted_remove_buf(row_buf, row_lens, 0, 7)  # absent
    ln = int(row_lens[0])
    assert ln == 3
    assert list(row_buf[0, :ln]) == [2, 5, 8]


def test_sorted_has_buf_matches_membership():
    row_buf, row_lens = _make_buf({1, 4, 9})
    for x in range(12):
        assert _sorted_has_buf(row_buf, row_lens, 0, x) == (x in {1, 4, 9})


# -------------------------------------------------------------------
# nbrs_from_adj / nbrs_to_csr / rows_from_adj / adj_from_rows round-trips
# -------------------------------------------------------------------

def _adj(n, p, seed):
    return nx.to_numpy_array(nx.erdos_renyi_graph(n, p, seed=seed))


def test_nbrs_from_adj_matches_adjacency_and_excludes_self():
    adj = _adj(15, 0.3, seed=0)
    nbrs = nbrs_from_adj(adj)
    for i in range(adj.shape[0]):
        expected = {int(j) for j in np.nonzero(adj[i])[0] if j != i}
        assert nbrs[i] == expected
        assert i not in nbrs[i]


def test_nbrs_to_csr_is_sorted_with_correct_indptr():
    adj = _adj(12, 0.3, seed=1)
    nbrs = nbrs_from_adj(adj)
    indptr, indices = nbrs_to_csr(nbrs)
    n = len(nbrs)
    assert indptr.shape == (n + 1,)
    assert indptr[0] == 0
    assert indptr[-1] == len(indices)
    for v in range(n):
        row = indices[indptr[v]:indptr[v + 1]]
        assert list(row) == sorted(nbrs[v])  # CSR rows are sorted


def test_rows_adj_round_trip_is_identity():
    adj = _adj(14, 0.25, seed=2)
    rows = rows_from_adj(adj)
    recovered = adj_from_rows(rows, adj.shape[0])
    np.testing.assert_array_equal(recovered, adj)


# -------------------------------------------------------------------
# density_from_rows
# -------------------------------------------------------------------

def test_density_from_rows_matches_edge_density():
    adj = _adj(20, 0.2, seed=3)
    rows = rows_from_adj(adj)
    n = adj.shape[0]
    expected = adj.sum() / (n * (n - 1))  # directed-sum / n(n-1)
    assert density_from_rows(rows, n) == pytest.approx(expected)


def test_density_from_rows_single_node_is_zero():
    assert density_from_rows([np.empty(0, dtype=np.int32)], 1) == 0.0


# -------------------------------------------------------------------
# make_gibbs_draws — shape and value ranges
# -------------------------------------------------------------------

def test_make_gibbs_draws_shape_and_ranges():
    n, n_iter = 25, 500
    rng = np.random.default_rng(0)
    draws = make_gibbs_draws(n, n_iter, rng)
    assert draws.shape == (n_iter, 3)
    assert draws[:, 0].min() >= 0 and draws[:, 0].max() < n
    assert draws[:, 1].min() >= 0 and draws[:, 1].max() < n - 1
    assert draws[:, 2].min() >= 0.0 and draws[:, 2].max() < 1.0


def test_make_gibbs_draws_deterministic_under_seed():
    a = make_gibbs_draws(20, 100, np.random.default_rng(7))
    b = make_gibbs_draws(20, 100, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


# -------------------------------------------------------------------
# FastGibbsGraph — determinism and bookkeeping
# -------------------------------------------------------------------

def test_fast_gibbs_edge_count_matches_degrees():
    adj = _adj(30, 0.15, seed=4)
    fg = FastGibbsGraph(
        30, 1, -3.0, er_p=0.15, rng=np.random.default_rng(0),
        feature_mode="incremental", adj=adj,
    )
    assert fg.edge_count == int(fg.degrees.sum() // 2)
    assert fg.edge_count == int(adj.sum() // 2)


def test_fast_gibbs_same_draws_are_deterministic():
    adj = _adj(30, 0.15, seed=5)
    draws = make_gibbs_draws(30, 2000, np.random.default_rng(11))

    def _run():
        fg = FastGibbsGraph(
            30, 1, -3.0, er_p=0.15, rng=np.random.default_rng(0),
            feature_mode="incremental", adj=adj.copy(),
        )
        fg.run_from_draws(draws.copy())
        return fg.to_adjacency()

    np.testing.assert_array_equal(_run(), _run())


def test_fast_gibbs_to_adjacency_is_symmetric():
    adj = _adj(25, 0.2, seed=6)
    fg = FastGibbsGraph(
        25, 1, -2.5, er_p=0.2, rng=np.random.default_rng(0),
        feature_mode="bounded", adj=adj,
    )
    fg.run_steps(1000, np.random.default_rng(3))
    out = fg.to_adjacency()
    np.testing.assert_array_equal(out, out.T)
    assert np.all(np.diag(out) == 0)  # no self-loops
