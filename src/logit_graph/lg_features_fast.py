"""Fast Gibbs graph state: adjacency-only storage, incremental CSR, Numba features."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numba import njit

from .lg_features import ALPHA_GWESP_DEFAULT, FeatureMode

MODE_TO_CODE: dict[str, int] = {
    "paper_raw": 0,
    "bounded": 1,
    "incremental": 2,
    "common_dhop": 3,
}


# ---------------------------------------------------------------------------
# Numba CSR utilities
# ---------------------------------------------------------------------------

@njit(cache=True)
def _sorted_has(row: np.ndarray, x: int) -> bool:
    lo, hi = 0, row.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if row[mid] == x:
            return True
        if row[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return False


@njit(cache=True)
def _sorted_add(row: np.ndarray, x: int) -> np.ndarray:
    out = np.empty(row.shape[0] + 1, dtype=np.int32)
    inserted = False
    pos = 0
    for k in range(row.shape[0]):
        if not inserted and row[k] > x:
            out[pos] = x
            pos += 1
            inserted = True
        out[pos] = row[k]
        pos += 1
    if not inserted:
        out[pos] = x
    return out


@njit(cache=True)
def _sorted_remove(row: np.ndarray, x: int) -> np.ndarray:
    out = np.empty(max(0, row.shape[0] - 1), dtype=np.int32)
    pos = 0
    for k in range(row.shape[0]):
        if row[k] != x:
            out[pos] = row[k]
            pos += 1
    return out[:pos]


@njit(cache=True)
def _sorted_has_buf(row_buf: np.ndarray, row_lens: np.ndarray, v: int, x: int) -> bool:
    ln = int(row_lens[v])
    row = row_buf[v]
    lo, hi = 0, ln
    while lo < hi:
        mid = (lo + hi) // 2
        if row[mid] == x:
            return True
        if row[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return False


@njit(cache=True)
def _sorted_add_buf(
    row_buf: np.ndarray, row_lens: np.ndarray, v: int, x: int,
) -> None:
    ln = int(row_lens[v])
    row = row_buf[v]
    pos = 0
    while pos < ln and row[pos] < x:
        pos += 1
    if pos < ln and row[pos] == x:
        return
    for k in range(ln, pos, -1):
        row[k] = row[k - 1]
    row[pos] = x
    row_lens[v] = ln + 1


@njit(cache=True)
def _sorted_remove_buf(
    row_buf: np.ndarray, row_lens: np.ndarray, v: int, x: int,
) -> None:
    ln = int(row_lens[v])
    row = row_buf[v]
    pos = 0
    for k in range(ln):
        if row[k] != x:
            if pos != k:
                row[pos] = row[k]
            pos += 1
    row_lens[v] = pos


@njit(cache=True)
def _row_buf_to_rows_list(row_buf: np.ndarray, row_lens: np.ndarray, n: int) -> list:
    rows = []
    for v in range(n):
        ln = int(row_lens[v])
        out = np.empty(ln, dtype=np.int32)
        for k in range(ln):
            out[k] = row_buf[v, k]
        rows.append(out)
    return rows


@njit(cache=True)
def _rows_list_to_row_buf(rows: list, n: int, max_deg: int) -> tuple[np.ndarray, np.ndarray]:
    row_buf = np.zeros((n, max_deg), dtype=np.int32)
    row_lens = np.zeros(n, dtype=np.int32)
    for v in range(n):
        row = rows[v]
        ln = row.shape[0]
        row_lens[v] = ln
        for k in range(ln):
            row_buf[v, k] = row[k]
    return row_buf, row_lens


@njit(cache=True)
def _rebuild_csr(rows: list, n: int) -> tuple[np.ndarray, np.ndarray]:
    indptr = np.zeros(n + 1, dtype=np.int64)
    total = 0
    for v in range(n):
        indptr[v] = total
        total += rows[v].shape[0]
    indptr[n] = total
    indices = np.empty(total, dtype=np.int32)
    pos = 0
    for v in range(n):
        row = rows[v]
        for k in range(row.shape[0]):
            indices[pos] = row[k]
            pos += 1
    return indptr, indices


@njit(cache=True)
def _ball_mark_rows(
    rows: list,
    source: int,
    depth: int,
    n: int,
    mark: np.ndarray,
    l2_skip: bool,
    ei: int,
    ej: int,
) -> None:
    frontier = np.empty(n, dtype=np.int32)
    frontier_len = 1
    frontier[0] = source
    mark[source] = 1
    for _ in range(depth):
        nxt = np.empty(n, dtype=np.int32)
        nxt_len = 0
        for fi in range(frontier_len):
            v = frontier[fi]
            row = rows[v]
            for k in range(row.shape[0]):
                u = int(row[k])
                if l2_skip and ((v == ei and u == ej) or (v == ej and u == ei)):
                    continue
                if mark[u] == 0:
                    mark[u] = 1
                    nxt[nxt_len] = u
                    nxt_len += 1
        if nxt_len == 0:
            break
        frontier_len = nxt_len
        for k in range(nxt_len):
            frontier[k] = nxt[k]


@njit(cache=True)
def _effective_degree(
    rows: list, v: int, l2_skip: bool, ei: int, ej: int,
) -> float:
    deg = float(rows[v].shape[0])
    if l2_skip:
        if v == ei and _sorted_has(rows[ei], ej):
            return deg - 1.0
        if v == ej and _sorted_has(rows[ej], ei):
            return deg - 1.0
    return deg


@njit(cache=True)
def _sum_degree_marked_rows(
    rows: list, mark: np.ndarray, n: int, l2_skip: bool, ei: int, ej: int,
) -> float:
    total = 0.0
    for v in range(n):
        if mark[v] == 1:
            total += _effective_degree(rows, v, l2_skip, ei, ej)
    return total


@njit(cache=True)
def _common_neighbors_skip_rows(
    rows: list, i: int, j: int, l2_skip: bool,
) -> int:
    """Count common neighbors of ``i`` and ``j`` (exact ``d=1`` common-dhop)."""
    ri = rows[i]
    rj = rows[j]
    a = 0
    b = 0
    count = 0
    while a < ri.shape[0] and b < rj.shape[0]:
        vi = int(ri[a])
        vj = int(rj[b])
        if l2_skip:
            if vi == j:
                a += 1
                continue
            if vj == i:
                b += 1
                continue
        if vi == i or vi == j:
            a += 1
            continue
        if vj == i or vj == j:
            b += 1
            continue
        if vi == vj:
            count += 1
            a += 1
            b += 1
        elif vi < vj:
            a += 1
        else:
            b += 1
    return count


@njit(cache=True)
def _bfs_distances_skip_rows(
    rows: list,
    source: int,
    max_depth: int,
    n: int,
    dist: np.ndarray,
    l2_skip: bool,
    ei: int,
    ej: int,
) -> None:
    for v in range(n):
        dist[v] = -1
    dist[source] = 0
    frontier = np.empty(n, dtype=np.int32)
    frontier_len = 1
    frontier[0] = source
    head = 0
    while head < frontier_len:
        v = int(frontier[head])
        head += 1
        dv = int(dist[v])
        if dv >= max_depth:
            continue
        row = rows[v]
        for k in range(row.shape[0]):
            u = int(row[k])
            if l2_skip and ((v == ei and u == ej) or (v == ej and u == ei)):
                continue
            if dist[u] < 0:
                dist[u] = dv + 1
                frontier[frontier_len] = u
                frontier_len += 1


@njit(cache=True)
def _count_common_within_dist(
    dist_i: np.ndarray,
    dist_j: np.ndarray,
    n: int,
    i: int,
    j: int,
    max_d: int,
) -> int:
    count = 0
    for v in range(n):
        if v == i or v == j:
            continue
        di = int(dist_i[v])
        dj = int(dist_j[v])
        if di >= 0 and dj >= 0 and di <= max_d and dj <= max_d:
            count += 1
    return count


@njit(cache=True)
def _common_dhop_skip_rows(
    rows: list, i: int, j: int, depth: int, n: int, l2_skip: bool,
) -> int:
    if depth == 0:
        return 0
    if depth == 1:
        return _common_neighbors_skip_rows(rows, i, j, l2_skip)
    dist_i = np.empty(n, dtype=np.int16)
    dist_j = np.empty(n, dtype=np.int16)
    _bfs_distances_skip_rows(rows, i, depth, n, dist_i, l2_skip, i, j)
    _bfs_distances_skip_rows(rows, j, depth, n, dist_j, l2_skip, i, j)
    return _count_common_within_dist(dist_i, dist_j, n, i, j, depth)


@njit(cache=True)
def _incremental_h_skip_rows(
    rows: list, i: int, j: int, d: int, n: int, l2_skip: bool, alpha_gwesp: float,
) -> float:
    if d == 0:
        return 0.0
    if d == 1:
        c = _common_neighbors_skip_rows(rows, i, j, l2_skip)
        if c <= 0:
            return 0.0
        return alpha_gwesp * (1.0 - (1.0 - 1.0 / alpha_gwesp) ** c)
    dist_i = np.empty(n, dtype=np.int16)
    dist_j = np.empty(n, dtype=np.int16)
    _bfs_distances_skip_rows(rows, i, d, n, dist_i, l2_skip, i, j)
    _bfs_distances_skip_rows(rows, j, d, n, dist_j, l2_skip, i, j)
    c_d = _count_common_within_dist(dist_i, dist_j, n, i, j, d)
    c_dm1 = _count_common_within_dist(dist_i, dist_j, n, i, j, d - 1)
    delta = c_d - c_dm1
    if delta < 0:
        delta = 0
    return math.log(1.0 + delta)


@njit(cache=True)
def _sum_ball_degree_skip_rows(
    rows: list, source: int, depth: int, n: int, l2_skip: bool, ei: int, ej: int,
) -> float:
    mark = np.zeros(n, dtype=np.int8)
    _ball_mark_rows(rows, source, depth, n, mark, l2_skip, ei, ej)
    return _sum_degree_marked_rows(rows, mark, n, l2_skip, ei, ej)


@njit(cache=True)
def _precompute_vertex_ball_sums_skip_rows(
    rows: list, n: int, d: int, mode_code: int,
) -> np.ndarray:
    """Ball-degree sums per vertex for paper_raw / bounded (leave-one-out via -1 fixup)."""
    sums = np.empty(n, dtype=np.float64)
    for v in range(n):
        raw = _sum_ball_degree_skip_rows(rows, v, d, n, False, -1, -1)
        if mode_code == 0:
            sums[v] = raw
        else:
            sums[v] = math.log(1.0 + raw)
    return sums


@njit(cache=True)
def _layer2_offset_from_vertex_sums(
    vertex_sums: np.ndarray, i: int, j: int, had: bool, mode_code: int,
) -> float:
    si = vertex_sums[i]
    sj = vertex_sums[j]
    if had:
        si -= 1.0
        sj -= 1.0
    if mode_code == 0:
        return si + sj
    return si + sj


@njit(cache=True)
def run_gibbs_numba(
    rows: list,
    degrees: np.ndarray,
    draws: np.ndarray,
    d: int,
    mode_code: int,
    sigma: float,
    alpha: float,
    beta: float,
    alpha_gwesp: float,
    n: int,
) -> None:
    """Run all Gibbs steps in one compiled loop (``draws`` shape (n_iter, 3))."""
    max_deg = n - 1
    if max_deg < 1:
        max_deg = 1
    row_buf, row_lens = _rows_list_to_row_buf(rows, n, max_deg)
    run_gibbs_numba_buf(
        row_buf, row_lens, draws, d, mode_code, sigma, alpha, beta, alpha_gwesp, n,
    )
    for v in range(n):
        ln = int(row_lens[v])
        out = np.empty(ln, dtype=np.int32)
        for k in range(ln):
            out[k] = row_buf[v, k]
        rows[v] = out
        degrees[v] = ln


@njit(cache=True)
def build_multi_d_pair_dataset_skip_rows(
    rows: list,
    n: int,
    d_values: np.ndarray,
    mode_code: int,
    alpha_gwesp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """leave-one-out offsets for several ``d`` in one pass over upper-triangle pairs."""
    nd = d_values.shape[0]
    m = n * (n - 1) // 2
    offsets = np.empty((nd, m), dtype=np.float64)
    labels = np.empty(m, dtype=np.int8)
    use_vertex_cache = mode_code == 0 or mode_code == 1
    max_d_bfs = 0
    if mode_code == 2 or mode_code == 3:
        for di in range(nd):
            d_val = int(d_values[di])
            if d_val > max_d_bfs:
                max_d_bfs = d_val
    vertex_caches = np.empty((nd, n), dtype=np.float64)
    if use_vertex_cache:
        for di in range(nd):
            d = int(d_values[di])
            vc = _precompute_vertex_ball_sums_skip_rows(rows, n, d, mode_code)
            for v in range(n):
                vertex_caches[di, v] = vc[v]
    dist_i = np.empty(n, dtype=np.int16)
    dist_j = np.empty(n, dtype=np.int16)
    k = 0
    for i in range(n):
        row_i = rows[i]
        for j in range(i + 1, n):
            had = _sorted_has(row_i, j)
            labels[k] = 1 if had else 0
            l2 = had
            dist_ready = False
            for di in range(nd):
                d = int(d_values[di])
                if use_vertex_cache:
                    vc_row = vertex_caches[di]
                    offsets[di, k] = _layer2_offset_from_vertex_sums(
                        vc_row, i, j, had, mode_code,
                    )
                elif mode_code == 2 and d >= 2:
                    if not dist_ready:
                        _bfs_distances_skip_rows(
                            rows, i, max_d_bfs, n, dist_i, l2, i, j,
                        )
                        _bfs_distances_skip_rows(
                            rows, j, max_d_bfs, n, dist_j, l2, i, j,
                        )
                        dist_ready = True
                    c_d = _count_common_within_dist(dist_i, dist_j, n, i, j, d)
                    c_dm1 = _count_common_within_dist(dist_i, dist_j, n, i, j, d - 1)
                    delta = c_d - c_dm1
                    if delta < 0:
                        delta = 0
                    offsets[di, k] = math.log(1.0 + delta)
                elif mode_code == 3 and d >= 2:
                    if not dist_ready:
                        _bfs_distances_skip_rows(
                            rows, i, max_d_bfs, n, dist_i, l2, i, j,
                        )
                        _bfs_distances_skip_rows(
                            rows, j, max_d_bfs, n, dist_j, l2, i, j,
                        )
                        dist_ready = True
                    c = _count_common_within_dist(dist_i, dist_j, n, i, j, d)
                    offsets[di, k] = math.log(1.0 + c)
                elif mode_code == 2 and d == 1:
                    c = _common_neighbors_skip_rows(rows, i, j, l2)
                    if c <= 0:
                        offsets[di, k] = 0.0
                    else:
                        offsets[di, k] = alpha_gwesp * (
                            1.0 - (1.0 - 1.0 / alpha_gwesp) ** c
                        )
                elif mode_code == 3 and d == 1:
                    c = _common_neighbors_skip_rows(rows, i, j, l2)
                    offsets[di, k] = math.log(1.0 + c)
                elif d == 0:
                    offsets[di, k] = 0.0
                else:
                    offsets[di, k] = pair_feature_layer2_skip_rows(
                        rows, i, j, d, n, mode_code, had, alpha_gwesp,
                    )
            k += 1
    return offsets, labels


@njit(cache=True)
def build_pair_dataset_skip_rows(
    rows: list,
    n: int,
    d: int,
    mode_code: int,
    alpha_gwesp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """leave-one-out offsets and edge labels for all upper-triangle pairs."""
    d_values = np.array([d], dtype=np.int32)
    offsets_2d, labels = build_multi_d_pair_dataset_skip_rows(
        rows, n, d_values, mode_code, alpha_gwesp,
    )
    return offsets_2d[0], labels


@njit(cache=True)
def pair_feature_layer2_skip_rows(
    rows: list, i: int, j: int, d: int, n: int, mode_code: int,
    had_edge: bool, alpha_gwesp: float,
) -> float:
    l2 = had_edge
    if mode_code == 0:
        si = _sum_ball_degree_skip_rows(rows, i, d, n, l2, i, j)
        sj = _sum_ball_degree_skip_rows(rows, j, d, n, l2, i, j)
        return si + sj
    if mode_code == 1:
        si = _sum_ball_degree_skip_rows(rows, i, d, n, l2, i, j)
        sj = _sum_ball_degree_skip_rows(rows, j, d, n, l2, i, j)
        return math.log(1.0 + si) + math.log(1.0 + sj)
    if mode_code == 2:
        return _incremental_h_skip_rows(rows, i, j, d, n, l2, alpha_gwesp)
    c = _common_dhop_skip_rows(rows, i, j, d, n, l2)
    return math.log(1.0 + c)


@njit(cache=True)
def _expit(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@njit(cache=True)
def _common_neighbors_buf(
    row_buf: np.ndarray, row_lens: np.ndarray, i: int, j: int, l2_skip: bool,
) -> int:
    ri_len = int(row_lens[i])
    rj_len = int(row_lens[j])
    ri = row_buf[i]
    rj = row_buf[j]
    a = 0
    b = 0
    count = 0
    while a < ri_len and b < rj_len:
        vi = int(ri[a])
        vj = int(rj[b])
        if l2_skip:
            if vi == j:
                a += 1
                continue
            if vj == i:
                b += 1
                continue
        if vi == i or vi == j:
            a += 1
            continue
        if vj == i or vj == j:
            b += 1
            continue
        if vi == vj:
            count += 1
            a += 1
            b += 1
        elif vi < vj:
            a += 1
        else:
            b += 1
    return count


@njit(cache=True)
def _bfs_distances_buf(
    row_buf: np.ndarray,
    row_lens: np.ndarray,
    source: int,
    max_depth: int,
    n: int,
    dist: np.ndarray,
    frontier: np.ndarray,
    l2_skip: bool,
    ei: int,
    ej: int,
) -> int:
    # Init dist only for the previous frontier — but we don't know it here,
    # so reset all n entries (caller-owned buffer reused across steps).
    for v in range(n):
        dist[v] = -1
    dist[source] = 0
    frontier[0] = source
    frontier_len = 1
    head = 0
    while head < frontier_len:
        v = int(frontier[head])
        head += 1
        dv = int(dist[v])
        if dv >= max_depth:
            continue
        ln = int(row_lens[v])
        row = row_buf[v]
        for k in range(ln):
            u = int(row[k])
            if l2_skip and ((v == ei and u == ej) or (v == ej and u == ei)):
                continue
            if dist[u] < 0:
                dist[u] = dv + 1
                frontier[frontier_len] = u
                frontier_len += 1
    return frontier_len


@njit(cache=True)
def _intersect_both_d(
    small_frontier: np.ndarray,
    small_len: int,
    small_dist: np.ndarray,
    other_dist: np.ndarray,
    i: int,
    j: int,
    d: int,
) -> tuple:
    """O(ball) intersection: count |B_d(i) ∩ B_d(j) \\ {i,j}| and same for d-1,
    iterating the smaller of the two BFS frontiers instead of all n nodes.
    Returns (c_d, c_dm1)."""
    c_d = 0
    c_dm1 = 0
    dm1 = d - 1
    for k in range(small_len):
        v = int(small_frontier[k])
        if v == i or v == j:
            continue
        ov = int(other_dist[v])
        if ov < 0 or ov > d:
            continue
        # v is reachable from `source` within depth d (it's in the frontier).
        # Check membership in d-ball of other vertex.
        c_d += 1
        sv = int(small_dist[v])
        if sv <= dm1 and ov <= dm1:
            c_dm1 += 1
    return c_d, c_dm1


@njit(cache=True)
def _sum_ball_degree_buf(
    row_buf: np.ndarray, row_lens: np.ndarray, source: int, depth: int, n: int,
    l2_skip: bool, ei: int, ej: int,
) -> float:
    mark = np.zeros(n, dtype=np.int8)
    frontier = np.empty(n, dtype=np.int32)
    frontier_len = 1
    frontier[0] = source
    mark[source] = 1
    for _ in range(depth):
        nxt = np.empty(n, dtype=np.int32)
        nxt_len = 0
        for fi in range(frontier_len):
            v = frontier[fi]
            ln = int(row_lens[v])
            row = row_buf[v]
            for k in range(ln):
                u = int(row[k])
                if l2_skip and ((v == ei and u == ej) or (v == ej and u == ei)):
                    continue
                if mark[u] == 0:
                    mark[u] = 1
                    nxt[nxt_len] = u
                    nxt_len += 1
        if nxt_len == 0:
            break
        frontier_len = nxt_len
        for k in range(nxt_len):
            frontier[k] = nxt[k]
    total = 0.0
    for v in range(n):
        if mark[v] == 1:
            deg = float(row_lens[v])
            if l2_skip:
                if v == ei and _sorted_has_buf(row_buf, row_lens, ei, ej):
                    deg -= 1.0
                if v == ej and _sorted_has_buf(row_buf, row_lens, ej, ei):
                    deg -= 1.0
            total += deg
    return total


@njit(cache=True)
def _incremental_h_buf(
    row_buf: np.ndarray, row_lens: np.ndarray, i: int, j: int, d: int, n: int,
    l2_skip: bool, alpha_gwesp: float,
    dist_i: np.ndarray, dist_j: np.ndarray,
    frontier_i: np.ndarray, frontier_j: np.ndarray,
) -> float:
    if d == 0:
        return 0.0
    if d == 1:
        c = _common_neighbors_buf(row_buf, row_lens, i, j, l2_skip)
        if c <= 0:
            return 0.0
        return alpha_gwesp * (1.0 - (1.0 - 1.0 / alpha_gwesp) ** c)
    # Fast path: if either endpoint has no neighbors, the d-ball collapses to {v}
    # and any intersection (excluding i,j) is empty → feature is 0.
    if row_lens[i] == 0 or row_lens[j] == 0:
        return 0.0
    fl_i = _bfs_distances_buf(
        row_buf, row_lens, i, d, n, dist_i, frontier_i, l2_skip, i, j,
    )
    fl_j = _bfs_distances_buf(
        row_buf, row_lens, j, d, n, dist_j, frontier_j, l2_skip, i, j,
    )
    # Iterate over the smaller ball for O(min(|B_i|, |B_j|)) intersection.
    if fl_i <= fl_j:
        c_d, c_dm1 = _intersect_both_d(
            frontier_i, fl_i, dist_i, dist_j, i, j, d,
        )
    else:
        c_d, c_dm1 = _intersect_both_d(
            frontier_j, fl_j, dist_j, dist_i, i, j, d,
        )
    if c_d == 0:
        return 0.0
    delta = c_d - c_dm1
    if delta < 0:
        delta = 0
    return math.log(1.0 + delta)


@njit(cache=True)
def _common_dhop_buf(
    row_buf: np.ndarray, row_lens: np.ndarray, i: int, j: int, depth: int, n: int,
    l2_skip: bool,
    dist_i: np.ndarray, dist_j: np.ndarray,
    frontier_i: np.ndarray, frontier_j: np.ndarray,
) -> int:
    if depth == 0:
        return 0
    if depth == 1:
        return _common_neighbors_buf(row_buf, row_lens, i, j, l2_skip)
    if row_lens[i] == 0 or row_lens[j] == 0:
        return 0
    fl_i = _bfs_distances_buf(
        row_buf, row_lens, i, depth, n, dist_i, frontier_i, l2_skip, i, j,
    )
    fl_j = _bfs_distances_buf(
        row_buf, row_lens, j, depth, n, dist_j, frontier_j, l2_skip, i, j,
    )
    if fl_i <= fl_j:
        c_d, _c_dm1 = _intersect_both_d(
            frontier_i, fl_i, dist_i, dist_j, i, j, depth,
        )
    else:
        c_d, _c_dm1 = _intersect_both_d(
            frontier_j, fl_j, dist_j, dist_i, i, j, depth,
        )
    return c_d


@njit(cache=True)
def pair_feature_layer2_row_buf(
    row_buf: np.ndarray, row_lens: np.ndarray, i: int, j: int, d: int, n: int,
    mode_code: int, had_edge: bool, alpha_gwesp: float,
    dist_i: np.ndarray, dist_j: np.ndarray,
    frontier_i: np.ndarray, frontier_j: np.ndarray,
) -> float:
    l2 = had_edge
    if mode_code == 0:
        si = _sum_ball_degree_buf(row_buf, row_lens, i, d, n, l2, i, j)
        sj = _sum_ball_degree_buf(row_buf, row_lens, j, d, n, l2, i, j)
        return si + sj
    if mode_code == 1:
        si = _sum_ball_degree_buf(row_buf, row_lens, i, d, n, l2, i, j)
        sj = _sum_ball_degree_buf(row_buf, row_lens, j, d, n, l2, i, j)
        return math.log(1.0 + si) + math.log(1.0 + sj)
    if mode_code == 2:
        return _incremental_h_buf(
            row_buf, row_lens, i, j, d, n, l2, alpha_gwesp,
            dist_i, dist_j, frontier_i, frontier_j,
        )
    c = _common_dhop_buf(
        row_buf, row_lens, i, j, d, n, l2,
        dist_i, dist_j, frontier_i, frontier_j,
    )
    return math.log(1.0 + c)


@njit(cache=True)
def run_gibbs_numba_buf(
    row_buf: np.ndarray,
    row_lens: np.ndarray,
    draws: np.ndarray,
    d: int,
    mode_code: int,
    sigma: float,
    alpha: float,
    beta: float,
    alpha_gwesp: float,
    n: int,
) -> None:
    """Run Gibbs steps with in-place CSR row buffers (no per-flip allocations). BFS
    scratch (dist + frontier for i,j) is allocated once here and reused across all
    n_iter steps, avoiding ~4*n_iter Numba allocations that dominate at large sparse n."""
    n_iter = draws.shape[0]
    # Pre-allocate BFS scratch space once per chain.
    dist_i = np.empty(n, dtype=np.int16)
    dist_j = np.empty(n, dtype=np.int16)
    frontier_i = np.empty(n, dtype=np.int32)
    frontier_j = np.empty(n, dtype=np.int32)
    for t in range(n_iter):
        i_raw = draws[t, 0]
        j_raw = draws[t, 1]
        u = draws[t, 2]
        i = int(i_raw)
        j = int(j_raw)
        if j >= i:
            j += 1
        had = _sorted_has_buf(row_buf, row_lens, i, j)
        feat = pair_feature_layer2_row_buf(
            row_buf, row_lens, i, j, d, n, mode_code, had, alpha_gwesp,
            dist_i, dist_j, frontier_i, frontier_j,
        )
        logit = sigma + alpha * beta * feat
        p = _expit(logit)
        new_val = u < p
        if new_val != had:
            if new_val:
                _sorted_add_buf(row_buf, row_lens, i, j)
                _sorted_add_buf(row_buf, row_lens, j, i)
            else:
                _sorted_remove_buf(row_buf, row_lens, i, j)
                _sorted_remove_buf(row_buf, row_lens, j, i)


# ---------------------------------------------------------------------------
# Python wrappers (legacy nbrs path — kept for equivalence tests)
# ---------------------------------------------------------------------------

def nbrs_from_adj(adj: np.ndarray) -> list[set[int]]:
    n = adj.shape[0]
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in np.nonzero(adj[i])[0]:
            j = int(j)
            if j != i:
                nbrs[i].add(j)
    return nbrs


def nbrs_to_csr(nbrs: list[set[int]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(nbrs)
    indptr = np.zeros(n + 1, dtype=np.int64)
    parts: list[int] = []
    for v in range(n):
        indptr[v] = len(parts)
        parts.extend(sorted(nbrs[v]))
    indptr[n] = len(parts)
    return indptr, np.asarray(parts, dtype=np.int32)


def ball_vertices_nbrs(nbrs: list[set[int]], v: int, d: int) -> set[int]:
    if d == 0:
        return {v}
    visited: set[int] = {v}
    current = [v]
    for _ in range(d):
        nxt: list[int] = []
        for u in current:
            for nv in nbrs[u]:
                if nv not in visited:
                    visited.add(nv)
                    nxt.append(nv)
        current = nxt
        if not current:
            break
    return visited


def sum_degree_nbrs(nbrs: list[set[int]], vertex: int, d: int) -> float:
    deg_v = len(nbrs[vertex])
    if d == 0:
        return float(deg_v)
    visited: set[int] = {vertex}
    total = float(deg_v)
    current = [vertex]
    for _ in range(d):
        nxt: list[int] = []
        for u in current:
            for nv in nbrs[u]:
                if nv not in visited:
                    visited.add(nv)
                    nxt.append(nv)
                    total += float(len(nbrs[nv]))
        current = nxt
        if not current:
            break
    return total


def common_dhop_nbrs(nbrs: list[set[int]], i: int, j: int, d: int) -> int:
    if d == 0:
        return 0
    bi = ball_vertices_nbrs(nbrs, i, d)
    bj = ball_vertices_nbrs(nbrs, j, d)
    inter = bi & bj
    inter.discard(i)
    inter.discard(j)
    return len(inter)


def incremental_h_nbrs(
    nbrs: list[set[int]], i: int, j: int, d: int,
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
) -> float:
    if d == 0:
        return 0.0
    if d == 1:
        c = common_dhop_nbrs(nbrs, i, j, 1)
        if c <= 0:
            return 0.0
        return alpha_gwesp * (1.0 - (1.0 - 1.0 / alpha_gwesp) ** c)
    delta = common_dhop_nbrs(nbrs, i, j, d) - common_dhop_nbrs(nbrs, i, j, d - 1)
    return math.log(1.0 + max(0, delta))


def pair_feature_nbrs(
    nbrs: list[set[int]], i: int, j: int, d: int,
    mode: FeatureMode = "bounded",
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
) -> float:
    if mode == "paper_raw":
        return sum_degree_nbrs(nbrs, i, d) + sum_degree_nbrs(nbrs, j, d)
    if mode == "bounded":
        si = sum_degree_nbrs(nbrs, i, d)
        sj = sum_degree_nbrs(nbrs, j, d)
        return math.log(1.0 + si) + math.log(1.0 + sj)
    if mode == "incremental":
        return incremental_h_nbrs(nbrs, i, j, d, alpha_gwesp=alpha_gwesp)
    if mode == "common_dhop":
        return math.log(1.0 + common_dhop_nbrs(nbrs, i, j, d))
    raise ValueError(f"Unknown feature mode: {mode!r}")


def pair_feature_layer2_nbrs(
    nbrs: list[set[int]], i: int, j: int, d: int,
    mode: FeatureMode = "bounded",
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
) -> float:
    had = j in nbrs[i]
    if had:
        nbrs[i].discard(j)
        nbrs[j].discard(i)
    try:
        return pair_feature_nbrs(nbrs, i, j, d, mode=mode, alpha_gwesp=alpha_gwesp)
    finally:
        if had:
            nbrs[i].add(j)
            nbrs[j].add(i)


def rows_from_adj(adj: np.ndarray) -> list:
    n = adj.shape[0]
    rows: list = []
    for i in range(n):
        nbs = np.asarray([int(j) for j in np.nonzero(adj[i])[0] if j != i], dtype=np.int32)
        rows.append(np.sort(nbs))
    return rows


def adj_from_rows(rows: list, n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in rows[i]:
            adj[i, j] = 1.0
    return adj


class FastGibbsGraph:
    """Adjacency-only Gibbs state with incremental CSR + Numba features."""

    __slots__ = (
        "n", "rows", "row_buf", "row_lens", "degrees", "edge_count",
        "d", "mode_code", "sigma", "alpha", "beta", "alpha_gwesp", "_max_deg",
    )

    def __init__(
        self,
        n: int,
        d: int,
        sigma: float,
        *,
        er_p: float,
        rng: np.random.Generator,
        feature_mode: FeatureMode = "incremental",
        alpha: float = 1.0,
        beta: float = 1.0,
        alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
        adj: Optional[np.ndarray] = None,
    ) -> None:
        self.n = n
        self.d = d
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.alpha_gwesp = float(alpha_gwesp)
        self.mode_code = MODE_TO_CODE[feature_mode]
        self._max_deg = max(n - 1, 1)

        if adj is not None:
            self.rows = rows_from_adj(adj)
        else:
            upper = rng.random((n, n)) < er_p
            self.rows = []
            for i in range(n):
                nbs: list[int] = []
                for j in range(i + 1, n):
                    if upper[i, j]:
                        nbs.append(j)
                for j in range(i):
                    if upper[j, i]:
                        nbs.append(j)
                self.rows.append(np.sort(np.asarray(nbs, dtype=np.int32)))

        self.row_buf, self.row_lens = _rows_list_to_row_buf(
            self.rows, n, self._max_deg,
        )
        self.degrees = self.row_lens.copy()
        self.edge_count = int(self.degrees.sum() // 2)

    def to_rows_list(self) -> list:
        return _row_buf_to_rows_list(self.row_buf, self.row_lens, self.n)

    def to_adjacency(self) -> np.ndarray:
        return adj_from_rows(self.to_rows_list(), self.n)

    def run_from_draws(self, draws: np.ndarray) -> None:
        run_gibbs_numba_buf(
            self.row_buf,
            self.row_lens,
            draws,
            self.d,
            self.mode_code,
            self.sigma,
            self.alpha,
            self.beta,
            self.alpha_gwesp,
            self.n,
        )
        self.degrees = self.row_lens.copy()
        self.rows = self.to_rows_list()
        self.edge_count = int(self.degrees.sum() // 2)

    def run_steps(self, n_iter: int, rng: np.random.Generator) -> None:
        draws = make_gibbs_draws(self.n, n_iter, rng)
        self.run_from_draws(draws)


def make_gibbs_draws(n: int, n_iter: int, rng: np.random.Generator) -> np.ndarray:
    """Precompute Gibbs randomness (vectorized; ``run_gibbs_numba`` applies ``j`` fixup)."""
    draws = np.empty((n_iter, 3), dtype=np.float64)
    draws[:, 0] = rng.integers(0, n, size=n_iter)
    draws[:, 1] = rng.integers(0, n - 1, size=n_iter)
    draws[:, 2] = rng.random(n_iter)
    return draws


def build_pair_dataset_from_rows(
    rows: list,
    d: int,
    mode: FeatureMode = "incremental",
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """leave-one-out pair dataset from CSR rows (no dense adjacency conversion)."""
    n = len(rows)
    offsets, labels = build_pair_dataset_skip_rows(
        rows, n, d, MODE_TO_CODE[mode], alpha_gwesp,
    )
    return offsets, labels.astype(int)


def build_pair_dataset_fast(
    graph: np.ndarray,
    d: int,
    mode: FeatureMode = "incremental",
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
    *,
    rows: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast leave-one-out pair dataset via Numba CSR (matches ``build_pair_dataset``)."""
    if rows is None:
        adj = np.asarray(graph, dtype=float)
        rows = rows_from_adj(adj)
    return build_pair_dataset_from_rows(rows, d, mode=mode, alpha_gwesp=alpha_gwesp)


def build_multi_d_pair_datasets_fast(
    graph: np.ndarray,
    d_values: list[int],
    mode: FeatureMode = "incremental",
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
    *,
    rows: Optional[list] = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Fast multi-``d`` leave-one-out pair datasets sharing one CSR + label pass."""
    if rows is None:
        adj = np.asarray(graph, dtype=float)
        rows = rows_from_adj(adj)
    n = len(rows)
    d_arr = np.asarray(sorted(d_values), dtype=np.int32)
    offsets_2d, labels = build_multi_d_pair_dataset_skip_rows(
        rows, n, d_arr, MODE_TO_CODE[mode], alpha_gwesp,
    )
    offsets_by_d = {int(d): offsets_2d[i] for i, d in enumerate(d_arr)}
    return labels.astype(int), offsets_by_d


def density_from_rows(rows: list, n: int) -> float:
    if n <= 1:
        return 0.0
    return float(sum(int(r.shape[0]) for r in rows) / (n * (n - 1)))


def pair_feature_layer2_csr_py(
    nbrs: list[set[int]], i: int, j: int, d: int,
    mode: FeatureMode = "bounded",
    alpha_gwesp: float = ALPHA_GWESP_DEFAULT,
) -> float:
    n = len(nbrs)
    row_list = [np.asarray(sorted(nbrs[v]), dtype=np.int32) for v in range(n)]
    had = j in nbrs[i]
    return float(
        pair_feature_layer2_skip_rows(
            row_list, i, j, d, n, MODE_TO_CODE[mode], had, alpha_gwesp,
        )
    )
