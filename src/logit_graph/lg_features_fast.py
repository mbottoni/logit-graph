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
    """Ball-degree sums per vertex for paper_raw / bounded (Layer-2 via -1 fixup)."""
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
def build_multi_d_pair_dataset_skip_rows(
    rows: list,
    n: int,
    d_values: np.ndarray,
    mode_code: int,
    alpha_gwesp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Layer-2 offsets for several ``d`` in one pass over upper-triangle pairs."""
    nd = d_values.shape[0]
    m = n * (n - 1) // 2
    offsets = np.empty((nd, m), dtype=np.float64)
    labels = np.empty(m, dtype=np.int8)
    use_vertex_cache = mode_code == 0 or mode_code == 1
    has_vcache = False
    vertex_cache = np.empty(n, dtype=np.float64)
    if use_vertex_cache and nd == 1:
        vertex_cache = _precompute_vertex_ball_sums_skip_rows(
            rows, n, int(d_values[0]), mode_code,
        )
        has_vcache = True
    k = 0
    for i in range(n):
        row_i = rows[i]
        for j in range(i + 1, n):
            had = _sorted_has(row_i, j)
            labels[k] = 1 if had else 0
            for di in range(nd):
                d = int(d_values[di])
                if has_vcache and di == 0:
                    offsets[di, k] = _layer2_offset_from_vertex_sums(
                        vertex_cache, i, j, had, mode_code,
                    )
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
    """Layer-2 offsets and edge labels for all upper-triangle pairs."""
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
    n_iter = draws.shape[0]
    for t in range(n_iter):
        i_raw = draws[t, 0]
        j_raw = draws[t, 1]
        u = draws[t, 2]
        i = int(i_raw)
        j = int(j_raw)
        if j >= i:
            j += 1
        had = _sorted_has(rows[i], j)
        feat = pair_feature_layer2_skip_rows(
            rows, i, j, d, n, mode_code, had, alpha_gwesp,
        )
        logit = sigma + alpha * beta * feat
        p = _expit(logit)
        new_val = u < p
        if new_val != had:
            if new_val:
                rows[i] = _sorted_add(rows[i], j)
                rows[j] = _sorted_add(rows[j], i)
                degrees[i] += 1
                degrees[j] += 1
            else:
                rows[i] = _sorted_remove(rows[i], j)
                rows[j] = _sorted_remove(rows[j], i)
                degrees[i] -= 1
                degrees[j] -= 1


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
        "n", "rows", "degrees", "edge_count",
        "d", "mode_code", "sigma", "alpha", "beta", "alpha_gwesp",
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

        self.degrees = np.asarray([r.shape[0] for r in self.rows], dtype=np.int32)
        self.edge_count = int(self.degrees.sum() // 2)

    def to_adjacency(self) -> np.ndarray:
        return adj_from_rows(self.rows, self.n)

    def run_from_draws(self, draws: np.ndarray) -> None:
        run_gibbs_numba(
            self.rows,
            self.degrees,
            draws,
            self.d,
            self.mode_code,
            self.sigma,
            self.alpha,
            self.beta,
            self.alpha_gwesp,
            self.n,
        )
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
    """Layer-2 pair dataset from CSR rows (no dense adjacency conversion)."""
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
    """Fast Layer-2 pair dataset via Numba CSR (matches ``build_pair_dataset``)."""
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
    """Fast multi-``d`` Layer-2 pair datasets sharing one CSR + label pass."""
    if rows is None:
        adj = np.asarray(graph, dtype=float)
        rows = rows_from_adj(adj)
    n = len(rows)
    d_arr = np.asarray(sorted(d_values), dtype=np.int32)
    offsets_2d, labels = build_multi_d_pair_dataset_skip_rows(
        rows, n, d_arr, MODE_TO_CODE[mode], alpha_gwesp,
    )
    offsets_by_d = {int(d): offsets_2d[i].copy() for i, d in enumerate(d_arr)}
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
