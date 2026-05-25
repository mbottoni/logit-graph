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
def _common_dhop_skip_rows(
    rows: list, i: int, j: int, depth: int, n: int, l2_skip: bool,
) -> int:
    if depth == 0:
        return 0
    mark_i = np.zeros(n, dtype=np.int8)
    mark_j = np.zeros(n, dtype=np.int8)
    _ball_mark_rows(rows, i, depth, n, mark_i, l2_skip, i, j)
    _ball_mark_rows(rows, j, depth, n, mark_j, l2_skip, i, j)
    count = 0
    for v in range(n):
        if v == i or v == j:
            continue
        if mark_i[v] == 1 and mark_j[v] == 1:
            count += 1
    return count


@njit(cache=True)
def _incremental_h_skip_rows(
    rows: list, i: int, j: int, d: int, n: int, l2_skip: bool, alpha_gwesp: float,
) -> float:
    if d == 0:
        return 0.0
    if d == 1:
        c = _common_dhop_skip_rows(rows, i, j, 1, n, l2_skip)
        if c <= 0:
            return 0.0
        return alpha_gwesp * (1.0 - (1.0 - 1.0 / alpha_gwesp) ** c)
    c_d = _common_dhop_skip_rows(rows, i, j, d, n, l2_skip)
    c_dm1 = _common_dhop_skip_rows(rows, i, j, d - 1, n, l2_skip)
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
    draws = np.empty((n_iter, 3), dtype=np.float64)
    for t in range(n_iter):
        draws[t, 0] = float(rng.integers(0, n))
        draws[t, 1] = float(rng.integers(0, n - 1))
        draws[t, 2] = float(rng.random())
    return draws


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
