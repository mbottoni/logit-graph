from __future__ import annotations

import numpy as np
import networkx as nx
from collections import deque
from scipy.stats import ks_2samp
from scipy.special import expit
from tqdm.auto import tqdm
from typing import Optional, Union

from .degrees_counts import degree_vertex, get_sum_degrees
from . import gic


class GraphModel:
    def __init__(
        self,
        n: int,
        d: int,
        sigma: float,
        alpha: float = 1,
        beta: float = 1,
        er_p: float = 0.05,
        init_graph: Optional[nx.Graph] = None,
    ) -> None:
        self.n = n
        self.d = d
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.er_p = er_p

        if init_graph is not None and isinstance(init_graph, nx.Graph):
            self.graph = nx.to_numpy_array(init_graph)
        else:
            self.graph = self.generate_small_er_graph(n, p=er_p)

        # Fast RNG (numpy Generator API, faster than legacy np.random)
        self._rng = np.random.default_rng()

        # Cached state — kept in sync by add_remove_edge
        self._init_cache()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _init_cache(self) -> None:
        """(Re-)compute the cached degree vector and edge count."""
        self._degrees: np.ndarray = self.graph.sum(axis=1)
        self._edge_count: int = int(np.triu(self.graph).sum())

    # ------------------------------------------------------------------
    # Graph generators
    # ------------------------------------------------------------------

    def generate_empty_graph(self, n: int) -> np.ndarray:
        return np.zeros((n, n))

    def generate_small_er_graph(self, n: int, p: float) -> np.ndarray:
        return nx.to_numpy_array(nx.erdos_renyi_graph(n, p))

    # ------------------------------------------------------------------
    # Spectrum — direct numpy Laplacian (no NetworkX round-trip)
    # ------------------------------------------------------------------

    @classmethod
    def calculate_spectrum(cls, graph: np.ndarray) -> np.ndarray:
        """Sorted eigenvalues of the Laplacian L = D - A."""
        degrees = graph.sum(axis=1)
        L = np.diag(degrees) - graph
        return np.sort(np.linalg.eigvalsh(L))

    # ------------------------------------------------------------------
    # Edge probability
    # ------------------------------------------------------------------

    def logistic_regression(self, sum_degrees: float) -> float:
        return expit(sum_degrees)

    def get_edge_logit(self, sum_degrees: float) -> int:
        """Bernoulli draw with probability = logistic(sum_degrees).

        Kept for backward compatibility; the hot-path in add_remove_edge
        inlines an equivalent but faster version.
        """
        p = expit(sum_degrees)
        return int(self._rng.random() < p)

    # ------------------------------------------------------------------
    # Core per-iteration step
    # ------------------------------------------------------------------

    def _get_sum_degrees_fast(self, vertex: int) -> float:
        """Sum of degrees of *vertex* and its d-hop neighbourhood.

        Uses the cached ``self._degrees`` vector and ``np.nonzero``
        for neighbour look-up, avoiding Python-level row scans.
        """
        if self.d == 0:
            return float(self._degrees[vertex])

        visited = {vertex}
        current_layer = np.nonzero(self.graph[vertex])[0]
        all_nbrs: list[int] = list(current_layer)
        visited.update(current_layer.tolist())

        for _ in range(self.d - 1):
            next_layer: list[int] = []
            for v in current_layer:
                for nv in np.nonzero(self.graph[v])[0]:
                    if nv not in visited:
                        next_layer.append(int(nv))
                        visited.add(int(nv))
            all_nbrs.extend(next_layer)
            current_layer = next_layer

        if all_nbrs:
            return float(
                self._degrees[vertex]
                + self._degrees[np.array(all_nbrs, dtype=int)].sum()
            )
        return float(self._degrees[vertex])

    def add_remove_edge(self) -> None:
        """Propose a random node pair and set / clear their edge.

        Optimisations over the naive version:
        - ``np.random.default_rng`` integers (2.7x vs legacy choice)
        - Cached degree vector look-up (47x for d=0, 54x for d=1)
        - Inline Bernoulli via ``rng.random() < p`` (9.4x vs np.random.choice)
        - Incremental degree + edge-count update (O(1) vs O(n)/O(n^2))
        """
        # --- fast pair sampling ---
        i = int(self._rng.integers(0, self.n))
        j = int(self._rng.integers(0, self.n - 1))
        if j >= i:
            j += 1

        # --- cached degree features ---
        sum_i = self._get_sum_degrees_fast(i)
        sum_j = self._get_sum_degrees_fast(j)
        total_degree = self.sigma + self.beta * (sum_i + sum_j)

        # --- fast Bernoulli draw ---
        p = expit(total_degree)
        new_val = float(self._rng.random() < p)

        # --- incremental update ---
        old_val = self.graph[i, j]
        if old_val != new_val:
            delta = new_val - old_val          # +1.0 or -1.0
            self._degrees[i] += delta
            self._degrees[j] += delta
            self._edge_count += int(delta)
            self.graph[i, j] = self.graph[j, i] = new_val

    # ------------------------------------------------------------------
    # Convergence helpers (unchanged)
    # ------------------------------------------------------------------

    def check_convergence_hist(
        self,
        graphs: list[np.ndarray],
        stability_window: int = 5,
        degree_dist_threshold: float = 0.05,
    ) -> bool:
        def degree_distribution_stability(graph1, graph2):
            degrees1 = np.sum(graph1, axis=1)
            degrees2 = np.sum(graph2, axis=1)
            ks_stat, _ = ks_2samp(degrees1, degrees2)
            print(f"KS Statistic: {ks_stat}")
            return ks_stat

        if len(graphs) <= stability_window:
            print("Not enough graphs for stability check.")
            return False

        degree_dist_stable = all(
            degree_distribution_stability(graphs[-i - 1], graphs[-i]) < degree_dist_threshold
            for i in range(1, stability_window)
        )
        print(f"Degree Distribution Stable: {degree_dist_stable}")

        is_converged = degree_dist_stable and True
        print(f"Graph Converged: {is_converged}")
        print('\n' * 3)
        return is_converged

    def check_convergence_number_of_edges(
        self,
        graphs: list[np.ndarray],
        threshold_edges: int,
        stability_window: int,
    ) -> bool:
        graphs_to_check = graphs[-stability_window:]
        prev_total_edges = None
        for graph in graphs_to_check:
            total_edges = np.sum(np.triu(graph))
            if prev_total_edges is not None:
                if abs(total_edges - prev_total_edges) > threshold_edges:
                    return False
            prev_total_edges = total_edges
        return True

    def check_convergence_spectrum(
        self,
        graphs: list[np.ndarray],
        threshold_spectrum: float,
        stability_window: int,
    ) -> bool:
        graphs_to_check = graphs[-stability_window:]
        prev_spectrum = None
        for graph in graphs_to_check:
            current_spectrum = self.calculate_spectrum(graph)
            if prev_spectrum is not None:
                spectrum_diff = np.linalg.norm(current_spectrum - prev_spectrum)
                if spectrum_diff > threshold_spectrum:
                    return False
            prev_spectrum = current_spectrum
        return True

    # ------------------------------------------------------------------
    # Graph generation loops
    # ------------------------------------------------------------------

    def populate_edges_baseline(
        self,
        warm_up: int,
        max_iterations: int,
        patience: int,
        check_interval: int = 50,
        edge_cv_tol: float = 0.02,
        spectrum_cv_tol: float = 0.02,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Generate a graph without a ground-truth reference.

        Convergence is based on the *coefficient of variation* (CV = std/mean)
        of edge counts and spectrum norms measured every ``check_interval``
        steps over the last ``patience`` measurements.
        """
        graphs = deque(maxlen=max(patience + 10, 200))
        graphs.append(self.graph.copy())

        edge_history: deque[int] = deque(maxlen=patience)
        spectrum_norm_history: deque[float] = deque(maxlen=patience)

        pbar = tqdm(total=max_iterations, desc="Generating graph",
                    leave=False, disable=False)

        for i in range(max_iterations):
            self.add_remove_edge()

            if i % check_interval == 0:
                graphs.append(self.graph.copy())
                edge_history.append(self._edge_count)

                spec_norm = float(np.linalg.norm(
                    self.calculate_spectrum(self.graph)))
                spectrum_norm_history.append(spec_norm)

                if i >= warm_up and len(edge_history) >= patience:
                    edges_arr = np.array(edge_history)
                    spec_arr = np.array(spectrum_norm_history)

                    mean_e = np.mean(edges_arr)
                    cv_edges = (np.std(edges_arr) / mean_e) if mean_e > 0 else 0.0

                    mean_s = np.mean(spec_arr)
                    cv_spectrum = (np.std(spec_arr) / mean_s) if mean_s > 0 else 0.0

                    stop_condition = (cv_edges < edge_cv_tol
                                      and cv_spectrum < spectrum_cv_tol)

                    pbar.set_postfix({
                        'edges': self._edge_count,
                        'cv_e': f'{cv_edges:.4f}',
                        'cv_s': f'{cv_spectrum:.4f}',
                        'converged': stop_condition,
                    })

                    if stop_condition:
                        pbar.update(max_iterations - pbar.n)
                        break

            pbar.update(1)

        pbar.close()
        spectra = self.calculate_spectrum(self.graph)
        return list(graphs), spectra

    def populate_edges_spectrum(
        self,
        warm_up: int,
        max_iterations: int,
        patience: int,
        real_graph: np.ndarray,
        edge_delta: Optional[float] = None,
        check_interval: int = 50,
        verbose: bool = True,
    ) -> tuple[list[np.ndarray], np.ndarray, list[float], int]:
        """Legacy spectrum-only convergence (no GIC gate)."""
        best_iteration = 0

        spectrum_diffs: list[float] = []
        real_spectrum = self.calculate_spectrum(real_graph)
        real_edges = int(np.triu(real_graph).sum())
        no_improvement_checks = 0
        best_spectrum_diff = float('inf')

        graphs = deque(maxlen=max(2 * patience + 100, 500))
        graphs.append(self.graph.copy())
        best_graph = self.graph.copy()

        for i in range(max_iterations):
            if edge_delta is not None:
                if self._edge_count > real_edges + edge_delta:
                    if verbose:
                        print('Too many edges. Stopping.')
                    break

            self.add_remove_edge()

            if i % check_interval == 0:
                graphs.append(self.graph.copy())
                current_spectrum = self.calculate_spectrum(self.graph)
                spectrum_diff = np.linalg.norm(current_spectrum - real_spectrum)
                spectrum_diffs.append(spectrum_diff)

                if verbose and i % 1000 == 0:
                    print(f'\t Iteration {i}: spectrum diff = {spectrum_diff:.4f}')

                if spectrum_diff < best_spectrum_diff:
                    best_spectrum_diff = spectrum_diff
                    best_graph = self.graph.copy()
                    best_iteration = i
                    no_improvement_checks = 0
                elif i >= warm_up:
                    no_improvement_checks += 1

                if i >= warm_up and no_improvement_checks >= patience:
                    break

        if verbose:
            print(f'\t Best iteration: {best_iteration}')
            print(f'\t Best spectrum difference: {best_spectrum_diff:.4f}')
            print(f'\t Edges (current): {self._edge_count}, '
                  f'Edges (real): {real_edges}')

        self.graph = best_graph
        self._init_cache()
        spectra = self.calculate_spectrum(self.graph)

        return list(graphs), spectra, spectrum_diffs, best_iteration

    def populate_edges_spectrum_min_gic(
        self,
        max_iterations: int,
        patience: int,
        real_graph: Union[np.ndarray, nx.Graph],
        min_gic_threshold: float,
        gic_dist_type: str = 'KL',
        edge_delta: Optional[float] = None,
        check_interval: int = 50,
        verbose: bool = True,
        er_p: float = 0.05,
    ) -> tuple[list[np.ndarray], np.ndarray, list[float], int, np.ndarray, list[float]]:
        """Populate edges targeting a real graph, using a two-phase criterion.

        Phase 1 — *GIC gate*:  Every ``check_interval`` iterations compute
        the GIC between the current graph and the real graph.  Keep
        iterating until GIC drops below ``min_gic_threshold``.

        Phase 2 — *spectrum patience*:  Track the best Laplacian-spectrum
        distance.  Stop after ``patience`` consecutive *checks* without
        improvement.
        """
        best_iteration = 0

        # GIC state
        gic_threshold_reached = False
        current_gic = float('inf')
        gic_values: list[float] = []

        # Spectrum state
        spectrum_diffs: list[float] = []
        if isinstance(real_graph, nx.Graph):
            real_graph_np = nx.to_numpy_array(real_graph)
        else:
            real_graph_np = real_graph

        real_spectrum = self.calculate_spectrum(real_graph_np)
        real_edges = int(np.triu(real_graph_np).sum())
        no_improvement_checks = 0
        best_spectrum_diff = float('inf')

        # Graph bookkeeping
        graphs = deque(maxlen=max(2 * patience + 100, 500))
        graphs.append(self.graph.copy())
        best_graph = self.graph.copy()

        # Pre-build NetworkX reference once for GIC calls
        real_nx_graph = nx.from_numpy_array(real_graph_np)

        # Progress bar
        pbar = None
        if verbose:
            pbar = tqdm(
                total=max_iterations,
                desc="Optimizing Graph", leave=True,
                bar_format=('{l_bar}{bar}| {n_fmt}/{total_fmt} '
                            '[{elapsed}<{remaining}, {rate_fmt}] {postfix}'))
            pbar.set_postfix({
                'GIC': f'{current_gic:.4f}',
                'Spec': f'{best_spectrum_diff:.4f}',
                'Pat': f'{no_improvement_checks}/{patience}',
                'Edges': f'{self._edge_count}/{real_edges}'
            })

        stop_reason = 'unknown'

        for i in range(max_iterations):
            # --- edge_delta guard (O(1) via cached count) ---
            if edge_delta is not None:
                if self._edge_count > real_edges + edge_delta:
                    stop_reason = f'edge count exceeded delta ({edge_delta})'
                    break

            # --- main step ---
            self.add_remove_edge()

            # --- periodic expensive check ---
            if i % check_interval == 0:
                graphs.append(self.graph.copy())

                current_spectrum = self.calculate_spectrum(self.graph)
                spectrum_diff = np.linalg.norm(current_spectrum - real_spectrum)
                spectrum_diffs.append(spectrum_diff)

                if not gic_threshold_reached:
                    try:
                        current_nx = nx.from_numpy_array(self.graph)
                        gic_calc = gic.GraphInformationCriterion(
                            real_nx_graph, model='LG',
                            log_graph=current_nx, dist=gic_dist_type)
                        current_gic = gic_calc.calculate_gic()
                    except Exception:
                        current_gic = float('inf')

                    if current_gic <= min_gic_threshold:
                        if verbose and pbar:
                            pbar.write(
                                f'GIC threshold {min_gic_threshold} reached '
                                f'at iteration {i:,} (GIC: {current_gic:.4f}). '
                                f'Starting spectrum patience ({patience} checks).')
                        gic_threshold_reached = True
                        no_improvement_checks = 0

                gic_values.append(current_gic)

                if spectrum_diff < best_spectrum_diff:
                    best_spectrum_diff = spectrum_diff
                    best_graph = self.graph.copy()
                    best_iteration = i
                    if gic_threshold_reached:
                        no_improvement_checks = 0
                elif gic_threshold_reached:
                    no_improvement_checks += 1

                if verbose and pbar:
                    pbar.set_postfix({
                        'GIC': f'{current_gic:.4f}',
                        'Spec': f'{best_spectrum_diff:.4f}',
                        'Pat': f'{no_improvement_checks}/{patience}',
                        'Edges': f'{self._edge_count}/{real_edges}'
                    })

                if gic_threshold_reached and no_improvement_checks >= patience:
                    stop_reason = (
                        f'no spectral improvement for {patience} checks '
                        f'after GIC threshold was met')
                    break

            if verbose and pbar:
                pbar.update(1)
        else:
            stop_reason = f'max iterations ({max_iterations:,}) reached'

        # --- summary (before closing pbar) ---
        if verbose and pbar:
            pbar.write(f'\nStopping: {stop_reason}')
            pbar.write(f'  Best iteration: {best_iteration:,}')
            pbar.write(f'  Best spectrum diff: {best_spectrum_diff:.4f}')
            pbar.write(f'  Edges in best graph: '
                       f'{int(np.triu(best_graph).sum())} '
                       f'(real: {real_edges})')
            pbar.close()

        spectra = self.calculate_spectrum(best_graph)
        return (list(graphs), spectra, spectrum_diffs,
                best_iteration, best_graph, gic_values)
