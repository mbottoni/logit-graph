import numpy as np
import networkx as nx
from collections import deque
from scipy.stats import ks_2samp
from scipy.special import expit
from tqdm.auto import tqdm

from .degrees_counts import degree_vertex, get_sum_degrees
from . import gic

class GraphModel:
    def __init__(self, n, d, sigma, alpha=1, beta=1, er_p=0.05, init_graph=None):
        self.n = n # number of nodes
        self.d = d # number of neighbors to consider 
        self.sigma = sigma # Offset weights
        self.alpha = alpha # weights on i node
        self.beta = beta   # weights on j node
        #self.graph = self.generate_empty_graph(n)
        self.er_p = er_p
        # If an initial NetworkX graph is provided, start from it; otherwise use ER seed
        if init_graph is not None and isinstance(init_graph, nx.Graph):
            self.graph = nx.to_numpy_array(init_graph)
        else:
            self.graph = self.generate_small_er_graph(n, p=er_p)

    def generate_empty_graph(self, n):
        return np.zeros((n, n))
    
    def generate_small_er_graph(self, n, p):
        # return the numpy array of the graph
        return nx.to_numpy_array(nx.erdos_renyi_graph(n, p))
    
    @classmethod
    def calculate_spectrum(cls, graph):
        G = nx.from_numpy_array(graph)
        eigenvalues = nx.laplacian_spectrum(G)
        return np.sort(eigenvalues)

    def logistic_regression(self, sum_degrees):
        return expit(sum_degrees)

    def get_edge_logit(self, sum_degrees):
        val_log = self.logistic_regression(sum_degrees)
        # Randomly choose 1 or 0 based on the probability
        random_choice = np.random.choice([1, 0], p=[val_log, 1 - val_log])

        #if val_log > 0.5:
        #    random_choice = 1
        #else:
        #    random_choice = 0

        return random_choice

    def add_remove_edge(self):
        i, j = np.random.choice(self.n, 2, replace=False)
        sum_i = get_sum_degrees(self.graph, vertex=i, d=self.d)
        sum_j = get_sum_degrees(self.graph, vertex=j, d=self.d)
        # Symmetric formulation: P(edge i,j) = logistic(sigma + beta * (S_i + S_j))
        total_degree = self.sigma + self.beta * (sum_i + sum_j)
        self.graph[j, i] = self.graph[i, j] = self.get_edge_logit(total_degree)

    def check_convergence_hist(self, graphs, stability_window=5, degree_dist_threshold=0.05):
        def degree_distribution_stability(graph1, graph2):
            # Calculate degree sequences
            degrees1 = np.sum(graph1, axis=1)
            degrees2 = np.sum(graph2, axis=1)
            # Compute the Kolmogorov-Smirnov statistic
            ks_stat, _ = ks_2samp(degrees1, degrees2)
            print(f"KS Statistic: {ks_stat}")
            return ks_stat

        if len(graphs) <= stability_window:
            print("Not enough graphs for stability check.")
            return False

        # Degree Distribution Stability: Check if the KS distance between degree distributions is below the threshold
        degree_dist_stable = all(
            degree_distribution_stability(graphs[-i - 1], graphs[-i]) < degree_dist_threshold
            for i in range(1, stability_window)
        )
        print(f"Degree Distribution Stable: {degree_dist_stable}")

        # Final convergence check
        is_converged = degree_dist_stable and True
        print(f"Graph Converged: {is_converged}")
        print('\n'*3)
        return is_converged

    def check_convergence_number_of_edges(self, graphs, threshold_edges, stability_window):
        # Check only the last stability_window graphs
        graphs_to_check = graphs[-stability_window:]
        
        prev_total_edges = None
        for graph in graphs_to_check:
            total_edges = np.sum(np.triu(graph))
            if prev_total_edges is not None:
                if abs(total_edges - prev_total_edges) > threshold_edges:
                    return False
            prev_total_edges = total_edges
        return True

    def check_convergence_spectrum(self, graphs, threshold_spectrum, stability_window):
        # Check only the last stability_window graphs
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

    def populate_edges_baseline(self, warm_up, max_iterations, patience,
                                check_interval=50, edge_cv_tol=0.02,
                                spectrum_cv_tol=0.02):
        """Generate a graph without a ground-truth reference.

        Convergence is based on the *coefficient of variation* (CV = std/mean)
        of edge counts and spectrum norms measured every ``check_interval``
        steps over the last ``patience`` measurements.  Both CVs must drop
        below their respective tolerances for convergence to be declared.

        Args:
            warm_up (int): Minimum iterations before convergence checks begin.
            max_iterations (int): Hard upper-bound on iterations.
            patience (int): Number of *measurements* (not iterations) in the
                rolling window used for the CV convergence test.
            check_interval (int): How often (in iterations) to record a
                measurement and test for convergence.  Spectrum is only
                computed at these checkpoints, keeping cost manageable.
            edge_cv_tol (float): CV threshold for edge-count stability.
            spectrum_cv_tol (float): CV threshold for spectrum-norm stability.
        """
        graphs = deque(maxlen=max(patience + 10, 200))
        graphs.append(self.graph.copy())

        # Rolling measurement buffers
        edge_history = deque(maxlen=patience)
        spectrum_norm_history = deque(maxlen=patience)

        stop_condition = False
        pbar = tqdm(total=max_iterations, desc="Generating graph",
                    leave=False, disable=False)

        for i in range(max_iterations):
            self.add_remove_edge()

            # --- periodic checkpoint ---
            if i % check_interval == 0:
                graphs.append(self.graph.copy())
                current_edges = int(np.sum(np.triu(self.graph)))
                edge_history.append(current_edges)

                spec_norm = float(np.linalg.norm(
                    self.calculate_spectrum(self.graph)))
                spectrum_norm_history.append(spec_norm)

                # Convergence check (only after warm-up and enough samples)
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
                        'edges': current_edges,
                        'cv_e': f'{cv_edges:.4f}',
                        'cv_s': f'{cv_spectrum:.4f}',
                        'converged': stop_condition,
                    })

                    if stop_condition:
                        pbar.update(max_iterations - pbar.n)  # fill bar
                        break

            pbar.update(1)

        pbar.close()
        spectra = self.calculate_spectrum(self.graph)
        return list(graphs), spectra

    
    def populate_edges_spectrum(self, warm_up, max_iterations, patience, real_graph, edge_delta=None, verbose=True):
        #TODO Addnstead of warm up something related to minimum gic that i want comparing with the real graph 
        i = 0
        best_iteration = 0

        # Spectrum variables
        spectrum_diffs = []
        real_spectrum = self.calculate_spectrum(real_graph)
        real_edges = np.sum(real_graph)
        no_improvement_count = 0
        best_spectrum_diff = float('inf')

        # Graph variables
        graphs = [self.graph.copy()]
        current_edges = np.sum(self.graph)
        best_graph = self.graph.copy()  # Initialize with the starting graph

        while ((no_improvement_count < patience) or 
               (i < warm_up)):

            current_edges = np.sum(self.graph)

            if i > max_iterations:
                if verbose:
                    print('Max iterations reached. Convergence reached')
                break

            # Check edge criteria only if edge_delta is provided
            if edge_delta is not None:
                if current_edges < real_edges - edge_delta:
                    pass
                if current_edges > real_edges + edge_delta:
                    print('Too many edges. Convergence reached')
                    break

            # Main add remove step
            self.add_remove_edge()

            current_spectrum = self.calculate_spectrum(self.graph)
            spectrum_diff = np.linalg.norm(current_spectrum - real_spectrum)
            spectrum_diffs.append(spectrum_diff)

            if verbose and i % 1000 == 0:
                print(f'\t Spectrum difference: {spectrum_diff}')
            
            graphs.append(self.graph.copy())

            if spectrum_diff < best_spectrum_diff:
                best_spectrum_diff = spectrum_diff
                best_graph = self.graph.copy()
                best_iteration = i
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            i += 1

        if verbose:
            print(f'\t Best iteration: {best_iteration}')
            print(f'\t Best spectrum difference: {best_spectrum_diff}')
            print(f'\t Number of edges: {np.sum(self.graph)}, Number of edges real graph: {real_edges}')

        self.graph = best_graph
        spectra = self.calculate_spectrum(self.graph)

        return graphs, spectra, spectrum_diffs, best_iteration

    def populate_edges_spectrum_min_gic(self, max_iterations, patience,
                                       real_graph, min_gic_threshold,
                                       gic_dist_type='KL', edge_delta=None,
                                       check_interval=50, verbose=True,
                                       er_p=0.05):
        """Populate edges targeting a real graph, using a two-phase criterion.

        Phase 1 — *GIC gate*:  Every ``check_interval`` iterations compute
        the GIC (spectral divergence) between the current graph and the real
        graph.  Keep iterating until GIC drops below ``min_gic_threshold``.

        Phase 2 — *spectrum patience*:  After the GIC gate is passed, track
        the best Laplacian-spectrum distance (measured every
        ``check_interval`` steps).  Stop after ``patience`` consecutive
        *checks* without improvement.

        The graph with the smallest spectrum distance found during the run
        is returned regardless of which phase it was found in.

        Args:
            max_iterations: Hard upper-bound on iterations.
            patience: Number of consecutive *checks* (not raw iterations)
                without spectral improvement before stopping (phase 2).
            real_graph: Target adjacency matrix (np.ndarray or nx.Graph).
            min_gic_threshold: GIC value that must be reached to enter
                phase 2.
            gic_dist_type: Distance metric for GIC ('KL', 'L1', 'L2').
            edge_delta: Optional.  If the generated edge count deviates from
                the real graph by more than this, stop early.
            check_interval: How often (in iterations) to compute the
                expensive spectrum / GIC.  Defaults to 50.
            verbose: Print progress information.
            er_p: (unused, kept for backward compatibility)

        Returns:
            graphs, spectra, spectrum_diffs, best_iteration, best_graph,
            gic_values
        """
        best_iteration = 0

        # GIC state
        gic_threshold_reached = False
        current_gic = float('inf')
        gic_values = []

        # Spectrum state
        spectrum_diffs = []
        if isinstance(real_graph, nx.Graph):
            real_graph_np = nx.to_numpy_array(real_graph)
        else:
            real_graph_np = real_graph

        real_spectrum = self.calculate_spectrum(real_graph_np)
        real_edges = np.sum(np.triu(real_graph_np))
        no_improvement_checks = 0  # counts *checks*, not iterations
        best_spectrum_diff = float('inf')

        # Graph bookkeeping
        graphs = deque(maxlen=max(2 * patience + 100, 500))
        graphs.append(self.graph.copy())
        best_graph = self.graph.copy()

        # Pre-build NetworkX reference once for GIC calls
        real_nx_graph = nx.from_numpy_array(real_graph_np)

        # Progress bar — all writes happen *before* close
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
                'Edges': f'{int(np.sum(np.triu(self.graph)))}/{int(real_edges)}'
            })

        stop_reason = 'unknown'

        for i in range(max_iterations):
            # --- edge_delta guard ---
            if edge_delta is not None:
                current_edges = np.sum(np.triu(self.graph))
                if current_edges > real_edges + edge_delta:
                    stop_reason = f'edge count exceeded delta ({edge_delta})'
                    break

            # --- main step ---
            self.add_remove_edge()

            # --- periodic expensive check ---
            if i % check_interval == 0:
                graphs.append(self.graph.copy())

                # Spectrum distance
                current_spectrum = self.calculate_spectrum(self.graph)
                spectrum_diff = np.linalg.norm(current_spectrum - real_spectrum)
                spectrum_diffs.append(spectrum_diff)

                # GIC (computed regardless of whether spectrum improved)
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

                # Track best graph by spectrum distance
                if spectrum_diff < best_spectrum_diff:
                    best_spectrum_diff = spectrum_diff
                    best_graph = self.graph.copy()
                    best_iteration = i
                    if gic_threshold_reached:
                        no_improvement_checks = 0
                elif gic_threshold_reached:
                    no_improvement_checks += 1

                # Update progress bar
                if verbose and pbar:
                    pbar.set_postfix({
                        'GIC': f'{current_gic:.4f}',
                        'Spec': f'{best_spectrum_diff:.4f}',
                        'Pat': f'{no_improvement_checks}/{patience}',
                        'Edges': f'{int(np.sum(np.triu(self.graph)))}/{int(real_edges)}'
                    })

                # Check patience exhaustion
                if gic_threshold_reached and no_improvement_checks >= patience:
                    stop_reason = (
                        f'no spectral improvement for {patience} checks '
                        f'after GIC threshold was met')
                    break
            # end periodic check

            if verbose and pbar:
                pbar.update(1)
        else:
            # for-else: loop finished without break
            stop_reason = f'max iterations ({max_iterations:,}) reached'

        # --- summary (printed before closing pbar) ---
        if verbose and pbar:
            current_edges = int(np.sum(np.triu(self.graph)))
            pbar.write(f'\nStopping: {stop_reason}')
            pbar.write(f'  Best iteration: {best_iteration:,}')
            pbar.write(f'  Best spectrum diff: {best_spectrum_diff:.4f}')
            pbar.write(f'  Edges in best graph: '
                       f'{int(np.sum(np.triu(best_graph)))} '
                       f'(real: {int(real_edges)})')
            pbar.close()

        spectra = self.calculate_spectrum(best_graph)
        return (list(graphs), spectra, spectrum_diffs,
                best_iteration, best_graph, gic_values)
