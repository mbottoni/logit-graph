import numpy as np
import networkx as nx
from scipy.stats import ks_2samp
from scipy.special import expit

#from tqdm import tqdm
from tqdm.notebook import tqdm

from .degrees_counts import degree_vertex, get_sum_degrees
from . import gic

class GraphModel:
    def __init__(self, n, d, sigma, alpha=1, beta=1, er_p=0.05):
        self.n = n # number of nodes
        self.d = d # number of neighbors to consider 
        self.sigma = sigma # Offset weights
        self.alpha = alpha # weights on i node
        self.beta = beta   # weights on j node
        #self.graph = self.generate_empty_graph(n)
        self.er_p = er_p
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
        # Pre compute
        # TODO: I dont need to pick every node
        #sum_degrees = np.zeros(self.n)
        #for i in range(self.n):
        #    sum_degrees[i] = get_sum_degrees(self.graph, vertex=i, d = self.d)

        i, j = np.random.choice(self.n, 2, replace=False)
        #total_degree = (sum_degrees[i] + sum_degrees[j]) + self.sigma
        total_degree = self.alpha * ( get_sum_degrees(self.graph, vertex=i, d=self.d) + self.beta * get_sum_degrees(self.graph, vertex=j, d=self.d) ) + self.sigma
        self.graph[j, i] = self.graph[i, j] = self.get_edge_logit(total_degree) # here we can add or remove vertex

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

    def populate_edges_baseline(self, warm_up, max_iterations, patience):
        i = 0
        stop_condition = False
        graphs = [self.graph.copy()]  # List to store the graphs

        while i < max_iterations and (i < warm_up or not stop_condition):
            print(f'iteration: {i}')
            self.add_remove_edge()  # add or remove vertex
            graphs.append(self.graph.copy())

            if len(graphs) > 1000:
                graphs.pop(0)

            if i > warm_up:
                stop_condition_n_edges = self.check_convergence_number_of_edges(graphs, threshold_edges=10, stability_window=patience)
                stop_condition_spectrum = self.check_convergence_spectrum(graphs, threshold_spectrum=100, stability_window=patience)
                stop_condition = stop_condition_n_edges and stop_condition_spectrum

            i += 1

        spectra = self.calculate_spectrum(self.graph)
        return graphs, spectra

    
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

            if verbose and i % 1000 == 0:
                print(f'iteration: {i}')

            if i > max_iterations:
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

        print(f'\t Best iteration: {best_iteration}')
        print(f'\t Best spectrum difference: {best_spectrum_diff}')
        print(f'\t Number of edges: {np.sum(self.graph)}, Number of edges real graph: {real_edges}')

        self.graph = best_graph
        spectra = self.calculate_spectrum(self.graph)

        return graphs, spectra, spectrum_diffs, best_iteration

    def populate_edges_spectrum_min_gic(self, max_iterations, patience, real_graph, min_gic_threshold, gic_dist_type='KL', edge_delta=None, verbose=True, er_p=0.05):
        """
        Populates edges by iteratively adding/removing edges, aiming to minimize
        the spectral difference to a real graph, using GIC as an initial threshold.

        Args:
            max_iterations (int): Maximum number of iterations allowed.
            patience (int): Number of iterations without spectral improvement before stopping
                           (only active after GIC threshold is met).
            real_graph (np.ndarray): The target graph (adjacency matrix) to compare against.
            min_gic_threshold (float): The GIC value that must be reached before checking
                                      for convergence based on spectrum difference.
            gic_dist_type (str): The distance metric for GIC ('KL', 'L1', 'L2'). Defaults to 'KL'.
            edge_delta (Optional[int]): If set, stops if the number of edges deviates
                                       from the real graph's edges by more than this delta.
            verbose (bool): If True, prints progress information. Defaults to True.

        Returns:
            tuple: Contains the list of graphs generated, the final spectrum,
                   the list of spectrum differences, and the best iteration index.
        """
        i = 0
        best_iteration = 0

        # GIC variables
        gic_threshold_reached = False
        current_gic = float('inf')
        gic_values = []

        # Spectrum variables
        spectrum_diffs = []
        # Ensure real_graph is a NumPy array for calculations if it isn't already
        if isinstance(real_graph, nx.Graph):
             real_graph_np = nx.to_numpy_array(real_graph)
        else:
             real_graph_np = real_graph # Assume it's already numpy

        real_spectrum = self.calculate_spectrum(real_graph_np)
        real_edges = np.sum(np.triu(real_graph_np)) # Use triu for undirected edges count
        no_improvement_count = 0
        best_spectrum_diff = float('inf')

        # Graph variables
        graphs = [self.graph.copy()]
        best_graph = self.graph.copy()  # Initialize with the starting graph

        # Convert real_graph_np to NetworkX for GIC once
        real_nx_graph = nx.from_numpy_array(real_graph_np)

        # Progress bar setup
        pbar = None
        if verbose:
            pbar = tqdm(total=max_iterations, desc="🔄 Optimizing Graph", leave=True, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
            pbar.set_postfix({
                'GIC': f'{current_gic:.4f}',
                'Spec': f'{best_spectrum_diff:.4f}',
                'Pat': f'{no_improvement_count}/{patience}',
                'Edges': f'{np.sum(np.triu(self.graph)):.0f}/{real_edges:.0f}'
            })

        while (not gic_threshold_reached or no_improvement_count < patience):

            current_edges = np.sum(np.triu(self.graph)) # Use triu for undirected edges count

            if verbose and i % 1000 == 0 and pbar and i != 0 :
                pbar.write(f"📊 Iteration {i:,}: 🎯 GIC ({gic_dist_type}): {current_gic:.4f} (Target: ≤{min_gic_threshold}) 📈 Best Spectrum Diff: {best_spectrum_diff:.4f} ⏱️  Patience: {no_improvement_count}/{patience} 🔗 Edges: {current_edges:.0f} (Target: {real_edges:.0f})")

            if i >= max_iterations:
                if verbose and pbar:
                    pbar.write(f'⏰ Max iterations ({max_iterations:,}) reached. Stopping.')
                break

            # Check edge criteria only if edge_delta is provided
            if edge_delta is not None:
                if current_edges < real_edges - edge_delta:
                    pass
                if current_edges > real_edges + edge_delta:
                    if verbose and pbar:
                        pbar.write('🚫 Too many edges. Convergence reached')
                    break

            # Main add remove step
            self.add_remove_edge()
            graphs.append(self.graph.copy()) # Store graph after modification

            # --- Calculate Differences ---
            current_spectrum = self.calculate_spectrum(self.graph)
            spectrum_diff = np.linalg.norm(current_spectrum - real_spectrum)
            spectrum_diffs.append(spectrum_diff)

            # --- Update Best Graph based on Spectrum ---
            # Always track the best graph found so far based on spectrum difference
            if spectrum_diff < best_spectrum_diff:
                best_spectrum_diff = spectrum_diff
                best_graph = self.graph.copy()
                best_iteration = i
                # If GIC threshold was already met, finding a better graph resets patience
                if gic_threshold_reached:
                    no_improvement_count = 0
            # --- Check GIC Threshold ---
            elif not gic_threshold_reached:
                 # Convert current numpy graph to NetworkX for GIC
                 current_nx_graph = nx.from_numpy_array(self.graph)
                 # Calculate GIC
                 try:
                     # Use the real graph as the reference ('graph' param) and the current generated graph
                     # as the model ('log_graph' param with model='LG')
                     gic_calculator = gic.GraphInformationCriterion(real_nx_graph, model='LG', log_graph=current_nx_graph, dist=gic_dist_type)
                     current_gic = gic_calculator.calculate_gic()
                 except Exception as e:
                     if verbose and pbar:
                         pbar.write(f'⚠️  Warning: GIC calculation failed at iteration {i:,}: {e}')
                     # Decide how to handle error: continue, break, assign high GIC?
                     # Let's assign high GIC and continue for now.
                     current_gic = float('inf')

                 if current_gic <= min_gic_threshold:
                     if verbose and pbar:
                         pbar.write(f'🎉 GIC threshold {min_gic_threshold} reached at iteration {i:,} (GIC: {current_gic:.4f})')
                         pbar.write(f'🔍 Starting convergence check based on spectrum difference (Patience: {patience})')
                     gic_threshold_reached = True
                     no_improvement_count = 0 # Reset patience counter when threshold is first met
                 else:
                     # If GIC threshold not met, patience counter doesn't increase.
                     # We reset it here explicitly to avoid carrying over counts from before GIC was checked.
                     no_improvement_count = 0

            # --- Increment Patience Counter (only if applicable) ---
            elif gic_threshold_reached:
                # Increment patience counter only if GIC threshold is met AND no improvement was found
                no_improvement_count += 1

            # Update progress bar
            if verbose and pbar:
                pbar.update(1)
                pbar.set_postfix({
                    'GIC': f'{current_gic:.4f}',
                    'Spec': f'{best_spectrum_diff:.4f}',
                    'Pat': f'{no_improvement_count}/{patience}',
                    'Edges': f'{current_edges:.0f}/{real_edges:.0f}'
                })

            # --- Iteration Increment ---
            i += 1
            gic_values.append(current_gic)

            # Save mem - This part makes returning best_graph crucial
            if len(graphs) > 2 * patience + 100: # Keep slightly more than patience window
               graphs.pop(0)
               # IMPORTANT: If pop(0) is used, best_iteration index becomes unreliable
               # for accessing the graphs list later.

        if verbose and pbar:
            pbar.close()

        if verbose and pbar:
            pbar.write(f'\n🏁 Stopping Condition Met')
            if i >= max_iterations:
                 pbar.write(f'   📍 Reason: Max iterations ({max_iterations:,}) reached.')
            elif not gic_threshold_reached:
                 pbar.write(f'   📍 Reason: Stopped before GIC threshold ({min_gic_threshold}) was reached.')
                 pbar.write(f'   📊 Final GIC: {current_gic:.4f}')
            elif no_improvement_count >= patience:
                 pbar.write(f'   📍 Reason: No improvement in spectrum difference for {patience:,} iterations after GIC threshold was met.')
            elif edge_delta is not None and current_edges > real_edges + edge_delta:
                 pbar.write(f'   📍 Reason: Edge count difference exceeded delta ({edge_delta}).')
            else:
                 pbar.write(f'   📍 Reason: Unknown (Loop condition terminated unexpectedly).')
            pbar.write(f'   📈 Results Summary')
            pbar.write(f'   🏆 Best iteration found: {best_iteration:,}')
            pbar.write(f'   📊 Best spectrum difference: {best_spectrum_diff:.4f}')
            final_edges = np.sum(np.triu(best_graph))
            pbar.write(f'   🔗 Edges in best graph: {final_edges:.0f} (Real graph edges: {real_edges:.0f})')

        spectra = self.calculate_spectrum(best_graph)

        return graphs, spectra, spectrum_diffs, best_iteration, best_graph, gic_values
