import numpy as np
import networkx as nx
from scipy.stats import ks_2samp
from scipy.special import expit

from src.degrees_counts import degree_vertex, get_sum_degrees
import src.gic as gic

class GraphModel:
    def __init__(self, n, d, sigma, alpha=1, beta=1):
        self.n = n # number of nodes
        self.d = d # number of neighbors to consider 
        self.sigma = sigma # Offset weights
        self.alpha = alpha # weights on i node
        self.beta = beta   # weights on j node
        self.graph = self.generate_empty_graph(n)

    def generate_empty_graph(self, n):
        return np.zeros((n, n))

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

