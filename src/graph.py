import numpy as np
import networkx as nx
from scipy.stats import ks_2samp

class GraphModel:
    def __init__(self, n, p, c, beta, threshold, sigma=1):
        self.n = n # number of nodes
        self.p = p # number of neighbors to consider 
        self.c = c # numerator of the logit model
        self.beta = beta # denominator of the logit model
        self.threshold = threshold # theshold for creating and edge
        self.sigma = sigma # stddev of noise when adding an vertex
        self.graph = self.generate_random_graph(n, p)

    def generate_random_graph(self, n, p):
        adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < p:
                    adj_matrix[i, j] = 0
                    adj_matrix[j, i] = 0
        return adj_matrix

    def calculate_spectrum(self):
        G = nx.from_numpy_array(self.graph)
        eigenvalues = nx.laplacian_spectrum(G)
        return np.sort(eigenvalues)

    def logistic_regression(self, sum_degrees):
        num = self.c
        denom = 1 + self.beta * np.exp(sum_degrees)
        return num / denom

    def degree_vertex(self, vertex, p):
        def get_neighbors(v):
            return [i for i, x in enumerate(self.graph[v]) if x == 1]

        def get_degree(v):
            return sum(self.graph[v])

        if p == 0:
            return [get_degree(vertex)]
        if p == 1:
            neighbors = get_neighbors(vertex)
            return [get_degree(vertex)] + [get_degree(neighbor) for neighbor in neighbors]

        visited, current_neighbors = set([vertex]), get_neighbors(vertex)
        for _ in range(int(p) - 1):
            next_neighbors = []
            for v in current_neighbors:
                next_neighbors.extend([nv for nv in get_neighbors(v) if nv not in visited])
                visited.add(v)
            current_neighbors = list(set(next_neighbors))
        return [get_degree(vertex)] + [get_degree(neighbor) for neighbor in current_neighbors]

    def get_sum_degrees(self, vertex, p=1):
        return sum(self.degree_vertex(vertex, p))

    def get_edge_logit(self, sum_degrees):
        val_log = self.logistic_regression(sum_degrees)
        return np.random.choice(np.arange(0, 2), p=[1 - val_log, val_log])

    def add_vertex(self, p):
        for i in range(self.n):
            for j in range(self.n):
                if i != j and self.graph[i, j] == 0:
                    normalization = self.n * (self.n - 1) / 2
                    #noise = abs(np.random.normal(0, self.sigma))
                    noise = 0 # deprecated
                    sum_degrees_raw = (self.get_sum_degrees(i, p) + self.get_sum_degrees(j, p) )
                    sum_degrees = (sum_degrees_raw + noise) / normalization
                    #print(f"sum_degrees: {sum_degrees}  noise: {noise}  normalization: {normalization} sum_degrees_raw: {sum_degrees_raw}")
                    self.graph[i, j] = self.get_edge_logit(sum_degrees)

    def check_convergence(self, graphs, spectra, stability_window=5,
                          spectral_stability_threshold=0.1, degree_dist_threshold=0.05):

        def degree_distribution_stability(graph1, graph2):
            # Calculate degree sequences
            degrees1 = np.sum(graph1, axis=1)
            degrees2 = np.sum(graph2, axis=1)
            # Compute the Kolmogorov-Smirnov statistic
            ks_stat, _ = ks_2samp(degrees1, degrees2)
            print(f"KS Statistic: {ks_stat}")
            return ks_stat

        def spectral_change_stability(spectrum1, spectrum2):
            # Filter out zero eigenvalues and focus on the leading non-zero eigenvalues
            non_zero_spectrum1 = spectrum1[spectrum1 > 1e-5][:5]  # Consider the first 5 non-zero eigenvalues
            non_zero_spectrum2 = spectrum2[spectrum2 > 1e-5][:5]
            epsilon = 1e-9
            relative_changes = np.abs((non_zero_spectrum1 - non_zero_spectrum2) / (non_zero_spectrum1 + epsilon))
            print(f"Max Relative Change in Spectrum: {np.max(relative_changes)}")
            return np.max(relative_changes)

        if len(graphs) <= stability_window:
            print("Not enough graphs for stability check.")
            return False

        # Spectral Stability: Check if the spectral changes are below the threshold
        spectral_changes_stable = all(
            spectral_change_stability(spectra[-i - 1], spectra[-i]) < spectral_stability_threshold
            for i in range(1, stability_window)
        )
        print(f"Spectral Changes Stable: {spectral_changes_stable}")

        # Degree Distribution Stability: Check if the KS distance between degree distributions is below the threshold
        degree_dist_stable = all(
            degree_distribution_stability(graphs[-i - 1], graphs[-i]) < degree_dist_threshold
            for i in range(1, stability_window)
        )
        print(f"Degree Distribution Stable: {degree_dist_stable}")

        # Final convergence check
        is_converged = spectral_changes_stable and degree_dist_stable
        print(f"Graph Converged: {is_converged}")
        print('\n'*3)
        return is_converged


    def populate_edges(self, warm_up=50, max_iterations=100):
        i = 0
        stop_condition = False
        graphs = [self.graph.copy()]  # List to store the graphs
        spectra = []  # List to store the spectrum at each iteration

        while i < max_iterations and (i < warm_up or not stop_condition):
            print(f'iteration: {i}')
            self.add_vertex(self.p)  # add vertex
            spectrum = self.calculate_spectrum()

            spectra.append(spectrum)
            graphs.append(self.graph.copy())

            if i > warm_up:
                #stop_condition = self.check_convergence(graphs, tolerance=1)
                stop_condition = self.check_convergence(graphs, spectra)


            i += 1

        return graphs, spectra
