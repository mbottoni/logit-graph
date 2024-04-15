import numpy as np
import networkx as nx
from scipy.stats import ks_2samp

class GraphModel:
    def __init__(self, n, p, alpha, beta, sigma, threshold, n_iteration, warm_up):
        self.n = n # number of nodes
        self.p = p # number of neighbors to consider 
        self.alpha = alpha 
        self.beta  = beta
        self.sigma = sigma
        self.threshold = threshold # theshold for creating and edge
        self.n_iteration = n_iteration
        self.warm_up = warm_up
        self.graph = self.generate_random_graph(n, p)

    def generate_random_graph(self, n, p):
        adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                #if np.random.rand() < p:
                adj_matrix[i, j] = 0
                adj_matrix[j, i] = 0

        return adj_matrix

    @classmethod
    def calculate_spectrum(cls, graph):
        G = nx.from_numpy_array(graph)
        eigenvalues = nx.laplacian_spectrum(G)
        return np.sort(eigenvalues)

    def logistic_regression(self, sum_degrees):
        num = 1
        denom = 1 + 1 * np.exp(sum_degrees)
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

        normalization = self.n * (self.n - 1) / 2
        return [get_degree(vertex)/normalization] + [get_degree(neighbor)/normalization for neighbor in current_neighbors]

    def get_sum_degrees(self, vertex, p=1):
        return sum(self.degree_vertex(vertex, p))

    def get_edge_logit(self, sum_degrees):
        val_log = self.logistic_regression(sum_degrees)
        print('sum_degrees',sum_degrees)
        print('val_log', val_log)
        #random_choice = np.random.choice(np.arange(0, 2), p=[val_log, 1 - val_log])
        random_choice = np.random.choice([1, 0], p=[val_log, 1 - val_log])
        print('random_choice', random_choice)
        print()
        return random_choice

    def add_remove_vertex(self, p):
        graph_new = self.generate_random_graph(self.n, p) # 
        for i in range(self.n):
            for j in range(self.n):
                # Adding or removing vertex 
                if i != j:
                    degrees_i = self.get_sum_degrees(i, p)
                    degrees_j = self.get_sum_degrees(j, p)

                    sum_degrees_raw = ( self.alpha * degrees_i + self.beta * degrees_j )
                    sum_degrees = ( sum_degrees_raw + self.sigma ) 

                    #self.graph[i, j] = self.get_edge_logit(sum_degrees) # here we can add or remove vertex
                    graph_new[i, j] = self.get_edge_logit(sum_degrees) # here we can add or remove vertex

        self.graph= graph_new.copy() 


    def check_convergence(self, graphs, stability_window=5, degree_dist_threshold=0.05):

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

    def populate_edges(self, warm_up, max_iterations):
        i = 0
        stop_condition = False
        graphs = [self.graph.copy()]  # List to store the graphs
        spectra = []  # List to store the spectrum at each iteration
        print('oiiii')

        while i < max_iterations and (i < warm_up or not stop_condition):
            print(f'iteration: {i}')
            self.add_remove_vertex(self.p)  # add or remove vertex
            spectrum = self.calculate_spectrum(self.graph)

            spectra.append(spectrum)
            graphs.append(self.graph.copy())

            if i > warm_up:
                #stop_condition = self.check_convergence(graphs, tolerance=1)
                stop_condition = self.check_convergence(graphs, stability_window=3)


            i += 1

        return graphs, spectra


