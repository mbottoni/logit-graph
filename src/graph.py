import numpy as np
import networkx as nx
from scipy.stats import ks_2samp
from scipy.special import expit

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
        self.graph = self.generate_empty_graph(n)

    def generate_empty_graph(self, n):
        return np.zeros((n, n))

    @classmethod
    def calculate_spectrum(cls, graph):
        G = nx.from_numpy_array(graph)
        eigenvalues = nx.laplacian_spectrum(G)
        return np.sort(eigenvalues)

    def logistic_regression(self, sum_degrees):
        #TODO: It is possible to optimize putting in 1/1+exp(-sum_degrees)
        #num = np.exp(sum_degrees)
        #denom = 1 + 1 * np.exp(sum_degrees)
        #return num / denom
        return expit(sum_degrees)


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

        #normalization = self.n - 1
        normalization =  1
        return [get_degree(vertex)/normalization] + [get_degree(neighbor)/normalization for neighbor in current_neighbors]

    def get_sum_degrees(self, vertex, p=1):
        return sum(self.degree_vertex(vertex, p))

    def get_edge_logit(self, sum_degrees):
        val_log = self.logistic_regression(sum_degrees)
        random_choice = np.random.choice([1, 0], p=[val_log, 1 - val_log])
        return random_choice

    def add_remove_edge(self, p):
        # Gen a new graph 
        #graph_new = self.generate_empty_graph(self.n)

        # Pre compute
        sum_degrees = np.zeros(self.n)
        for i in range(self.n):
            sum_degrees[i] = self.get_sum_degrees(i, p)
    
        # add or remove edge
        for i in range(self.n):
            for j in range(i + 1, self.n):  # Use symmetry, only compute half and mirror
                if i != j:
                    # Calculate the edge logit only once per pair
                    total_degree = self.alpha * sum_degrees[i] + self.beta * sum_degrees[j] + self.sigma
                    #graph_new[j, i] = graph_new[i, j] = self.get_edge_logit(total_degree) # here we can add or remove vertex
                    self.graph[j, i] = self.graph[i, j] = self.get_edge_logit(total_degree) # here we can add or remove vertex

        # Update the graph directly
        #self.graph= graph_new.copy() 

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

    def populate_edges(self, warm_up, max_iterations, degree_dist_threshold=0.05, stability_window=5):
        i = 0
        stop_condition = False
        graphs = [self.graph.copy()]  # List to store the graphs
        #spectra = []  # List to store the spectrum at each iteration

        while i < max_iterations and (i < warm_up or not stop_condition):
            print(f'iteration: {i}')
            self.add_remove_edge(self.p)  # add or remove vertex
            #spectrum = self.calculate_spectrum(self.graph)

            #spectra.append(spectrum)
            graphs.append(self.graph.copy())

            if i > warm_up:
                #stop_condition = self.check_convergence(graphs, tolerance=1)
                stop_condition = self.check_convergence(graphs, stability_window=stability_window, degree_dist_threshold=degree_dist_threshold)


            i += 1

        spectra = self.calculate_spectrum(self.graph)
        return graphs, spectra


