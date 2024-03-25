import numpy as np
import networkx as nx

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
                    noise = abs(np.random.normal(0, self.sigma))
                    sum_degrees_raw = (self.get_sum_degrees(i, p) + self.get_sum_degrees(j, p) )
                    sum_degrees = (sum_degrees_raw + noise) / normalization
                    print(f"sum_degrees: {sum_degrees}  noise: {noise}  normalization: {normalization} sum_degrees_raw: {sum_degrees_raw}")
                    self.graph[i, j] = self.get_edge_logit(sum_degrees)

    def check_convergence(self, graph_list, tolerance=1):
        #difference = sum(np.sum(np.abs(graph_list[-i] - graph_list[-(i + 1)])) for i in range(1, 20, 2))
        #return difference <= tolerance
        return False

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
                stop_condition = self.check_convergence(graphs, tolerance=1)

            i += 1

        return graphs, spectra
