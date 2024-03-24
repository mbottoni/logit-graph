import numpy as np
import networkx as nx

class GraphModel:
    def __init__(self, n, p, c, beta, threshold, sigma=1):
        self.n = n
        self.p = p
        self.c = c
        self.beta = beta
        self.threshold = threshold
        self.sigma = sigma
        self.graph = self.generate_random_graph(n, p)

    def generate_random_graph(self, n, p):
        adj_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < p:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        return adj_matrix

    def calculate_spectrum(self):
        G = nx.from_numpy_array(self.graph)
        eigenvalues = nx.laplacian_spectrum(G)
        return np.sort(eigenvalues)

    def logistic_regression(self, sum_degrees):
        num = self.c
        denom = 1 + self.beta * np.exp(-sum_degrees)
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
        for _ in range(p - 1):
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
                    sum_degrees = (self.get_sum_degrees(i, p) + self.get_sum_degrees(j, p) + abs(np.random.normal(0, self.sigma))) / (self.n * (self.n - 1) / 2)
                    self.graph[i, j] = self.get_edge_logit(sum_degrees)

    def check_convergence(self, graph_list, tolerance=1):
        difference = sum(np.sum(np.abs(graph_list[-i] - graph_list[-(i + 1)])) for i in range(1, 20, 2))
        return difference <= tolerance
