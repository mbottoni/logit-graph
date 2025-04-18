import networkx as nx
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import euclidean, cityblock
from numpy import errstate

class GraphInformationCriterion:
    def __init__(self, graph, model, log_graph=None, p=None, dist='KL', **kwargs):
        self.graph = graph
        self.log_graph =  log_graph
        self.model = model
        self.parameter = p
        self.dist_type = dist
        self.kwargs = kwargs
        self.n = graph.number_of_nodes()

    def compute_spectral_density(self, graph):
        laplacian = nx.normalized_laplacian_matrix(graph)
        eigenvalues = np.linalg.eigvalsh(laplacian.todense())
        hist, bin_edges = np.histogram(eigenvalues, bins=10, range=(0, 2), density=True)
        return hist, bin_edges

    def generate_model_graph(self):
        if isinstance(self.model, str):
            if self.model == "ER":
                return nx.erdos_renyi_graph(self.n, self.parameter)
            elif self.model == "GRG":
                return nx.random_geometric_graph(self.n, self.parameter)
            elif self.model == "KR":
                return nx.random_regular_graph(int(self.parameter), self.n)
            elif self.model == "WS":
                k = int(np.ceil(np.sqrt(self.n)))
                return nx.watts_strogatz_graph(self.n, k, self.parameter)
            elif self.model == "BA":
                m = max(1, int(self.parameter))  # Ensure m is at least 1
                return nx.barabasi_albert_graph(self.n, m)
            elif self.model == "LG":
                if isinstance(self.log_graph, tuple):
                    return self.log_graph[0]
                else:
                    return self.log_graph

        elif callable(self.model):
            return self.model(self.n, self.parameter)
        else:
            raise ValueError(f"{self.model}: Model definition is not recognized.")

    def calculate_gic(self, model_den=None):
        graph_den, _ = self.compute_spectral_density(self.graph)
        if model_den is None:
            model_graph = self.generate_model_graph()
            if self.model == "LG":
                #print('gic module : ', type(model_graph), model_graph)
                pass
            model_den, _ = self.compute_spectral_density(model_graph)
        else:
            model_den = model_den # for getting the avg spectrum

        if self.dist_type == 'KL':
            distance = entropy(graph_den + 1e-10, model_den + 1e-10)  # KL divergence
        elif self.dist_type == 'L1':
            distance = cityblock(graph_den, model_den)  # L1 distance
        elif self.dist_type == 'L2':
            distance = euclidean(graph_den, model_den)  # L2 distance
        else:
            raise ValueError("Unsupported distance type specified.")

        return distance
