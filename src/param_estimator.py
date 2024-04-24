
import networkx as nx
import numpy as np
import sys
sys.path.append('..')
import src.gic as gic

class GraphParameterEstimator():
    def __init__(self, graph, model, interval=None, eps=0.01, search='grid', **kwargs):
        self.graph = graph
        self.model = model
        self.interval = interval
        self.eps = eps
        self.search = search
        self.kwargs = kwargs
        self.n = graph.number_of_nodes()
        self.m = graph.number_of_edges()

        if isinstance(self.model, str):
            self.model_function = self.get_model_function(self.model)
        else:
            self.model_function = self.model

        self.search_interval = self.get_search_interval()

    def get_model_function(self, model_name):
        if model_name == "ER":
            return lambda n, p: nx.erdos_renyi_graph(n, p)
        elif model_name == "GRG":
            return lambda n, r: nx.random_geometric_graph(n, r)
        elif model_name == "KR":
            return lambda n, k: nx.random_regular_graph(k, n)
        elif model_name == "WS":
            return lambda n, p: nx.watts_strogatz_graph(n, k=int(np.ceil(np.sqrt(n))), p=p)
        elif model_name == "BA":
            return lambda n, m: nx.barabasi_albert_graph(n, m)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_search_interval(self):
        if self.interval is not None:
            return np.arange(self.interval['lo'], self.interval['hi'], self.eps)
        elif isinstance(self.model, str):
            if self.model == "ER" or self.model == "WS":
                return np.arange(0, 1, self.eps)
            elif self.model == "GRG":
                return np.arange(0, np.sqrt(2), self.eps)
            elif self.model == "KR":
                return np.arange(0, self.n, int(self.n * self.eps))
            elif self.model == "BA":
                return np.arange(0, 3, self.eps)
        else:
            raise ValueError("Interval or model must be provided")

    def estimate(self):
        if self.search == "grid":
            return self.grid_search()
        elif self.search == "ternary":
            return self.ternary_search()
        else:
            raise ValueError(f"Unknown search method: {self.search}")

    def grid_search(self):
        min_param = None
        min_gic = float('inf')
        for param in self.search_interval:
            gic = self.calculate_gic(param)
            if gic < min_gic:
                min_gic = gic
                min_param = param
        return {'param': min_param, 'gic': min_gic}

    def ternary_search(self):
        min_param = None
        min_gic = float('inf')

        for interval in self.search_interval:
            lo = interval['lo']
            hi = interval['hi']
            while True:
                if abs(lo - hi) < self.eps or lo > hi:
                    mid = (lo + hi) / 2
                    if self.model == 'KR':
                        mid = int(round(mid))
                    current_gic = self.calculate_gic(mid)
                    if current_gic < min_gic:
                        min_gic = current_gic
                        min_param = mid
                    break

                left_third = (2 * lo + hi) / 3
                right_third = (lo + 2 * hi) / 3

                if self.model == 'KR':
                    left_third = int(round(left_third))
                    right_third = int(round(right_third))

                gic1 = self.calculate_gic(left_third)
                gic2 = self.calculate_gic(right_third)

                if gic1 <= gic2:
                    hi = right_third
                else:
                    lo = left_third

        return {'param': min_param, 'gic': min_gic}

    def calculate_gic(self, param):
        # Instantiate the GraphInformationCriterion with the current parameter
        gic_calculator = gic.GraphInformationCriterion(
            graph=self.graph, model=self.model_function, p=param, **self.kwargs
        )
        return gic_calculator.calculate_gic()