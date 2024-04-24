
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp
import networkx as nx
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import euclidean, cityblock

import sys
sys.path.append('..')
import src.gic as gic
import src.param_estimator as pe
import src.logit_estimator as est


# Initial try to implement the model selection
class RandomGraphModelSelector:
    def __init__(self, real_graph, logit_graph):
        self.real_graph = real_graph
        self.logit_graph = logit_graph
        self.models = {'ER': self.erdos_renyi, 'WS': self.watts_strogatz, 'BA': self.barabasi_albert, 'LG': self.logit_graph}
        self.best_model = None
        self.model_scores = {}

    def erdos_renyi(self, n, p):
        return nx.erdos_renyi_graph(n, p)

    def watts_strogatz(self, n, k, p):
        return nx.watts_strogatz_graph(n, k, p)

    def barabasi_albert(self, n, m):
        return nx.barabasi_albert_graph(n, m)

    def evaluate_model(self, model_graph):
        # Evaluate based on degree distribution (KS test), clustering coefficient, and average shortest path length
        ks_stat, _ = ks_2samp(sorted([d for n, d in self.real_graph.degree()]),
                              sorted([d for n, d in model_graph.degree()]))
        cc_diff = abs(nx.average_clustering(self.real_graph) - nx.average_clustering(model_graph))
        try:
            aspl_diff = abs(nx.average_shortest_path_length(self.real_graph) - nx.average_shortest_path_length(model_graph))
        except nx.NetworkXError:  # Handle disconnected graph
            #aspl_diff = float('inf')
            aspl_diff = 0 # TODO: Fix this metric
        
        # Simple score combining the three metrics (lower is better)
        print(f'For the {model_graph} i get the following scores: ks_stat: {ks_stat}, cc_diff: {cc_diff}, aspl_diff: {aspl_diff}')
        score = ks_stat + cc_diff + aspl_diff
        return score

    def fit(self):
        n = len(self.real_graph.nodes())
        p = np.mean([d for n, d in self.real_graph.degree()]) / (n - 1)
        k = int(np.mean(list(dict(self.real_graph.degree()).values())))
        m = int(p * n)

        for model_name, model_func in self.models.items():
            if model_name == 'ER':
                model_graph = model_func(n, p)
            elif model_name == 'WS':
                model_graph = model_func(n, k, p)
            elif model_name == 'BA':
                model_graph = model_func(n, m)
            elif model_name == 'LG':
                model_graph = self.logit_graph
            else:
                continue

            print(model_name)
            score = self.evaluate_model(model_graph)
            print()

            self.model_scores[model_name] = score

        self.best_model = min(self.model_scores, key=self.model_scores.get)
        return self.best_model, self.model_scores

# Simplified way to do the model selection
class ModelSelectorSpectrum:
    def __init__(self, real_graph, logit_graph):
        self.real_graph = real_graph
        self.logit_graph = logit_graph
        self.models = {'ER': self.erdos_renyi, 'WS': self.watts_strogatz, 'BA': self.barabasi_albert, 'LG': self.logit_graph}
        self.best_model = None
        self.model_scores = {}

    def erdos_renyi(self, n, p):
        return nx.erdos_renyi_graph(n, p)

    def watts_strogatz(self, n, k, p):
        return nx.watts_strogatz_graph(n, k, p)

    def barabasi_albert(self, n, m):
        return nx.barabasi_albert_graph(n, m)

    def graph_spectrum(self, graph):
        laplacian = nx.laplacian_matrix(graph).astype(float)
        eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
        return eigenvalues

    def kl_divergence(self, real_spectrum, model_spectrum):
        hist_real, bins = np.histogram(real_spectrum, bins=100, density=True)
        hist_model, _ = np.histogram(model_spectrum, bins=bins, density=True)
        hist_real += np.finfo(float).eps
        hist_model += np.finfo(float).eps
        kl_div = np.sum(hist_real * np.log(hist_real / hist_model))
        return kl_div

    def model_penalty(self, model_name):
        penalties = {'ER': 1, 'WS': 2, 'BA': 1, 'LG': 3}
        return penalties.get(model_name, 0)

    def evaluate_model(self, model_graph, model_name):
        real_spectrum = self.graph_spectrum(self.real_graph)
        model_spectrum = self.graph_spectrum(model_graph)
        kl_div = self.kl_divergence(real_spectrum, model_spectrum)
        penalty = self.model_penalty(model_name)

        score = kl_div - penalty
        return score

    def fit(self):
        n = len(self.real_graph.nodes())
        p = np.mean([d for n, d in self.real_graph.degree()]) / (n - 1)
        k = int(np.mean(list(dict(self.real_graph.degree()).values())))
        m = int(p * n)

        for model_name, model_func in self.models.items():
            if model_name == 'ER':
                model_graph = model_func(n, p)
            elif model_name == 'WS':
                model_graph = model_func(n, k, p) # TODO: Fix
            elif model_name == 'BA':
                model_graph = model_func(n, m)
            elif model_name == 'LG':
                model_graph = self.logit_graph
            else:
                continue

            score = self.evaluate_model(model_graph, model_name)
            self.model_scores[model_name] = score

        self.best_model = min(self.model_scores, key=self.model_scores.get)
        return self.best_model, self.model_scores

# Class implementing the stat graph model selector
class GraphModelSelection:
    def __init__(self, graph, log_graph, log_params, models=None, parameters=None, **kwargs):
        self.graph = graph
        self.log_graph = log_graph,
        self.log_params = log_params,
        self.models = models if models is not None else ['ER', 'WS', 'BA']
        self.parameters = parameters
        self.kwargs = kwargs
        self.validate_input()

    def validate_input(self):
        if not isinstance(self.graph, nx.Graph):
            raise ValueError("The input should be a networkx Graph object!")

    def model_function(self, model_name):
        if model_name == "ER":
            return lambda n, p: nx.erdos_renyi_graph(n, p)
        elif model_name == "GRG":
            return lambda n, r: nx.random_geometric_graph(n, r)
        elif model_name == "KR":
            return lambda n, k: nx.random_regular_graph(k, n)
        elif model_name == "WS":
            return lambda n, p, k=8: nx.watts_strogatz_graph(n, k, p)
        elif model_name == "BA":
            return lambda n, m: nx.barabasi_albert_graph(n, m)
        elif model_name == "LG":
            pass
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def select_model(self):
        results = []
        for idx, model in enumerate(self.models):
            print(f'testing the selected model for {model}')
            if model == "LG":
                min_gic = gic.GraphInformationCriterion(graph=self.graph, model=model, log_graph=self.log_graph)
                params = self.log_params
                result = {'param': params, 'gic': min_gic}

            else:
                if callable(model):
                    model_func = model
                else:
                    model_func = self.model_function(model)

                param = None
                if self.parameters:
                    param = self.parameters[idx]

                print(f"Model: {model}, Parameters: {param}")
                print(f"model function {model_func}")
                estimator = pe.GraphParameterEstimator(self.graph, model=model_func, interval=param, **self.kwargs)
                result = estimator.estimate()

            results.append((model, result['param'], result['gic']))

        # Sort results based on GIC value
        results.sort(key=lambda x: x[2])

        # Prepare the output
        best_model = results[0][0]
        estimates = np.array([(m, p, gic) for m, p, gic in results], dtype=[('model', 'U10'), ('param', 'float'), ('GIC', 'float')])

        return {
            "method": "Graph Model Selection",
            "info": "Selects the graph model that best approximates the observed graph.",
            "model": best_model,
            "estimates": estimates
        }

