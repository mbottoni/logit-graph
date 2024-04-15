
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp

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
        penalties = {'ER': 2, 'WS': 3, 'BA': 2, 'LG': 3}
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
                model_graph = model_func(n, k, p)
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

