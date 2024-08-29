import networkx as nx
import numpy as np
import random
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
# Baseline
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
# Baseline for kl div
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


##################################################
##################################################

# Class implementing the stat graph model selector
class GraphModelSelection:
    def __init__(self, graph, log_graphs, log_params, models=None, parameters=None, n_runs=10, **kwargs):
        self.graph = graph
        self.log_graphs = log_graphs
        self.log_params = log_params
        self.models = models if models is not None else ['ER', 'WS', 'BA']
        self.parameters = parameters
        self.n_runs = n_runs
        self.kwargs = kwargs
        self.validate_input()

    def validate_input(self):
        if not isinstance(self.graph, nx.Graph):
            raise ValueError("The input should be a networkx Graph object!")
        if not isinstance(self.log_graphs, list) or len(self.log_graphs) == 0:
            raise ValueError("log_graphs should be a non-empty list of networkx Graph objects!")

    def model_function(self, model_name):
        if model_name == "ER":
            return lambda n, p: nx.erdos_renyi_graph(n, p)
        elif model_name == "GRG":
            return lambda n, r: nx.random_geometric_graph(n, r)
        elif model_name == "KR":
            return lambda n, k: nx.random_regular_graph(d=2*int(k), n=n)
        elif model_name == "WS":
            return lambda n, p, k=8: nx.watts_strogatz_graph(n, k, p)
        elif model_name == "BA":
            return lambda n, m: nx.barabasi_albert_graph(n, int(m))
        elif model_name == "LG":
            pass
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def calculate_average_spectrum(self, model, params):
        spectrum_values = []
        if model == "LG":
            # For LG model, we already have n_runs graphs
            for log_graph in self.log_graphs:
                gic_calculator = gic.GraphInformationCriterion(
                    graph=self.graph,
                    model=model,
                    log_graph=log_graph
                )
                spectrum, _ = gic_calculator.compute_spectral_density(log_graph)
                spectrum_values.append(spectrum)
        else:
            model_func = self.model_function(model)
            for _ in range(self.n_runs):
                # Convert params to appropriate type based on the model
                if model in ["BA", "WS"]:
                    model_params = int(params)
                elif model in ["ER", "GRG"]:
                    model_params = float(params)
                else:
                    model_params = params

                generated_graph = model_func(self.graph.number_of_nodes(), model_params)
                gic_calculator = gic.GraphInformationCriterion(
                    graph=self.graph,
                    model=model,
                    log_graph=generated_graph,
                    p=params
                )
                spectrum, _ = gic_calculator.compute_spectral_density(generated_graph)
                spectrum_values.append(spectrum)
        
        # Calculate the mean of the spectral densities
        mean_spectrum = np.mean(spectrum_values, axis=0)
        return mean_spectrum

    def select_model_avg_spectrum(self):
        results = []
        for idx, model in enumerate(self.models):
            print(f'Testing the selected model for {model}')
            if model == "LG":
                avg_spectrum = self.calculate_average_spectrum(model, None)
                params = self.log_params
                result = {'param': params, 'spectrum': avg_spectrum}
                print('LG result:', result)
            else:
                if callable(model):
                    model_func = model
                else:
                    model_func = self.model_function(model)
                
                param_range = self.parameters[idx]
                best_param = None
                best_spectrum = None
                best_distance = float('inf')
                
                for param in np.linspace(param_range['lo'], param_range['hi'], num=10):
                    avg_spectrum = self.calculate_average_spectrum(model, param)
                    real_spectrum, _ = gic.GraphInformationCriterion(self.graph, model).compute_spectral_density(self.graph)
                    distance = np.linalg.norm(real_spectrum - avg_spectrum)
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_param = param
                        best_spectrum = avg_spectrum
                
                result = {'param': best_param, 'spectrum': best_spectrum}
                print(f'{model} result:', result)
            
            results.append((model, result['param'], best_distance))

        # Sort results based on spectral distance
        results.sort(key=lambda x: x[2])

        # Prepare the output
        best_model = results[0][0]
        estimates = np.array([(m, str(p), d) for m, p, d in results], dtype=[('model', 'U10'), ('param', 'U20'), ('distance', 'float')])

        return {
            "method": "Graph Model Selection",
            "info": "Selects the graph model that best approximates the observed graph based on average spectral distance over multiple runs.",
            "model": best_model,
            "estimates": estimates
        }

    def calculate_average_gic(self, model, params):
        gic_values = []
        for _ in range(self.n_runs):
            if model == "LG":
                log_graph = random.choice(self.log_graphs)
                gic_calculator = gic.GraphInformationCriterion(
                    graph=self.graph,
                    model=model,
                    log_graph=log_graph
                )
            else:
                model_func = self.model_function(model)
                # Convert params to appropriate type based on the model
                if model in ["BA", "WS"]:
                    # BA and WS models require integer parameters
                    model_params = int(params)
                elif model in ["ER", "GRG"]:
                    # ER and GRG models use float parameters
                    model_params = float(params)
                else:
                    model_params = params

                generated_graph = model_func(self.graph.number_of_nodes(), model_params)
                gic_calculator = gic.GraphInformationCriterion(
                    graph=self.graph,
                    model=model,
                    log_graph=generated_graph,
                    p=params
                )
            
            gic_value = gic_calculator.calculate_gic()
            gic_values.append(gic_value)
        
        return np.mean(gic_values)

    def select_model_avg_gic(self):
        results = []
        for idx, model in enumerate(self.models):
            print(f'Testing the selected model for {model}')
            if model == "LG":
                avg_gic = self.calculate_average_gic(model, None)
                params = self.log_params
                result = {'param': params, 'gic': avg_gic}
                print('LG result:', result)
            else:
                if callable(model):
                    model_func = model
                else:
                    model_func = self.model_function(model)

                param = None
                if self.parameters:
                    param = self.parameters[idx]

                print(f"Model: {model}, Parameters: {param}")
                print(f"Model function: {model_func}")
                estimator = pe.GraphParameterEstimator(self.graph, model=model_func, interval=param, **self.kwargs)
                result = estimator.estimate()
                
                # Calculate average GIC over n_runs
                avg_gic = self.calculate_average_gic(model, result['param'])
                result['gic'] = avg_gic
                
                print(f'{model} result: ', result)

            results.append((model, result['param'], result['gic']))

        # Sort results based on GIC value
        results.sort(key=lambda x: x[2])

        # Prepare the output
        best_model = results[0][0]
        estimates = np.array([(m, str(p), gic) for m, p, gic in results], dtype=[('model', 'U10'), ('param', 'U20'), ('GIC', 'float')])

        return {
            "method": "Graph Model Selection",
            "info": "Selects the graph model that best approximates the observed graph based on average GIC over multiple runs.",
            "model": best_model,
            "estimates": estimates
        }
