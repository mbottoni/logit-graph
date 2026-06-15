import random
from typing import Optional

import networkx as nx
import numpy as np
from scipy.stats import ks_2samp

from . import gic
from . import param_estimator as pe


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

# Class implementing the MAIN stat graph model selector
class GraphModelSelection:
    def __init__(self, graph, log_graphs, log_params, models=None, parameters=None, n_runs=10, grid_points=10, random_state=None, **kwargs):
        self.graph = graph
        self.log_graphs = log_graphs
        self.log_params = log_params
        self.models = models if models is not None else ['ER', 'WS', 'BA']
        self.parameters = parameters
        self.n_runs = n_runs
        self.grid_points = grid_points
        self.random_state = random_state
        self.kwargs = kwargs
        self.validate_input()

    def _run_seed(self, seed_offset: int) -> Optional[int]:
        if self.random_state is None:
            return None
        return int(self.random_state) + 10_000 + seed_offset

    def _generate_graph(self, model: str, params, seed_offset: int):
        n = self.graph.number_of_nodes()
        seed = self._run_seed(seed_offset)
        if model == "ER":
            return nx.erdos_renyi_graph(n, float(params), seed=seed)
        if model == "GRG":
            return nx.random_geometric_graph(n, float(params), seed=seed)
        if model == "KR":
            return nx.random_regular_graph(d=2 * int(params), n=n, seed=seed)
        if model == "WS":
            if isinstance(params, (list, tuple, np.ndarray)) and len(params) >= 2:
                return nx.watts_strogatz_graph(n, int(params[0]), float(params[1]), seed=seed)
            avg_degree = 2 * self.graph.number_of_edges() / self.graph.number_of_nodes()
            k = max(2, int(avg_degree))
            if k % 2 != 0:
                k += 1
            return nx.watts_strogatz_graph(n, k, float(params), seed=seed)
        if model == "BA":
            return nx.barabasi_albert_graph(n, int(params), seed=seed)
        if model == "SBM":
            # SBM has no scalar grid parameter — Louvain on the observed
            # graph fixes the block structure; ``params`` is ignored.
            from .sbm import generate_sbm_from_real

            G_sbm, _ = generate_sbm_from_real(self.graph, seed=seed)
            return G_sbm
        raise ValueError(f"Unknown model: {model}")

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
            def ws_wrapper(n, params):
                if isinstance(params, (list, tuple, np.ndarray)) and len(params) >= 2:
                    # Multi-parameter case: [k, p]
                    return nx.watts_strogatz_graph(n, int(params[0]), params[1])
                else:
                    # Single parameter case (fallback) - assume it's p and use default k
                    avg_degree = 2 * self.graph.number_of_edges() / self.graph.number_of_nodes()
                    k = max(2, int(avg_degree))
                    if k % 2 != 0:  # Ensure k is even
                        k += 1
                    return nx.watts_strogatz_graph(n, k, float(params))
            return ws_wrapper
        elif model_name == "BA":
            return lambda n, m: nx.barabasi_albert_graph(n, int(m))
        elif model_name == "SBM":
            # Closure over self.graph: SBM is fit on the observed graph
            # (Louvain communities) rather than parameterised by n alone.
            from .sbm import generate_sbm_from_real

            def sbm_wrapper(n, _params):
                G_sbm, _ = generate_sbm_from_real(self.graph, seed=None)
                return G_sbm
            return sbm_wrapper
        elif model_name == "LG":
            pass
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def calculate_average_spectrum(self, model, params, seed_offset: int = 0):
        spectrum_values = []
        if model == "LG":
            # For LG model, we already have n_runs graphs
            for log_graph in self.log_graphs[-self.n_runs:]: # select the last n_runs graphs
                gic_calculator = gic.GraphInformationCriterion(
                    graph=self.graph,
                    model=model,
                    log_graph=log_graph
                )
                spectrum, _ = gic_calculator.compute_spectral_density(log_graph)
                spectrum_values.append(spectrum)
        else:
            for run_idx in range(self.n_runs):
                # Handle both single and multi-parameter models
                if model == "WS" and isinstance(params, list):
                    generated_graph = self._generate_graph(model, params, seed_offset + run_idx)
                    param_for_gic = params
                else:
                    if model in ["BA"]:
                        model_params = int(params)
                    elif model in ["ER", "GRG"]:
                        model_params = float(params)
                    else:
                        model_params = params

                    generated_graph = self._generate_graph(model, model_params, seed_offset + run_idx)
                    param_for_gic = params

                gic_calculator = gic.GraphInformationCriterion(
                    graph=self.graph,
                    model=model,
                    log_graph=generated_graph,
                    p=param_for_gic
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
            if model == "SBM":
                # No scalar grid — Louvain on the observed graph fixes the
                # block structure. Average across ``n_runs`` re-samples.
                grid_offset = idx * 1000
                avg_spectrum = self.calculate_average_spectrum(
                    model, None, seed_offset=grid_offset,
                )
                real_spectrum, _ = gic.GraphInformationCriterion(
                    self.graph, model,
                ).compute_spectral_density(self.graph)
                distance = float(np.linalg.norm(real_spectrum - avg_spectrum))
                # SBM fits k(k+1)/2 block-edge probabilities (k Louvain communities).
                # Penalize by that real count, not the _get_n_params() placeholder of 1,
                # else the GIC treats a ~k²-parameter block model as cheaply as ER.
                from .sbm import fit_sbm_from_graph
                sbm_sizes, _, _ = fit_sbm_from_graph(self.graph, seed=grid_offset)
                sbm_k = len(sbm_sizes)
                sbm_n_params = sbm_k * (sbm_k + 1) // 2
                current_gic = gic.GraphInformationCriterion(
                    graph=self.graph, model=model, p=None,
                ).calculate_gic(model_den=avg_spectrum, n_params=sbm_n_params)
                print(f'{model} gic: {current_gic} (k={sbm_k}, n_params={sbm_n_params})')
                results.append((model, f"Louvain-fit (k={sbm_k})", distance, current_gic))
                continue
            if model == "LG":
                avg_spectrum = self.calculate_average_spectrum(model, None)
                params = self.log_params
                
                # Calculate GIC for LG model
                gic_calculator = gic.GraphInformationCriterion(
                    graph=self.graph,
                    model=model,
                    log_graph=self.log_graphs[0],
                )
                lg_gic = gic_calculator.calculate_gic(model_den=avg_spectrum)
                
                real_spectrum, _ = gic_calculator.compute_spectral_density(self.graph)
                distance = np.linalg.norm(real_spectrum - avg_spectrum)
                
                result = {'param': params, 'spectrum': avg_spectrum, 'gic': lg_gic}
                print(f'LG gic: {lg_gic}')
                results.append((model, params, distance, lg_gic))
            else:
                if callable(model):
                    model_func = model
                else:
                    model_func = self.model_function(model)
                
                param_range = self.parameters[idx]
                best_param = None
                best_spectrum = None
                best_distance = float('inf')
                best_gic = float('inf')
                grid_offset = idx * 1000
                
                # Handle multi-parameter models (like WS)
                if model == "WS" and isinstance(param_range, dict) and 'k' in param_range and 'p' in param_range:
                    # Multi-parameter case for WS
                    k_values = np.arange(param_range['k']['lo'], param_range['k']['hi'], 
                                       param_range['k'].get('step', 2))
                    p_values = np.linspace(param_range['p']['lo'], param_range['p']['hi'], num=self.grid_points)
                    
                    combo_idx = 0
                    for k in k_values:
                        for p in p_values:
                            params = [int(k), float(p)]
                            avg_spectrum = self.calculate_average_spectrum(
                                model, params, seed_offset=grid_offset + combo_idx,
                            )
                            combo_idx += 1
                            real_spectrum, _ = gic.GraphInformationCriterion(self.graph, model).compute_spectral_density(self.graph)
                            distance = np.linalg.norm(real_spectrum - avg_spectrum)
                            
                            # Calculate GIC
                            gic_calculator = gic.GraphInformationCriterion(
                                graph=self.graph,
                                model=model,
                                p=params
                            )
                            current_gic = gic_calculator.calculate_gic(model_den=avg_spectrum)
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_param = params
                                best_spectrum = avg_spectrum
                                best_gic = current_gic
                else:
                    # Single parameter case
                    for grid_idx, param in enumerate(
                        np.linspace(param_range['lo'], param_range['hi'], num=self.grid_points)
                    ):
                        avg_spectrum = self.calculate_average_spectrum(
                            model, param, seed_offset=grid_offset + grid_idx,
                        )
                        real_spectrum, _ = gic.GraphInformationCriterion(self.graph, model).compute_spectral_density(self.graph)
                        distance = np.linalg.norm(real_spectrum - avg_spectrum)
                        
                        # Calculate GIC
                        gic_calculator = gic.GraphInformationCriterion(
                            graph=self.graph,
                            model=model,
                            p=param
                        )
                        current_gic = gic_calculator.calculate_gic(model_den=avg_spectrum)
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_param = param
                            best_spectrum = avg_spectrum
                            best_gic = current_gic
                
                result = {'param': best_param, 'spectrum': best_spectrum, 'gic': best_gic}
                print(f'{model} gic: {best_gic}')
                results.append((model, best_param, best_distance, best_gic))

        # Sort results based on GIC
        results.sort(key=lambda x: x[3])

        # Prepare the output
        best_model = results[0][0]
        estimates = np.array([(m, str(p), d, g) for m, p, d, g in results], 
                             dtype=[('model', 'U10'), ('param', 'U20'), ('distance', 'float'), ('GIC', 'float')])

        return {
            "method": "Graph Model Selection",
            "info": "Selects the graph model that best approximates the observed graph based on average spectral distance over multiple runs.",
            "model": best_model,
            "estimates": estimates
        }
    
    # Calculate average GIC over n_runs
    def calculate_average_gic(self, model, params, seed_offset: int = 0):
        gic_values = []
        for run_idx in range(self.n_runs):
            if model == "LG":
                log_graph = self.log_graphs[0]
                gic_calculator = gic.GraphInformationCriterion(
                    graph=self.graph,
                    model=model,
                    log_graph=log_graph
                )
            else:
                generated_graph = self._generate_graph(model, params, seed_offset + run_idx)
                gic_calculator = gic.GraphInformationCriterion(
                    graph=self.graph,
                    model=model,
                    log_graph=generated_graph,
                    p=params
                )
            
            gic_value = gic_calculator.calculate_gic()
            gic_values.append(gic_value)
        
        return np.mean(gic_values)

    # Select the model based on average GIC over n_runs
    def select_model_avg_gic(self):
        results = []
        for idx, model in enumerate(self.models):
            print(f'Testing the selected model for {model}')
            if model == "SBM":
                # No grid — Louvain on the observed graph fixes blocks.
                avg_gic = self.calculate_average_gic(model, None)
                result = {'param': 'Louvain-fit', 'gic': avg_gic}
                print(f'{model} result:', result)
                results.append((model, result['param'], result['gic']))
                continue
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
                
                # Handle multi-parameter models differently
                if model == "WS":
                    # Use custom parameter estimation for WS
                    estimator = pe.GraphParameterEstimator(self.graph, model=model, interval=param, **self.kwargs)
                else:
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
