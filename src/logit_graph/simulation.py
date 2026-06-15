from __future__ import annotations

import os
import gc
import math
import random
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import expit
from tqdm import tqdm

# Relative package imports
from . import graph
from .lg_features import FeatureMode, build_pair_dataset
from . import logit_estimator as estimator
from . import gic
from . import model_selection as ms


def _warm_start_er_p(sigma: float, lo: float = 0.02, hi: float = 0.5) -> float:
    """ER seed density clipped near the d>=1 equilibrium expit(sigma); starting the Gibbs
    chain here dramatically reduces burn-in for sparse regimes (very negative sigma)."""
    return float(np.clip(expit(sigma), lo, hi))


def _direct_er_at_sigma(n: int, sigma: float, seed: Optional[int]) -> np.ndarray:
    """Sample Erdős–Rényi adjacency at p = expit(sigma) — the exact equilibrium of d=0
    Gibbs with no degree feedback, used to short-circuit Gibbs sampling whenever d=0."""
    p = float(expit(sigma))
    rng = np.random.default_rng(seed)
    upper = rng.random((n, n)) < p
    upper = np.triu(upper, k=1)
    return (upper | upper.T).astype(float)



### Helper Functions ###
def _as_adj(graph_input: Union[np.ndarray, nx.Graph]) -> np.ndarray:
    if isinstance(graph_input, nx.Graph):
        return nx.to_numpy_array(graph_input)
    return np.asarray(graph_input, dtype=float)


def estimate_sigma_only(
    graph_input: Union[np.ndarray, nx.Graph],
    d: int,
    max_pairs: Optional[int] = None,
    feature_mode: FeatureMode = "incremental",
    random_state: Optional[int] = None,
    verbose: bool = False,
    # Deprecated kwargs kept for backwards compatibility ------------------
    max_edges: Optional[int] = None,
    max_non_edges: Optional[int] = None,
    l1_wt: float = 1,
    alpha: float = 0,
) -> tuple[float, Any]:
    """Estimate sigma via Layer-2 offset logit (paper formulation), using
    :func:`build_pair_dataset` (layer2=True) so it stays consistent with generation.
    Returns ``(sigma_hat, fit_result)``; legacy max_edges/max_non_edges/l1_wt/alpha unused."""
    del l1_wt, alpha  # accepted for backwards compatibility, unused

    if max_pairs is None and (max_edges is not None or max_non_edges is not None):
        max_pairs = (max_edges or 0) + (max_non_edges or 0) or None

    adj = _as_adj(graph_input)

    est = estimator.LogitRegEstimator(
        adj, d=d, layer2=True, feature_mode=feature_mode, verbose=verbose,
    )
    offsets, labels_arr = build_pair_dataset(
        adj, d=d, mode=feature_mode, layer2=True,
        max_pairs=max_pairs, seed=random_state,
    )
    result = est._fit_offset_logit(offsets, np.asarray(labels_arr, dtype=int))
    sigma = float(result.params[0])
    return sigma, result


def estimate_sigma_many(
    graph_input: Union[np.ndarray, nx.Graph],
    d: int,
    n_repeats: int = 30,
    max_pairs: Optional[int] = None,
    feature_mode: FeatureMode = "incremental",
    seed: int = 42,
    verbose: bool = False,
    # Deprecated kwargs kept for backwards compatibility ------------------
    max_edges: Optional[int] = None,
    max_non_edges: Optional[int] = None,
    l1_wt: float = 1,
    alpha: float = 0,
) -> list[float]:
    """Repeat Layer-2 sigma estimation ``n_repeats`` times with different seeds, returning
    a list of ``sigma_hat``. Variability comes from random pair sampling when ``max_pairs``
    is set; with ``max_pairs=None`` on a small graph every repetition is identical."""
    del l1_wt, alpha
    if max_pairs is None and (max_edges is not None or max_non_edges is not None):
        max_pairs = (max_edges or 0) + (max_non_edges or 0) or None

    sigmas: list[float] = []
    for r in tqdm(range(int(n_repeats))):
        rs = None if seed is None else (seed + r)
        sigma_r, _result = estimate_sigma_only(
            graph_input, d=d, max_pairs=max_pairs,
            feature_mode=feature_mode, random_state=rs, verbose=verbose,
        )
        sigmas.append(sigma_r)
    return sigmas

#####
#####

def calculate_graph_attributes(graph_to_analyze: Optional[nx.Graph]) -> dict[str, Any]:
    """Calculate various graph attributes for a given graph."""
    if graph_to_analyze is None or graph_to_analyze.number_of_nodes() == 0:
        return {attr: np.nan for attr in [
            'nodes', 'edges', 'density', 'avg_clustering', 'avg_path_length', 'diameter',
            'assortativity', 'num_components', 'largest_component_size'
        ]}
    
    attrs = {'nodes': graph_to_analyze.number_of_nodes(), 'edges': graph_to_analyze.number_of_edges()}
    
    try:
        attrs['density'] = nx.density(graph_to_analyze)
        attrs['avg_clustering'] = nx.average_clustering(graph_to_analyze)
        attrs['assortativity'] = nx.degree_assortativity_coefficient(graph_to_analyze)
        
        components = list(nx.connected_components(graph_to_analyze))
        attrs['num_components'] = len(components)
        if components:
            largest_cc = max(components, key=len)
            attrs['largest_component_size'] = len(largest_cc)
            subgraph = graph_to_analyze.subgraph(largest_cc)
            if len(largest_cc) > 1:
                attrs['avg_path_length'] = nx.average_shortest_path_length(subgraph)
                attrs['diameter'] = nx.diameter(subgraph)
            else:
                attrs['avg_path_length'] = 0
                attrs['diameter'] = 0
        else:
             attrs['largest_component_size'] = 0
             attrs['avg_path_length'] = np.nan
             attrs['diameter'] = np.nan
    
    except Exception as e:
        print(f"Could not calculate all attributes: {e}")

    return attrs

def clean_and_convert_param(param: Any) -> float:
    """Clean and convert parameter string to float."""
    if isinstance(param, (int, float)):
        return param
    cleaned_param = ''.join(c for c in str(param) if c.isdigit() or c == '.' or c == '-')
    try:
        return float(cleaned_param)
    except (ValueError, TypeError):
        return np.nan


class LogitGraphFitter:
    """
    Fits a single Logit Graph model to a real graph, following a scikit-learn-like API.
    """
    def __init__(
        self,
        d: int = 0,
        n_iteration: int = 10000,
        warm_up: int = 500,
        patience: int = 2000,
        dist_type: str = 'KL',
        edge_delta: Optional[float] = None,
        min_gic_threshold: float = 5,
        check_interval: int = 50,
        verbose: bool = True,
        er_p: float = 0.05,
        init_graph: Optional[nx.Graph] = None,
    ) -> None:
        """Initialize the LogitGraphFitter: d (latent depth), n_iteration cap, warm_up,
        patience (checks without improvement before stopping), dist_type for GIC,
        edge_delta / min_gic_threshold convergence thresholds, check_interval, verbose."""
        self.d = d
        self.n_iteration = n_iteration
        self.warm_up = warm_up
        self.er_p = er_p
        self.patience = patience
        self.dist_type = dist_type
        self.edge_delta = edge_delta
        self.min_gic_threshold = min_gic_threshold
        self.check_interval = check_interval
        self.verbose = verbose
        self.init_graph = init_graph
        
        self.fitted_graph = None
        self.metadata = {}

    def fit(self, original_graph: nx.Graph) -> LogitGraphFitter:
        """Fit the Logit Graph model to ``original_graph``; returns self with the
        ``fitted_graph`` and ``metadata`` attributes populated."""
        # Ensure we work with an undirected view for consistent edge counting and spectra
        # Many datasets load as directed; our fitter/populator assumes undirected adjacency.
        undirected_graph = original_graph.to_undirected()

        if self.verbose:
            print(f"\n{'='*20} Processing Graph {'='*20}")
            print(f"Original graph - Nodes: {undirected_graph.number_of_nodes()}, Edges: {undirected_graph.number_of_edges()}")

        self.metadata = {
            'original_nodes': undirected_graph.number_of_nodes(),
            'original_edges': undirected_graph.number_of_edges(),
            'fit_success': False,
            'error_message': None,
        }
        
        try:
            # Build adjacency from the undirected graph to avoid double-counting edges
            adj_matrix = nx.to_numpy_array(undirected_graph)
            
            best_graph_arr, sigma, gic_val, spectrum_diffs, edge_diffs, best_iter, all_graphs, gic_values = self._generate_graph(adj_matrix)
            
            self.fitted_graph = nx.from_numpy_array(best_graph_arr)
            
            self.metadata.update({
                'fit_success': True,
                'sigma': sigma,
                'gic_value': gic_val,
                'best_iteration': best_iter,
                'fitted_nodes': self.fitted_graph.number_of_nodes(),
                'fitted_edges': self.fitted_graph.number_of_edges(),
                'spectrum_diffs': spectrum_diffs,
                'edge_diffs': edge_diffs,
                'gic_values': gic_values,
            })
            
            if self.verbose:
                print(f"Fitting successful - GIC: {self.metadata['gic_value']:.4f}, Best iteration: {self.metadata['best_iteration']}")
                print(f"Fitted graph - Nodes: {self.metadata['fitted_nodes']}, Edges: {self.metadata['fitted_edges']}")
            
            del all_graphs
            gc.collect()

        except Exception as e:
            print(f"Error fitting graph: {e}")
            self.metadata['error_message'] = str(e)
            self.fitted_graph = None
        
        return self

    def _generate_graph(
        self, real_graph_arr: np.ndarray
    ) -> tuple[np.ndarray, float, float, list[float], list[float], int, list[np.ndarray], list[float]]:
        """
        Internal method to estimate parameters and generate the graph.
        """
        est = estimator.LogitRegEstimator(
            real_graph_arr, d=self.d, layer2=True, feature_mode="incremental",
        )
        features, labels = est.get_features_labels()
        _, params, _ = est.estimate_parameters(features=features, labels=labels)
        sigma = float(params[0])

        n = real_graph_arr.shape[0]

        if self.d == 0:
            best_graph_arr = _direct_er_at_sigma(n, sigma, seed=None)
            best_graph_nx = nx.from_numpy_array(best_graph_arr)
            gic_value = gic.GraphInformationCriterion(
                graph=nx.from_numpy_array(real_graph_arr),
                log_graph=best_graph_nx,
                model='LG',
                dist=self.dist_type,
            ).calculate_gic()
            real_edges = np.sum(real_graph_arr) / 2
            edge_diff = abs(np.sum(best_graph_arr) / 2 - real_edges)
            return (
                best_graph_arr, sigma, gic_value, [0.0], [edge_diff], 0,
                [best_graph_arr], [gic_value],
            )

        warm_start_p = (
            _warm_start_er_p(sigma) if self.init_graph is None else self.er_p
        )
        graph_model = graph.GraphModel(
            n=n, d=self.d, sigma=sigma, er_p=warm_start_p,
            init_graph=self.init_graph,
            layer2=True, feature_mode="incremental",
        )

        if self.verbose:
            print(f"Running LG generation for d={self.d}...")

        graphs, _, spectrum_diffs, best_iteration, best_graph_arr, gic_values = graph_model.populate_edges_spectrum_min_gic(
            er_p=warm_start_p,
            max_iterations=self.n_iteration,
            patience=self.patience,
            real_graph=real_graph_arr,
            edge_delta=self.edge_delta,
            min_gic_threshold=self.min_gic_threshold,
            gic_dist_type=self.dist_type,
            check_interval=self.check_interval,
            verbose=self.verbose,
        )
        
        best_graph_nx = nx.from_numpy_array(best_graph_arr)
        gic_value = gic.GraphInformationCriterion(
            graph=nx.from_numpy_array(real_graph_arr),
            log_graph=best_graph_nx,
            model='LG',
            dist=self.dist_type,
        ).calculate_gic()
        
        real_edges = np.sum(real_graph_arr) / 2
        edge_diffs = [abs(np.sum(g) / 2 - real_edges) for g in graphs]

        return best_graph_arr, sigma, gic_value, spectrum_diffs, edge_diffs, best_iteration, graphs, gic_values


class LogitGraphSimulation:
    """Simulate a Logit Graph directly from provided parameters (scikit-learn-like API):
    ``sim = LogitGraphSimulation(n=100, d=2, sigma=1.0, alpha=1.0, beta=1.0); sim.simulate()``,
    then read ``sim.simulated_graph`` and ``sim.metadata``."""
    def __init__(
        self,
        n: int,
        d: int,
        sigma: float,
        alpha: float = 1.0,
        beta: float = 1.0,
        er_p: float = 0.05,
        n_iteration: int = 10000,
        warm_up: int = 500,
        patience: int = 2000,
        check_interval: int = 50,
        edge_cv_tol: float = 0.02,
        spectrum_cv_tol: float = 0.02,
        verbose: bool = True,
        init_graph: Optional[nx.Graph] = None,
        layer2: bool = True,
        feature_mode: FeatureMode = "bounded",
        fast_mode: bool = False,
    ) -> None:
        """Initialize the LogitGraphSimulation: model params (n, d, sigma, alpha, beta,
        er_p) and run params (n_iteration, warm_up, patience window, check_interval,
        edge/spectrum CV tolerances, verbose, init_graph)."""
        self.n = n
        self.d = d
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.er_p = er_p
        self.n_iteration = n_iteration
        self.warm_up = warm_up
        self.patience = patience
        self.check_interval = check_interval
        self.edge_cv_tol = edge_cv_tol
        self.spectrum_cv_tol = spectrum_cv_tol
        self.verbose = verbose
        self.init_graph = init_graph
        self.layer2 = layer2
        self.feature_mode = feature_mode
        self.fast_mode = fast_mode

        self.simulated_graph = None
        self.metadata = {}

    def simulate(self) -> LogitGraphSimulation:
        """Run the simulation, storing the graph and metadata. d=0 samples directly from
        ER(p=expit(sigma)) (exact equilibrium); d>=1 warm-starts the Gibbs chain near
        expit(sigma) clipped to [0.02, 0.5], unless init_graph or a non-default er_p overrides."""
        if self.verbose:
            print(f"\n{'='*20} Simulating Logit Graph {'='*20}")
            print(f"Parameters - n: {self.n}, d: {self.d}, sigma: {self.sigma:.4f}, alpha: {self.alpha:.4f}, beta: {self.beta:.4f}, er_p: {self.er_p}")

        self.metadata = {
            'simulate_success': False,
            'error_message': None,
            'n': self.n,
            'd': self.d,
            'sigma': float(self.sigma),
            'alpha': float(self.alpha),
            'beta': float(self.beta),
            'er_p': float(self.er_p),
        }

        try:
            if self.d == 0:
                final_graph_arr = _direct_er_at_sigma(self.n, self.sigma, seed=None)
                self.simulated_graph = nx.from_numpy_array(final_graph_arr)
                self.metadata.update({
                    'simulate_success': True,
                    'iterations_ran': 0,
                    'final_nodes': self.simulated_graph.number_of_nodes(),
                    'final_edges': self.simulated_graph.number_of_edges(),
                    'final_spectrum': None,
                    'sampler': 'direct_er',
                })
                if self.verbose:
                    print(
                        f"d=0: direct ER at p=expit({self.sigma:.3f})={expit(self.sigma):.4f} — "
                        f"Nodes={self.metadata['final_nodes']}, Edges={self.metadata['final_edges']}"
                    )
                return self

            user_override = self.init_graph is not None or not math.isclose(self.er_p, 0.05)
            warm_start_p = self.er_p if user_override else _warm_start_er_p(self.sigma)

            graph_model = graph.GraphModel(
                n=self.n, d=self.d, sigma=self.sigma, alpha=self.alpha, beta=self.beta,
                er_p=warm_start_p, init_graph=self.init_graph,
                layer2=self.layer2, feature_mode=self.feature_mode,
            )

            graphs, spectra = graph_model.populate_edges_baseline(
                warm_up=self.warm_up, max_iterations=self.n_iteration,
                patience=self.patience, check_interval=self.check_interval,
                edge_cv_tol=self.edge_cv_tol, spectrum_cv_tol=self.spectrum_cv_tol,
                fast_mode=self.fast_mode,
            )

            final_graph_arr = graphs[-1] if graphs else graph_model.graph
            self.simulated_graph = nx.from_numpy_array(final_graph_arr)

            self.metadata.update({
                'simulate_success': True,
                'iterations_ran': max(0, len(graphs) - 1),
                'final_nodes': self.simulated_graph.number_of_nodes(),
                'final_edges': self.simulated_graph.number_of_edges(),
                'final_spectrum': spectra,
                'warm_start_er_p': float(warm_start_p),
                'sampler': 'gibbs_layer2',
            })

            if self.verbose:
                print(
                    f"Simulation successful (warm_start_p={warm_start_p:.4f}) — "
                    f"Nodes: {self.metadata['final_nodes']}, Edges: {self.metadata['final_edges']}"
                )

        except Exception as e:
            print(f"Error simulating logit graph: {e}")
            self.metadata['error_message'] = str(e)
            self.simulated_graph = None

        return self


class GraphModelComparator:
    """
    Compares Logit Graph with other random graph models, with a scikit-learn-like API.
    """
    def __init__(
        self,
        d_list: list[int],
        lg_params: dict[str, Any],
        other_model_n_runs: int = 2,
        other_model_params: Optional[list[dict[str, Any]]] = None,
        dist_type: str = 'KL',
        verbose: bool = True,
        other_models: Optional[list[str]] = None,
        other_model_grid_points: int = 5,
        random_state: Optional[int] = 42,
    ) -> None:
        """Initialize the GraphModelComparator: d_list to test, lg_params (LG fitting),
        other_model_n_runs / other_model_params for the baselines, dist_type for GIC,
        verbose, and random_state (None for non-reproducible runs)."""
        self.d_list = d_list
        self.lg_params = lg_params
        self.other_model_n_runs = other_model_n_runs
        self.random_state = random_state
        # ``other_model_params`` may be None (per-model defaults looked up by name in
        # ``_fit_other_models``; recommended), a list (positional pairing with
        # ``other_models``; legacy), or a dict keyed by model name (explicit overrides).
        self.other_model_params = other_model_params
        self.dist_type = dist_type
        self.verbose = verbose
        # Allow selecting which other models to evaluate and how dense the parameter grid should be
        # Default excludes GRG for speed; user can include it by passing other_models.
        self.other_models = other_models if other_models is not None else ["ER", "WS", "BA"]
        self.other_model_grid_points = other_model_grid_points
        
        self.summary_df = None
        self.fitted_graphs_data = {}

    def compare(self, original_graph: nx.Graph, graph_filepath: str) -> GraphModelComparator:
        """Fit LG and the other models to ``original_graph`` (``graph_filepath`` used only
        for logging) and compare them; returns self with ``summary_df`` and
        ``fitted_graphs_data`` populated."""
        if self.verbose:
            print(f"\n{'='*30} Processing Graph: {os.path.basename(graph_filepath)} {'='*30}")

        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        
        self.fitted_graphs_data = {
            'Original': {
                'graph': original_graph,
                'metadata': {'fit_success': True, 'param': 'N/A', 'gic_value': np.nan}
            }
        }
        adj_matrix = nx.to_numpy_array(original_graph)

        # 1. Fit Logit Graph (LG) model, finding the best `d`
        self._fit_best_lg(adj_matrix)

        # 2. Fit other random graph models
        self._fit_other_models(original_graph)

        # 3. Calculate attributes and build the summary DataFrame
        self._build_summary_df(graph_filepath)
        
        return self

    def _fit_best_lg(self, adj_matrix: np.ndarray) -> None:
        if self.verbose:
            print("\n--- Fitting Logit Graph (LG) model ---")
        
        best_lg_fit = {'gic': np.inf}
        #TODO: Add multithread
        for d in self.d_list:
            try:
                lg_arr, sigma, gic_val, _, gic_values, spectrum_diffs, edge_diffs = self._get_logit_graph_for_d(adj_matrix, d)
                if self.verbose:
                    print(f"d={d}: GIC={gic_val:.4f}, sigma={sigma:.4f}")
                if gic_val < best_lg_fit['gic']:
                    best_lg_fit = {
                        'gic': gic_val,
                        'graph': nx.from_numpy_array(lg_arr),
                        'param': f"d={d}, sigma={sigma:.4f}",
                        'sigma': sigma,
                        'd': d,
                        'min_gic_value': min(gic_values) if gic_values else np.nan,
                        'min_gic_iteration': gic_values.index(min(gic_values)) if gic_values else -1,
                        'spectrum_diffs': spectrum_diffs,
                        'edge_diffs': edge_diffs,
                        'gic_values': gic_values
                    }
            except Exception as e:
                if self.verbose:
                    print(f"Failed to fit LG for d={d}: {e}")

        if 'graph' in best_lg_fit:
            self.fitted_graphs_data['LG'] = {
                'graph': best_lg_fit['graph'],
                'metadata': {
                    'fit_success': True,
                    'param': best_lg_fit['param'],
                    'gic_value': best_lg_fit['gic'],
                    'd': best_lg_fit['d'],
                    'sigma': best_lg_fit['sigma'],
                    'min_gic_value': best_lg_fit['min_gic_value'],
                    'min_gic_iteration': best_lg_fit['min_gic_iteration'],
                    'spectrum_diffs': best_lg_fit['spectrum_diffs'],
                    'edge_diffs': best_lg_fit['edge_diffs'],
                    'gic_values': best_lg_fit['gic_values'],
                }
            }
            if self.verbose:
                print(f"Best LG fit found with GIC: {best_lg_fit['gic']:.4f}")
        else:
            self.fitted_graphs_data['LG'] = {'graph': None, 'metadata': {'fit_success': False, 'param': 'N/A', 'gic_value': np.inf}}
            if self.verbose:
                print("LG fitting failed for all values of d.")

    def _get_logit_graph_for_d(
        self, real_graph: Union[np.ndarray, nx.Graph], d: int
    ) -> tuple[np.ndarray, float, float, int, list[float], list[float], list[float]]:
        """Estimate parameters and generate a graph for a specific `d`. d=0 short-circuits
        to a direct ER sample at p=expit(sigma_hat); d>=1 warm-starts Gibbs at
        clip(expit(sigma_hat), 0.02, 0.5) unless the caller supplied an ``init_graph``."""
        if isinstance(real_graph, nx.Graph):
            real_graph = nx.to_numpy_array(real_graph)

        est = estimator.LogitRegEstimator(
            real_graph, d=d, layer2=True, feature_mode="incremental",
        )
        features, labels = est.get_features_labels()
        _, params, _ = est.estimate_parameters(features=features, labels=labels)
        sigma = float(params[0])

        n = real_graph.shape[0]
        init_graph = self.lg_params.get('init_graph') if isinstance(self.lg_params, dict) else None

        lg_seed = None if self.random_state is None else self.random_state + d

        if d == 0:
            best_graph_arr = _direct_er_at_sigma(n, sigma, seed=lg_seed)
            best_graph_nx = nx.from_numpy_array(best_graph_arr)
            gic_value = gic.GraphInformationCriterion(
                graph=nx.from_numpy_array(real_graph),
                log_graph=best_graph_nx,
                model='LG',
                dist=self.dist_type,
            ).calculate_gic()
            real_edges = np.sum(real_graph) / 2
            edge_diff = abs(np.sum(best_graph_arr) / 2 - real_edges)
            return (
                best_graph_arr, sigma, gic_value, 0, [gic_value],
                [0.0], [edge_diff],
            )

        warm_start_p = (
            _warm_start_er_p(sigma) if init_graph is None
            else float(self.lg_params.get('er_p', 0.05))
        )
        graph_model = graph.GraphModel(
            n=n, d=d, sigma=sigma, er_p=warm_start_p,
            init_graph=init_graph,
            layer2=True, feature_mode="incremental",
            seed=lg_seed,
        )

        if self.verbose:
            print(f"Running LG generation for d={d} (warm_start_p={warm_start_p:.4f})...")

        lg_params = self.lg_params.copy()
        lg_params['er_p'] = warm_start_p  # propagate warm-start into populate_edges
        lg_params['gic_dist_type'] = self.dist_type
        lg_params['verbose'] = self.verbose

        graphs, _, spectrum_diffs, best_iteration, best_graph_arr, gic_values = graph_model.populate_edges_spectrum_min_gic(
            real_graph=real_graph,
            **lg_params
        )
        
        best_graph_nx = nx.from_numpy_array(best_graph_arr)
        gic_value = gic.GraphInformationCriterion(
            graph=nx.from_numpy_array(real_graph),
            log_graph=best_graph_nx,
            model='LG',
            dist=self.dist_type,
        ).calculate_gic()

        real_edges = np.sum(real_graph) / 2
        edge_diffs = [abs(np.sum(g) / 2 - real_edges) for g in graphs]

        return best_graph_arr, sigma, gic_value, best_iteration, gic_values, spectrum_diffs, edge_diffs

    def _fit_other_models(self, original_graph: nx.Graph) -> None:
        if self.verbose:
            print("\n--- Fitting other random graph models ---")
        
        lg_graph_for_selection = self.fitted_graphs_data['LG']['graph'] if self.fitted_graphs_data['LG']['metadata']['fit_success'] else original_graph
        lg_metadata = self.fitted_graphs_data['LG']['metadata']
        
        # This part requires parsing sigma from the 'param' string, which is brittle.
        # A better approach would be to store sigma directly in metadata.
        # For now, we attempt to parse it.
        sigma = 1.0 
        if lg_metadata.get('fit_success'):
            try:
                # Example param string: "d=2, sigma=1.2345"
                param_str = lg_metadata.get('param', '')
                sigma_part = [p for p in param_str.split(',') if 'sigma' in p]
                if sigma_part:
                    sigma = float(sigma_part[0].split('=')[1])
            except (ValueError, IndexError):
                if self.verbose:
                    print("Could not parse sigma from LG metadata, defaulting to 1.0")
        
        default_params = {
            'ER': {'lo': 0.01, 'hi': 0.25},
            'WS': {'k': {'lo': 2, 'hi': 10, 'step': 2}, 'p': {'lo': 0.01, 'hi': 0.5}},
            'GRG': {'lo': 0.05, 'hi': 0.3},
            'BA': {'lo': 1, 'hi': 8},
            # SBM has no scalar grid parameter — Louvain on the observed
            # graph fixes the block structure. The dispatch in
            # ``GraphModelSelection`` special-cases SBM and ignores this.
            'SBM': None,
        }
        default_param_map: dict[str, Any] = {}
        if self.other_model_params is None:
            # No overrides: look up each model's range by name.
            for model in self.other_models:
                if model in default_params:
                    default_param_map[model] = default_params[model]
        elif isinstance(self.other_model_params, dict):
            # Explicit name->params dict; fall back to defaults for any missing key.
            for model in self.other_models:
                if model in self.other_model_params:
                    default_param_map[model] = self.other_model_params[model]
                elif model in default_params:
                    default_param_map[model] = default_params[model]
        else:
            # Legacy positional list — must be in the same order as ``self.other_models``.
            for i, model in enumerate(self.other_models):
                if i < len(self.other_model_params):
                    default_param_map[model] = self.other_model_params[i]
                elif model in default_params:
                    default_param_map[model] = default_params[model]
        filtered_params = [default_param_map[m] for m in self.other_models if m in default_param_map]

        selector = ms.GraphModelSelection(
            graph=original_graph,
            log_graphs=[lg_graph_for_selection],
            log_params=[sigma],
            models=self.other_models,
            n_runs=self.other_model_n_runs,
            parameters=filtered_params,
            grid_points=self.other_model_grid_points,
            random_state=self.random_state,
        )
        
        model_results = selector.select_model_avg_spectrum()

        for estimate in model_results['estimates']:
            model_name = estimate['model']
            if model_name == 'LG':
                continue
            if model_name == 'SBM':
                # SBM has no scalar parameter — Louvain on G_real fixes
                # the block structure. ``_generate_graph`` ignores the
                # passed ``param`` for SBM.
                fitted_graph = selector._generate_graph(model_name, None, seed_offset=0)
                gic_value = estimate['GIC']
                self.fitted_graphs_data[model_name] = {
                    'graph': fitted_graph,
                    'metadata': {'fit_success': True, 'param': 'Louvain-fit',
                                 'gic_value': gic_value},
                }
                if self.verbose:
                    print(f"{model_name} fitting - GIC: {gic_value:.4f}, "
                          f"Param: Louvain-fit")
                continue
            param = clean_and_convert_param(estimate['param'])
            gic_value = estimate['GIC']

            fitted_graph = selector._generate_graph(model_name, param, seed_offset=0)

            self.fitted_graphs_data[model_name] = {
                'graph': fitted_graph,
                'metadata': {'fit_success': True, 'param': param, 'gic_value': gic_value},
            }
            if self.verbose:
                print(f"{model_name} fitting - GIC: {gic_value:.4f}, Param: {param:.4f}")

    def _build_summary_df(self, graph_filepath: str) -> None:
        if self.verbose:
            print("\n--- Calculating graph attributes ---")
        
        df_rows = []
        for model_name, data in self.fitted_graphs_data.items():
            row = {'graph_filename': os.path.basename(graph_filepath), 'model': model_name}
            row.update(data['metadata'])
            
            attributes = calculate_graph_attributes(data['graph'])
            data['attributes'] = attributes
            row.update(attributes)
            
            df_rows.append(row)
            
        summary_df = pd.DataFrame(df_rows)
        cols = ['graph_filename', 'model', 'gic_value', 'param', 'fit_success', 
                'nodes', 'edges', 'density', 'avg_clustering', 'avg_path_length', 
                'diameter', 'assortativity', 'num_components', 'largest_component_size']
        
        for col in cols:
            if col not in summary_df.columns:
                summary_df[col] = np.nan
        
        self.summary_df = summary_df[cols]

