
import sys
import os

# Graph imports (relative within package)
from . import graph
from . import logit_estimator as estimator
from . import utils
from . import model_selection
from . import gic
from . import param_estimator as pe
from . import graph
from . import model_selection as ms

# usual imports
import matplotlib.pyplot as plt
import pickle
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import seaborn as sns
import gc
import random
import networkx as nx
from numpy import errstate

from IPython.display import display
from pyvis.network import Network
from mpl_toolkits.axes_grid1 import make_axes_locatable



### Helper Functions ###
def calculate_graph_attributes(graph_to_analyze):
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

def clean_and_convert_param(param):
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
    def __init__(self, d=0, n_iteration=10000, warm_up=500, patience=2000,
                 dist_type='KL', edge_delta=None, min_gic_threshold=5, verbose=True, er_p=0.05):
        """
        Initializes the LogitGraphFitter with model parameters.

        Args:
            d (int): The dimension of the latent space.
            n_iteration (int): The maximum number of iterations for edge generation.
            warm_up (int): The number of warm-up iterations.
            patience (int): The number of iterations to wait for improvement before stopping.
            dist_type (str): The distance type for GIC calculation ('KL', etc.).
            edge_delta (float, optional): Edge convergence threshold. Defaults to None.
            min_gic_threshold (float, optional): GIC convergence threshold. Defaults to 5.
            verbose (bool): Whether to print progress information.
        """
        self.d = d
        self.n_iteration = n_iteration
        self.warm_up = warm_up
        self.er_p = er_p
        self.patience = patience
        self.dist_type = dist_type
        self.edge_delta = edge_delta
        self.min_gic_threshold = min_gic_threshold
        self.verbose = verbose
        
        self.fitted_graph = None
        self.metadata = {}

    def fit(self, original_graph):
        """
        Fits the Logit Graph model to the provided graph.

        Args:
            original_graph (nx.Graph): The original graph to fit.

        Returns:
            self: The instance with fitted_graph and metadata attributes populated.
        """
        if self.verbose:
            print(f"\n{'='*20} Processing Graph {'='*20}")
            print(f"Original graph - Nodes: {original_graph.number_of_nodes()}, Edges: {original_graph.number_of_edges()}")

        self.metadata = {
            'original_nodes': original_graph.number_of_nodes(),
            'original_edges': original_graph.number_of_edges(),
            'fit_success': False,
            'error_message': None,
        }
        
        try:
            adj_matrix = nx.to_numpy_array(original_graph)
            
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

    def _generate_graph(self, real_graph_arr):
        """
        Internal method to estimate parameters and generate the graph.
        """
        est = estimator.LogitRegEstimator(real_graph_arr, d=self.d)
        features, labels = est.get_features_labels()
        _, params, _ = est.estimate_parameters(l1_wt=1, alpha=0, features=features, labels=labels)
        sigma = params[0]

        n = real_graph_arr.shape[0]
        graph_model = graph.GraphModel(n=n, d=self.d, sigma=sigma)

        if self.verbose:
            print(f"Running LG generation for d={self.d}...")

        graphs, _, spectrum_diffs, best_iteration, best_graph_arr, gic_values = graph_model.populate_edges_spectrum_min_gic(
            max_iterations=self.n_iteration,
            patience=self.patience,
            real_graph=real_graph_arr,
            edge_delta=self.edge_delta,
            min_gic_threshold=self.min_gic_threshold,
            gic_dist_type=self.dist_type,
            verbose=self.verbose,
        )
        
        best_graph_nx = nx.from_numpy_array(best_graph_arr)
        gic_value = gic.GraphInformationCriterion(
            graph=nx.from_numpy_array(real_graph_arr),
            log_graph=best_graph_nx,
            model='LG',
            dist_type=self.dist_type
        ).calculate_gic()
        
        real_edges = np.sum(real_graph_arr) / 2
        edge_diffs = [abs(np.sum(g) / 2 - real_edges) for g in graphs]

        return best_graph_arr, sigma, gic_value, spectrum_diffs, edge_diffs, best_iteration, graphs, gic_values


class GraphModelComparator:
    """
    Compares Logit Graph with other random graph models, with a scikit-learn-like API.
    """
    def __init__(self, d_list, lg_params, other_model_n_runs=5, other_model_params=None, dist_type='KL', verbose=True):
        """
        Initializes the GraphModelComparator.

        Args:
            d_list (list): A list of `d` values to test for the Logit Graph model.
            lg_params (dict): Parameters for the Logit Graph fitting process (e.g., n_iteration, patience).
            other_model_n_runs (int): Number of runs for other random graph models.
            other_model_params (list, optional): Parameters for other models. Defaults to [].
            dist_type (str): Distance type for GIC calculation.
            verbose (bool): Whether to print progress information.
        """
        self.d_list = d_list
        self.lg_params = lg_params
        self.other_model_n_runs = other_model_n_runs
        if other_model_params is None:
            self.other_model_params = [
                {'lo': 0.01, 'hi': 0.25},  # ER: p
                {'k': {'lo': 2, 'hi': 10, 'step': 2}, 'p': {'lo': 0.01, 'hi': 0.5}},  # WS: k, p
                {'lo': 0.05, 'hi': 0.3},   # GRG: r
                {'lo': 1, 'hi': 8}         # BA: m
            ]
        else:
            self.other_model_params = other_model_params
        self.dist_type = dist_type
        self.verbose = verbose
        
        self.summary_df = None
        self.fitted_graphs_data = {}

    def compare(self, original_graph, graph_filepath):
        """
        Fits LG and other models to the graph and compares them.

        Args:
            original_graph (nx.Graph): The original graph to analyze.
            graph_filepath (str): The path to the original graph file for logging.

        Returns:
            self: The instance with summary_df and fitted_graphs_data attributes populated.
        """
        if self.verbose:
            print(f"\n{'='*30} Processing Graph: {os.path.basename(graph_filepath)} {'='*30}")
        
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

    def _fit_best_lg(self, adj_matrix):
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

    def _get_logit_graph_for_d(self, real_graph, d):
        """Estimates parameters and generates a graph for a specific `d`."""
        if isinstance(real_graph, nx.Graph):
            real_graph = nx.to_numpy_array(real_graph)

        est = estimator.LogitRegEstimator(real_graph, d=d)
        features, labels = est.get_features_labels()
        _, params, _ = est.estimate_parameters(l1_wt=1, alpha=0, features=features, labels=labels)
        sigma = params[0]

        n = real_graph.shape[0]
        graph_model = graph.GraphModel(n=n, d=d, sigma=sigma, er_p=self.lg_params['er_p'])

        if self.verbose:
            print(f"Running LG generation for d={d}...")
        
        lg_params = self.lg_params.copy()

        # Ensure that gic_dist_type and verbose are not passed twice.
        # The values from the comparator instance are given priority.
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
            dist_type=self.dist_type
        ).calculate_gic()

        real_edges = np.sum(real_graph) / 2
        edge_diffs = [abs(np.sum(g) / 2 - real_edges) for g in graphs]

        return best_graph_arr, sigma, gic_value, best_iteration, gic_values, spectrum_diffs, edge_diffs

    def _fit_other_models(self, original_graph):
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
        
        selector = ms.GraphModelSelection(
            graph=original_graph,
            log_graphs=[lg_graph_for_selection],
            log_params=[sigma],
            models=["ER", "WS", "GRG", "BA"],
            n_runs=self.other_model_n_runs,
            parameters=self.other_model_params
        )
        
        model_results = selector.select_model_avg_spectrum()

        for estimate in model_results['estimates']:
            model_name = estimate['model']
            if model_name != 'LG':
                param = clean_and_convert_param(estimate['param'])
                gic_value = estimate['GIC']
                
                func = selector.model_function(model_name=model_name)
                fitted_graph = func(original_graph.number_of_nodes(), param)
                
                self.fitted_graphs_data[model_name] = {
                    'graph': fitted_graph,
                    'metadata': {'fit_success': True, 'param': param, 'gic_value': gic_value}
                }
                if self.verbose:
                    print(f"{model_name} fitting - GIC: {gic_value:.4f}, Param: {param:.4f}")

    def _build_summary_df(self, graph_filepath):
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

