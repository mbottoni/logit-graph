from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import Any, Optional, Union

try:
    import torch
except Exception:  # Make torch optional
    torch = None
import warnings

from sklearn.linear_model import LogisticRegression
import networkx as nx

import statsmodels.api as sm
import statsmodels.formula.api as smf

from .degrees_counts import degree_vertex, get_sum_degrees

max_val = np.nan
eps = 1e-5

class NegativeLogLikelihoodLoss(torch.nn.Module if torch is not None else object):
    def __init__(self, graph: np.ndarray, d: int) -> None:
        super(NegativeLogLikelihoodLoss, self).__init__()
        if torch is None:
            raise ImportError("PyTorch is required for NegativeLogLikelihoodLoss. Install with `pip install torch`.\n"
                              "Alternatively, use LogitRegEstimator which does not require torch.")
        self.graph = torch.tensor(graph, dtype=torch.float32)  # Ensure graph is a PyTorch tensor
        self.n = graph.shape[0]
        self.d = d

    def logistic_probability(self, sum_degrees: Any) -> Any:
        num = torch.exp(sum_degrees)
        denom = 1 + 1 * torch.exp(sum_degrees)
        return num / denom

    def forward(self, params: list) -> Any:
        alpha, beta, sigma = params
        likelihood = 0.0

        # Iterate over all possible edges
        for i in range(self.n):
            for j in range(i, self.n):
                degrees_i = get_sum_degrees(self.graph, i, self.d)
                degrees_j = get_sum_degrees(self.graph, j, self.d)
                sum_degrees_raw = ( alpha * degrees_i + beta * degrees_j )
                sum_degrees = ( sum_degrees_raw + sigma )

                p_ij = self.logistic_probability(sum_degrees)

                if self.graph[i, j] == 1:
                    likelihood += torch.log(p_ij + eps)  # Adding a small constant to avoid log(0)
                else:
                    likelihood += torch.log(1 - p_ij + eps)


        return -likelihood  # Return negative likelihood

class MLEGraphModelEstimator:
    """Maximum-likelihood estimator for the logit-graph model (requires PyTorch)."""

    def __init__(self, graph: np.ndarray, d: int) -> None:
        if torch is None:
            raise ImportError(
                "PyTorch is required for MLEGraphModelEstimator. "
                "Install with `pip install torch`.\n"
                "Alternatively, use LogitRegEstimator which does not require torch.")
        self.graph = graph  # The observed adjacency matrix
        self.n = graph.shape[0]  # Number of nodes in the graph
        self.d = d  # Neighbourhood depth for degree features
        self.params_history = []  # History of parameters during optimization

    def logistic_probability(self, sum_degrees: Any) -> Any:
        num = torch.exp(sum_degrees)
        denom = 1 + torch.exp(sum_degrees)
        return num / denom

    def likelihood_function(self, params: list) -> float:
        """Negative log-likelihood function to be minimized."""
        alpha, beta, sigma = params
        likelihood = 0

        for i in range(self.n):
            for j in range(i, self.n):
                degrees_i = get_sum_degrees(self.graph, i, self.d)
                degrees_j = get_sum_degrees(self.graph, j, self.d)
                sum_degrees_raw = alpha * degrees_i + beta * degrees_j
                sum_degrees = sum_degrees_raw + sigma
                p_ij = self.logistic_probability(sum_degrees)

                if 1 - p_ij + eps <= 0:
                    return float(max_val)

                if self.graph[i, j] == 1:
                    try:
                        likelihood += np.log(abs(p_ij + eps))
                    except Exception:
                        return float(max_val)
                else:
                    try:
                        likelihood += np.log(abs(1 - p_ij + eps))
                    except Exception:
                        return float(max_val)

        return likelihood

    def estimate_parameters(
        self,
        initial_guess: Optional[list[float]] = None,
        learning_rate: float = 0.01,
        max_iter: int = 1000,
    ) -> tuple[float, float, float]:
        if initial_guess is None:
            initial_guess = [0.5, 0.1, 0.1]
        alpha, beta, sigma = [
            torch.tensor(x, dtype=torch.float32, requires_grad=True)
            for x in initial_guess
        ]
        optimizer = torch.optim.SGD([alpha, beta, sigma], lr=learning_rate)

        loss_function = NegativeLogLikelihoodLoss(self.graph, self.d)
        for _ in range(max_iter):
            optimizer.zero_grad()
            loss = loss_function([alpha, beta, sigma])
            loss.backward()
            optimizer.step()

            self.params_history.append([alpha.item(), beta.item(), sigma.item()])

        return alpha.item(), beta.item(), sigma.item()



# Main estimator for the LG graph
class LogitRegEstimator:
    def __init__(
        self,
        graph: Union[np.ndarray, nx.Graph],
        d: int,
        verbose: bool = False,
    ) -> None:
        self.graph = graph  # The observed adjacency matrix
        if isinstance(graph, np.ndarray):
            self.n = graph.shape[0]  # Number of nodes in the graph
        elif isinstance(graph, nx.Graph):
            self.n = graph.number_of_nodes()
        else:
            raise ValueError("Unsupported graph type. Please provide a NumPy array or NetworkX graph.")
        self.d = d # number of degrees to search
        self.verbose = verbose

    def get_features_labels(self) -> tuple[np.ndarray, list[int]]:
        if self.verbose:
            print("Extracting features and labels...")
            
        G = nx.Graph(self.graph)

        edges = list(set(G.edges()))
        non_edges = list(set(nx.non_edges(G)))

        # Combine edges and non-edges to form the dataset
        data = edges + non_edges
        labels = [1] * len(edges) + [0] * len(non_edges)

        if self.verbose:
            print(f"Found {len(edges)} edges and {len(non_edges)} non-edges")
            print("Computing sum of degrees for each vertex...")

        # Pre compute
        sum_degrees = np.zeros(self.n)
        for i in range(self.n):
            sum_degrees[i] = get_sum_degrees(self.graph, vertex=i, d=self.d)

        # Symmetric feature: sum of degree features for both endpoints
        # This ensures P(edge i,j) = P(edge j,i) for undirected graphs
        features = np.array([sum_degrees[i] + sum_degrees[j] for i, j in data]).reshape(-1, 1)

        # Add a constant term for the intercept
        features = sm.add_constant(features)
        
        if self.verbose:
            print("Feature extraction complete")
            print(f"Feature matrix shape: {features.shape}")
            
        return features, labels

    def estimate_parameters(
        self,
        l1_wt: float = 1,
        alpha: float = 0,
        features: Optional[np.ndarray] = None,
        labels: Optional[list[int]] = None,
    ) -> tuple[Any, np.ndarray, np.ndarray]:
        """
        Fits the logistic regression model using the extracted features and labels.

        Args:
        l1_wt (float): The L1 weight (0 for pure L2, 1 for pure L1).
        alpha (float): Regularization strength. Larger values specify stronger regularization.
        """
        if self.verbose:
            print("\nStarting parameter estimation...")
            print(f"Regularization parameters: l1_weight={l1_wt}, alpha={alpha}")

        # Suppress statsmodels warnings for overflow and divide by zero
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered in exp")
            warnings.filterwarnings("ignore", message="divide by zero encountered in log")
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")
            
            # Logistic Regression Model using statsmodels with regularization
            model = sm.Logit(labels, features)

            ######################################
            
            # Fit the model with regularization
            if l1_wt in [0, 1]:
                # Pure L1 or L2 regularization
                if self.verbose:
                    print(f"Using pure {'L1' if l1_wt == 1 else 'L2'} regularization")
                result = model.fit_regularized(method='l1' if l1_wt == 1 else None, alpha=alpha, disp=self.verbose)
            else:
                # Elastic Net (combination of L1 and L2)
                if self.verbose:
                    print("Using Elastic Net regularization")
                result = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=l1_wt, disp=self.verbose)

        ######################################

        # Extract model parameters
        params = result.params
        
        if self.verbose:
            # Handle both pandas Series and numpy array formats
            if hasattr(params, 'values'):
                print(f"Estimated parameters: {params.values}")
            else:
                print(f"Estimated parameters: {params}")
            print(f"Logistic regression converged: {result.mle_retvals.get('converged', 'N/A')}")

        return result, params, features
