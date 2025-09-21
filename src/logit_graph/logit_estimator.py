import numpy as np
from scipy.optimize import minimize
import torch
import warnings

from sklearn.linear_model import LogisticRegression
import networkx as nx

import statsmodels.api as sm
import statsmodels.formula.api as smf

from .degrees_counts import degree_vertex, get_sum_degrees

max_val = np.nan
eps = 1e-5

class NegativeLogLikelihoodLoss(torch.nn.Module):
    def __init__(self, graph, d):
        super(NegativeLogLikelihoodLoss, self).__init__()
        self.graph = torch.tensor(graph, dtype=torch.float32)  # Ensure graph is a PyTorch tensor
        self.n = graph.shape[0]
        self.d = d

    def logistic_probability(self, sum_degrees):
        num = torch.exp(sum_degrees)
        denom = 1 + 1 * torch.exp(sum_degrees)
        return num / denom

    def forward(self, params):
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
    def __init__(self, graph, d):
        self.graph = graph  # The observed adjacency matrix
        self.n = graph.shape[0]  # Number of nodes in the graph
        self.params_history = []  # History of parameters during optimization
        self.p = p

    def logistic_probability(self, sum_degrees):
        num = torch.exp(sum_degrees)
        denom = 1 + 1 * torch.exp(sum_degrees)
        return num / denom

    def likelihood_function(self, params):
        """Negative log-likelihood function to be minimized."""
        alpha, beta, sigma = params
        likelihood = 0

        for i in range(self.n):
            for j in range(i, self.n):
                #sum_degrees = np.sum(self.graph[i]) + np.sum(self.graph[j])  # Sum of degrees of nodes i and j
                #p_ij = self.logistic_probability(c, beta, sum_degrees)  # Probability of edge (i, j)
                degrees_i = get_sum_degrees(self.graph, i, self.p)
                degrees_j = get_sum_degrees(self.graph, j, self.p)
                sum_degrees_raw = ( alpha * degrees_i + beta * degrees_j )
                sum_degrees = ( sum_degrees_raw + sigma )
                p_ij = self.logistic_probability(sum_degrees)

                if 1-p_ij+eps <= 0:
                    return max_val

                if self.graph[i, j] == 1:
                    try:
                        likelihood += np.log(abs(p_ij + eps))  # Adding a small constant to avoid log(0)
                    except:
                        return np.float(max_val)
                else:
                    try:
                        likelihood += np.log(abs(1 - p_ij + eps))
                    except:
                        return np.float(max_val)

        return likelihood  # Negative because we minimize in the optimization routine

    def estimate_parameters(self, initial_guess=[0.5, 0.1, 0.1], learning_rate=0.01, max_iter=1000):
            alpha, beta, sigma = [torch.tensor(x, dtype=torch.float32, requires_grad=True) for x in initial_guess]
            optimizer = torch.optim.SGD([alpha, beta, sigma], lr=learning_rate)  # Using SGD optimizer from PyTorch

            # Instantiate the loss function class with the graph
            loss_function = NegativeLogLikelihoodLoss(self.graph, self.p)
            for _ in range(max_iter):
                optimizer.zero_grad()  # Clear previous gradients
                # Compute the loss by passing the parameters to the loss function instance
                loss = loss_function([alpha, beta, sigma])
                # Call backward on the loss tensor to compute gradients
                loss.backward()
                # Update parameters based on gradients
                optimizer.step()

                # Store parameters
                self.params_history.append([alpha.item(), beta.item(), sigma.item()])
                print(f"Current parameters: alpha={alpha.item()}, beta={beta.item()}, sigma={sigma.item()}, Loss={loss.item()}")

            print(f"Optimization completed. Estimated parameters: alpha={alpha.item()}, beta={beta.item()}, sigma={sigma.item()}")
            return alpha.item(), beta.item(), sigma.item()



# Main estimator for the LG graph
class LogitRegEstimator:
    def __init__(self, graph, d, verbose=False):
        self.graph = graph  # The observed adjacency matrix
        if isinstance(graph, np.ndarray):
            self.n = graph.shape[0]  # Number of nodes in the graph
        elif isinstance(graph, nx.Graph):
            self.n = graph.number_of_nodes()
        else:
            raise ValueError("Unsupported graph type. Please provide a NumPy array or NetworkX graph.")
        self.d = d # number of degrees to search
        self.verbose = verbose

    def get_features_labels(self):
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

        #features = np.array([(G.degree(i) / normalization, G.degree(j) / normalization) for i, j in data])
        normalization = 1
        features = np.array([(sum_degrees[i] / normalization, sum_degrees[j] / normalization) for i, j in data])

        # Add a constant term for the intercept
        features = sm.add_constant(features)
        
        if self.verbose:
            print("Feature extraction complete")
            print(f"Feature matrix shape: {features.shape}")
            
        return features, labels

    def estimate_parameters(self, l1_wt=1, alpha=0, features=None, labels=None):
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
