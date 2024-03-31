import numpy as np
from scipy.optimize import minimize

class LogisticModelEstimator:
    def __init__(self, graph):
        self.graph = graph
        self.n = graph.shape[0]

    def edge_probability(self, c, beta, i, j):
        # Compute the sum of degrees for vertices i and j
        sum_degrees_i = np.sum(self.graph[i])
        sum_degrees_j = np.sum(self.graph[j])
        sum_degrees = sum_degrees_i + sum_degrees_j

        # Compute the logistic regression probability
        probability = c / (1 + beta * np.exp(sum_degrees))
        return probability

    def loss_function(self, params):
        c, beta = params
        loss = 0

        # Iterate over all pairs of vertices and compute the loss
        for i in range(self.n):
            for j in range(i + 1, self.n):
                p_ij = self.edge_probability(c, beta, i, j)
                loss += (self.graph[i, j] - p_ij) ** 2  # Squared error loss

        return loss

    def estimate_parameters(self, initial_guess=[0.5, 0.1]):
        # Define a callback function to print the current parameters and loss at each step
        def callback(params):
            c, beta = params
            loss = self.loss_function(params)
            print(f"Current parameters: c={c}, beta={beta}")
            print(f"Current loss: {loss}")
            print()

        # Run the optimization with the callback function
        result = minimize(self.loss_function,
                        initial_guess,
                        method='L-BFGS-B',
                        bounds=[(0, None), (0, None)],
                        callback=callback)

        estimated_c, estimated_beta = result.x # Extract the estimated parameters
        print(f"Optimization completed. Estimated parameters: c={estimated_c}, beta={estimated_beta}")
        return estimated_c, estimated_beta


class MLEGraphModelEstimator:
    def __init__(self, graph):
        self.graph = graph  # The observed adjacency matrix
        self.n = graph.shape[0]  # Number of nodes in the graph

    def logistic_probability(self, c, beta, sum_degrees):
        """Logistic regression probability function."""
        return c / (1 + beta * np.exp(sum_degrees))

    def likelihood_function(self, params):
        """Negative log-likelihood function to be minimized."""
        c, beta = params
        likelihood = 0

        # Iterate over all possible edges
        for i in range(self.n):
            for j in range(i, self.n):
                sum_degrees = np.sum(self.graph[i]) + np.sum(self.graph[j])  # Sum of degrees of nodes i and j
                p_ij = self.logistic_probability(c, beta, sum_degrees)  # Probability of edge (i, j)
                #p_ij = np.random.choice(np.arange(0, 2), p=[1 - p_ij, p_ij])

                # Add the log-likelihood for edge (i, j)
                if self.graph[i, j] == 1:
                    likelihood += np.log(p_ij + 1e-9)  # Adding a small constant to avoid log(0)
                else:
                    likelihood += np.log(1 - p_ij + 1e-9)

        return -likelihood  # Negative because we minimize in the optimization routine

    def estimate_parameters(self, initial_guess=[0.5, 0.1]):
        # Define a callback function to print the current parameters and loss at each step
        def callback(params):
            c, beta = params
            loss = self.likelihood_function(params)
            print(f"Current parameters: c={c}, beta={beta}")
            print(f"Current loss: {loss}")
            print()

        # Run the optimization with the callback function
        result = minimize(self.likelihood_function,
                        initial_guess,
                        method='L-BFGS-B',
                        bounds=[(0, None), (0, None)],
                        callback=callback)

        estimated_c, estimated_beta = result.x # Extract the estimated parameters
        print(f"Optimization completed. Estimated parameters: c={estimated_c}, beta={estimated_beta}")
        return estimated_c, estimated_beta


