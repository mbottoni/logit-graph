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
        result = minimize(self.loss_function, initial_guess, method='L-BFGS-B', bounds=[(0, None), (0, None)])

        estimated_c, estimated_beta = result.x # Extract the estimated parameters
        return estimated_c, estimated_beta



