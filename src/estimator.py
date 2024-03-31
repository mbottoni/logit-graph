import numpy as np
from scipy.optimize import minimize

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
        eps = 1e-5
        max_val = 1e10

        # Iterate over all possible edges
        for i in range(self.n):
            for j in range(i, self.n):
                sum_degrees = np.sum(self.graph[i]) + np.sum(self.graph[j])  # Sum of degrees of nodes i and j
                p_ij = self.logistic_probability(c, beta, sum_degrees)  # Probability of edge (i, j)

                if 1-p_ij+eps <= 0:
                    return max_val

                if self.graph[i, j] == 1:
                    try:
                        likelihood += np.log(p_ij + eps)  # Adding a small constant to avoid log(0)
                    except:
                        return max_val
                else:
                    try:
                        likelihood += np.log(1 - p_ij + eps)
                    except:
                        return max_val

        return likelihood  # Negative because we minimize in the optimization routine

    def estimate_parameters2(self, initial_guess=[0.5, 0.1]):
        # Define a callback function to print the current parameters and loss at each step
        def callback(params):
            c, beta = params
            loss = self.likelihood_function(params)
            print(f"Current parameters: c={c}, beta={beta}")
            print(f"Current loss: {loss}")
            print()

        # Define the constraint function: c should be less than beta
        def constraint(params):
            return params[1] - params[0]  # This should be greater than 0 to satisfy the constraint

        # Set up the constraint dictionary for scipy.optimize.minimize
        cons = ({'type': 'ineq', 'fun': constraint})

        # Run the optimization with the callback function and the constraint
        eps = 1e-5
        result = minimize(self.likelihood_function,
                        initial_guess,
                        method='SLSQP',  ## 'BFGS', 'Nelder-Mead', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP'
                        bounds=[(eps, 2), (eps, 2)],
                        constraints=cons,  # Include the constraint
                        callback=callback)

        estimated_c, estimated_beta = result.x  # Extract the estimated parameters
        print(f"Optimization completed. Estimated parameters: c={estimated_c}, beta={estimated_beta}")
        return estimated_c, estimated_beta, result



