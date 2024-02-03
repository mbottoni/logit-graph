import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import imageio
np.random.seed(42)

def calculate_spectrum(graph):
    """Calculate the eigenvalues of the adjacency matrix of the graph."""
    #adjacency_matrix = nx.adjacency_matrix(graph).todense()
    G = nx.from_numpy_array(graph)
    eigenvalues = nx.laplacian_spectrum(G)
    #eigenvalues = np.linalg.eigvals(graph)
    return np.sort(eigenvalues)

def generate_random_graph(n, p):
    """Generate a random graph represented by an adjacency matrix.

    Parameters:
    - n: Number of vertices
    - p: Probability of an edge between any two vertices

    Returns:
    - adj_matrix: Adjacency matrix representing the graph
    """

    adj_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):  # This ensures an undirected graph
            if np.random.rand() < p:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Symmetric entry for undirected graph

    return adj_matrix

# Define the graph as an adjcency matrix where n is the order of the graph
# Convention: Lines are vertex and columns are where the vertex is connected
def initialize_graph( n ):
    return np.zeros( ( n , n ) )

# Simple logistic regression function c / (1+beta*np.exp(sum_degrees))
def logistic_regression(c, beta, sum_degrees):
    num     = c
    denom   = 1 + beta * np.exp(-sum_degrees)
    return num / denom

# Returns a vector of degree of vertex i and a degree of each neighboor
# with a distance of p from the vertex i
def degree_vertex(adj_matrix, vertex, p):
    """Returns the degree of the vertex and degrees of neighbors within a distance of p."""
    n = len(adj_matrix)

    # Function to get neighbors of a given vertex
    def get_neighbors(v):
        return [i for i, x in enumerate(adj_matrix[v]) if x == 1]

    # Function to get degree of a given vertex
    def get_degree(v):
        return sum(adj_matrix[v])
        #return sum(adj_matrix[v]) / (n*(n-1)/2)

    # Base case for p=1 and p=0
    if p == 0:
        neighbors = get_neighbors(vertex)
        return [get_degree(vertex)]

    if p == 1:
        neighbors = get_neighbors(vertex)
        return [get_degree(vertex)] + [get_degree(neighbor) for neighbor in neighbors]

    # For p > 1
    visited = set([vertex])
    current_neighbors = get_neighbors(vertex)
    for _ in range(p-1):
        next_neighbors = []
        for v in current_neighbors:
            neighbors_of_v = get_neighbors(v)
            next_neighbors.extend([nv for nv in neighbors_of_v if nv not in visited])
            visited.add(v)
        current_neighbors = list(set(next_neighbors))

    return [get_degree(vertex)] + [get_degree(neighbor) for neighbor in current_neighbors]

def get_sum_degrees(graph, vertex, p=1):
    """Gets the sum of degrees for a vertex considering a distance p."""
    return sum(degree_vertex(graph, vertex, p))

def get_edge_logit(c, beta, sum_degrees, threshold):
    """Decides if an edge should be added based on the logistic regression output and a threshold."""
    val_log = logistic_regression(c, beta, sum_degrees)
    #return 1 if logistic_regression(c, beta, sum_degrees) >= threshold else 0
    return np.random.choice(np.arange(0, 2), p=[1-val_log, val_log])

def add_vertex(graph, c, beta, p, threshold, sigma=1):
    """Modified function to iterate over the graph's vertices and decide if an edge will be added."""
    n, m = graph.shape
    for i in range(n):
        for j in range(m):
            if i != j and graph[i,j] == 0:
                normalization = n * (n-1) /2
                sum_degrees_i = get_sum_degrees(graph, i, p) / (normalization)
                sum_degrees_j = get_sum_degrees(graph, j, p) / (normalization)
                sum_degrees = sum_degrees_i + sum_degrees_j + abs(np.random.normal(0, sigma))

                graph[i, j] = get_edge_logit(c, beta, sum_degrees, threshold)
                #print('Testing for:', i, j, logistic_regression(c, beta, sum_degrees), 'Value of edge', get_edge_logit(c, beta, sum_degrees, threshold))
    return graph

def check_convergence(graph_list, tolerance=1):
    """
    Checks if the graph has converged by comparing the current adjacency matrix
    with the one from the previous iteration.

    Args:
    - current_graph: The current adjacency matrix.
    - previous_graph: The adjacency matrix from the previous iteration.
    - tolerance: A small value. If the difference between the matrices (in terms of total number of edges)
                 is less than this value, the function returns True.

    Returns:
    - True if the graph has converged, False otherwise.
    """
    difference = 0
    for i in range(1,20,2):
        difference += np.sum(np.abs(graph_list[-i] - graph_list[-(i+1)]))

    #difference += np.sum(np.abs(graph_list[-1] - graph_list[-2]))
    #difference += np.sum(np.abs(graph_list[-3] - graph_list[-4]))
    #difference += np.sum(np.abs(graph_list[-5] - graph_list[-6]))
    #return False
    print('dif: ',difference)

    return difference <= tolerance
