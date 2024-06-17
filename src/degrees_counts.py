
import numpy as np
import networkx as nx

def degree_vertex(graph, vertex, d):
    def get_neighbors(v):
        return [i for i, x in enumerate(graph[v]) if x == 1]

    def get_degree(v):
        return sum(graph[v])

    if d == 0:
        return [get_degree(vertex)]
    if d == 1:
        neighbors = get_neighbors(vertex)
        return [get_degree(vertex)] + [get_degree(neighbor) for neighbor in neighbors]

    visited, current_neighbors = set([vertex]), get_neighbors(vertex)
    for _ in range(int(d) - 1):
        next_neighbors = []
        for v in current_neighbors:
            next_neighbors.extend([nv for nv in get_neighbors(v) if nv not in visited])
            visited.add(v)
        current_neighbors = list(set(next_neighbors))

    normalization =  1
    return [get_degree(vertex)/normalization] + [get_degree(neighbor)/normalization for neighbor in current_neighbors]

def get_sum_degrees(graph, vertex, d):
    return sum(degree_vertex(graph, vertex, d))
