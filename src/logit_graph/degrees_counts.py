
import numpy as np
import networkx as nx

def degree_vertex(graph, vertex, d):
    """Return degrees of the vertex and all nodes within distance d.

    Returns a list ``[deg(vertex)] + [deg(u) for u within distance 1..d]``.
    For d=0, returns only ``[deg(vertex)]``.
    """
    def get_neighbors(v):
        return [i for i, x in enumerate(graph[v]) if x == 1]

    def get_degree(v):
        return sum(graph[v])

    if d == 0:
        return [get_degree(vertex)]

    # BFS collecting all nodes at distances 1 through d
    visited = {vertex}
    current_layer = get_neighbors(vertex)
    all_neighbors = list(current_layer)
    visited.update(current_layer)

    for _ in range(int(d) - 1):
        next_layer = []
        for v in current_layer:
            for nv in get_neighbors(v):
                if nv not in visited:
                    next_layer.append(nv)
                    visited.add(nv)
        all_neighbors.extend(next_layer)
        current_layer = next_layer

    return [get_degree(vertex)] + [get_degree(n) for n in all_neighbors]

def get_sum_degrees(graph, vertex, d):
    return sum(degree_vertex(graph, vertex, d))
