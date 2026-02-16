from __future__ import annotations

import numpy as np


def degree_vertex(graph: np.ndarray, vertex: int, d: int) -> list[float]:
    """Return degrees of the vertex and all nodes within distance d.

    Returns a list ``[deg(vertex)] + [deg(u) for u within distance 1..d]``.
    For d=0, returns only ``[deg(vertex)]``.

    Uses vectorised numpy operations internally for speed.
    """
    degrees = graph.sum(axis=1)  # cached row sums

    if d == 0:
        return [float(degrees[vertex])]

    # BFS collecting all nodes at distances 1 through d
    visited = {vertex}
    current_layer = np.nonzero(graph[vertex])[0]
    all_neighbors: list[int] = list(current_layer)
    visited.update(current_layer.tolist())

    for _ in range(int(d) - 1):
        next_layer: list[int] = []
        for v in current_layer:
            for nv in np.nonzero(graph[v])[0]:
                if nv not in visited:
                    next_layer.append(int(nv))
                    visited.add(int(nv))
        all_neighbors.extend(next_layer)
        current_layer = next_layer

    return [float(degrees[vertex])] + [float(degrees[n]) for n in all_neighbors]


def get_sum_degrees(graph: np.ndarray, vertex: int, d: int) -> float:
    """Sum of degrees of vertex and all neighbours within d hops."""
    degrees = graph.sum(axis=1)

    if d == 0:
        return float(degrees[vertex])

    # BFS with vectorised neighbour lookup
    visited = {vertex}
    current_layer = np.nonzero(graph[vertex])[0]
    all_neighbors: list[int] = list(current_layer)
    visited.update(current_layer.tolist())

    for _ in range(int(d) - 1):
        next_layer: list[int] = []
        for v in current_layer:
            for nv in np.nonzero(graph[v])[0]:
                if nv not in visited:
                    next_layer.append(int(nv))
                    visited.add(int(nv))
        all_neighbors.extend(next_layer)
        current_layer = next_layer

    if all_neighbors:
        return float(degrees[vertex] + degrees[np.array(all_neighbors, dtype=int)].sum())
    return float(degrees[vertex])
