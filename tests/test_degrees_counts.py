import numpy as np
import networkx as nx

from logit_graph.degrees_counts import degree_vertex, get_sum_degrees


def build_adj_from_edges(n, edges):
    g = np.zeros((n, n), dtype=int)
    for i, j in edges:
        g[i, j] = 1
        g[j, i] = 1
    return g


def test_degree_vertex_d0_and_d1_simple_triangle():
    # Triangle: each node degree 2
    adj = build_adj_from_edges(3, [(0, 1), (1, 2), (0, 2)])

    # d=0 returns only the degree of the vertex
    assert degree_vertex(adj, 0, d=0) == [2]
    assert degree_vertex(adj, 1, d=0) == [2]

    # d=1 returns degree of vertex and neighbors
    dv0_d1 = degree_vertex(adj, 0, d=1)
    assert sorted(dv0_d1) == [2, 2, 2]


def test_degree_vertex_multi_hop_no_duplicates():
    # Path 0-1-2-3
    adj = build_adj_from_edges(4, [(0, 1), (1, 2), (2, 3)])

    # degrees: node0=1, node1=2, node2=2, node3=1
    # d=2 from vertex 0: vertex 0 (deg 1), dist-1 node 1 (deg 2), dist-2 node 2 (deg 2)
    vals = degree_vertex(adj, 0, d=2)
    assert sorted(vals) == [1, 2, 2]


def test_get_sum_degrees_matches_sum_of_degree_vertex():
    adj = build_adj_from_edges(5, [(0, 1), (1, 2), (1, 3), (3, 4)])
    for v in range(5):
        for d in [0, 1, 2, 3]:
            dv = degree_vertex(adj, v, d)
            assert get_sum_degrees(adj, v, d) == sum(dv)


