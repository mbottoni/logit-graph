"""Edge-case coverage for degree-counting primitives."""
import numpy as np

from logit_graph.degrees_counts import degree_vertex, get_sum_degrees


def _adj(n, edges):
    g = np.zeros((n, n), dtype=int)
    for i, j in edges:
        g[i, j] = g[j, i] = 1
    return g


def test_degree_vertex_isolated_returns_zero_at_all_depths():
    # vertex 2 is isolated; the other edge is (0,1)
    adj = _adj(3, [(0, 1)])
    for d in range(5):
        assert degree_vertex(adj, 2, d=d) == [0.0]


def test_get_sum_degrees_isolated_vertex_is_zero():
    adj = _adj(3, [(0, 1)])
    for d in range(4):
        assert get_sum_degrees(adj, 2, d=d) == 0.0


def test_degree_vertex_complete_graph_all_degrees_equal():
    n = 5
    adj = (np.ones((n, n), dtype=int) - np.eye(n, dtype=int))
    for v in range(n):
        # d=0 returns [deg(v)]
        assert degree_vertex(adj, v, d=0) == [float(n - 1)]
        # d=1 returns deg(v) + degs of n-1 neighbors → list of length n
        d1 = degree_vertex(adj, v, d=1)
        assert len(d1) == n
        assert all(x == float(n - 1) for x in d1)


def test_get_sum_degrees_complete_graph_d1():
    n = 4
    adj = (np.ones((n, n), dtype=int) - np.eye(n, dtype=int))
    # Each node deg n-1; d=1 ball covers all n nodes → sum = n*(n-1)
    assert get_sum_degrees(adj, 0, d=1) == float(n * (n - 1))


def test_degree_vertex_path_d0_equals_individual_degree():
    # Path 0-1-2-3-4: degrees 1,2,2,2,1
    adj = _adj(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    expected = [1, 2, 2, 2, 1]
    for v, deg in enumerate(expected):
        assert degree_vertex(adj, v, d=0) == [float(deg)]


def test_degree_vertex_cycle_d1_and_d2():
    # 4-cycle: every degree 2
    adj = _adj(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    # d=1 from vertex 0: includes vertex 0 + 2 neighbors (1, 3)
    d1 = degree_vertex(adj, 0, d=1)
    assert sorted(d1) == [2.0, 2.0, 2.0]
    # d=2: reaches all 4 nodes (vertex 2 via either side)
    d2 = degree_vertex(adj, 0, d=2)
    assert sorted(d2) == [2.0, 2.0, 2.0, 2.0]


def test_degree_vertex_disconnected_components_stops_at_boundary():
    # Two disjoint triangles
    adj = _adj(6, [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)])
    # BFS from vertex 0 cannot reach the other triangle even at large d
    result = degree_vertex(adj, 0, d=10)
    assert sorted(result) == [2.0, 2.0, 2.0]


def test_get_sum_degrees_matches_sum_of_degree_vertex_random_graph():
    rng = np.random.default_rng(42)
    n = 12
    upper = rng.random((n, n)) < 0.25
    upper = np.triu(upper, k=1)
    adj = (upper | upper.T).astype(int)
    for v in range(n):
        for d in range(4):
            assert get_sum_degrees(adj, v, d) == float(sum(degree_vertex(adj, v, d)))
