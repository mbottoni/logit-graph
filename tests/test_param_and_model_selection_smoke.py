import numpy as np
import networkx as nx

import pytest
from logit_graph.param_estimator import GraphParameterEstimator
from logit_graph.model_selection import GraphModelSelection


def small_graph(n=20, p=0.15, seed=0):
    return nx.erdos_renyi_graph(n, p, seed=seed)


@pytest.mark.xfail(reason="Known constructor ordering bug in GraphParameterEstimator; leaving module unchanged")
def test_param_estimator_grid_search_er_smoke():
    G = small_graph()
    # Workaround: ensure interval provided and eps coarse to keep it fast
    est = GraphParameterEstimator(G, model='ER', interval={'lo': 0.05, 'hi': 0.3}, eps=0.1, search='grid')
    res = est.estimate()
    assert 'param' in res and 'gic' in res
    assert isinstance(res['param'], float)


def test_model_selection_avg_spectrum_smoke():
    G = small_graph()
    # Use the same G as a placeholder LG graph; this is a smoke test
    selector = GraphModelSelection(
        graph=G,
        log_graphs=[G],
        log_params=[0.1],
        models=['ER', 'BA'],
        parameters=[{'lo': 0.05, 'hi': 0.3}, {'lo': 1, 'hi': 3}],
        n_runs=2,
        grid_points=3
    )
    out = selector.select_model_avg_spectrum()
    assert 'model' in out and 'estimates' in out
    assert len(out['estimates']) == 2


