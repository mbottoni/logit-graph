import numpy as np
import networkx as nx

from logit_graph.gic import GraphInformationCriterion


def test_compute_spectral_density_output_shapes():
    G = nx.erdos_renyi_graph(20, 0.2, seed=1)
    gic = GraphInformationCriterion(G, model='ER', p=0.2)
    hist, bins = gic.compute_spectral_density(G)
    assert hist.ndim == 1 and bins.ndim == 1
    assert len(bins) == len(hist) + 1
    assert np.isfinite(hist).all()


def test_calculate_gic_identical_graphs_near_zero():
    G = nx.erdos_renyi_graph(25, 0.3, seed=2)
    gic = GraphInformationCriterion(G, model='LG', log_graph=G, dist='L2')
    # When comparing identical graphs, their spectral densities should match closely
    d = gic.calculate_gic()
    assert d >= 0
    assert d < 1e-6 or np.isclose(d, 0.0)


def test_generate_model_graph_by_string_models():
    n = 30
    for model, p in [('ER', 0.1), ('GRG', 0.2), ('WS', 0.2), ('BA', 2)]:
        G = nx.erdos_renyi_graph(n, 0.1, seed=0)
        gi = GraphInformationCriterion(G, model=model, p=p)
        MG = gi.generate_model_graph()
        assert isinstance(MG, nx.Graph)
        assert MG.number_of_nodes() == n


