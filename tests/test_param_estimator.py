"""Coverage for GraphParameterEstimator.

Note: the module has a known constructor-ordering bug (``get_search_interval``
is called before ``is_multi_param`` is assigned) which crashes whenever an
explicit ``interval=`` is passed for a single-parameter model. We test only
the code paths that do not trigger it; per CLAUDE.md §3, we don't refactor
unrelated bugs.
"""
import math

import networkx as nx
import pytest

from logit_graph.param_estimator import GraphParameterEstimator


def _er(n, p, seed):
    return nx.erdos_renyi_graph(n, p, seed=seed)


# -------------------------------------------------------------------
# get_model_function — pure dispatch
# -------------------------------------------------------------------

@pytest.mark.parametrize("model", ["ER", "GRG", "KR", "WS", "BA"])
def test_get_model_function_returns_callable(model):
    # Construct with WS (multi-param) avoids the interval-branch bug
    G = _er(15, 0.3, seed=0)
    est = GraphParameterEstimator(G, model="WS")  # default interval path
    fn = est.get_model_function(model)
    assert callable(fn)


def test_get_model_function_unknown_raises():
    G = _er(15, 0.3, seed=0)
    est = GraphParameterEstimator(G, model="WS")
    with pytest.raises(ValueError):
        est.get_model_function("DEFINITELY_NOT_A_MODEL")


# -------------------------------------------------------------------
# Construction (default interval branches that don't touch is_multi_param)
# -------------------------------------------------------------------

def test_construct_er_default_interval_succeeds():
    G = _er(15, 0.3, seed=0)
    est = GraphParameterEstimator(G, model="ER")
    assert est.n == 15
    assert est.search_interval is not None
    # Default ER interval is np.arange(0, 1, eps)
    assert len(est.search_interval) > 0


def test_construct_ws_default_interval_returns_dict():
    G = _er(15, 0.3, seed=1)
    est = GraphParameterEstimator(G, model="WS")
    assert est.is_multi_param is True
    assert isinstance(est.search_interval, dict)
    assert "k" in est.search_interval and "p" in est.search_interval


def test_construct_ba_default_interval():
    G = _er(15, 0.3, seed=2)
    est = GraphParameterEstimator(G, model="BA")
    assert est.is_multi_param is False
    assert len(est.search_interval) > 0


# -------------------------------------------------------------------
# is_multi_parameter_model
# -------------------------------------------------------------------

def test_is_multi_param_true_for_ws():
    G = _er(15, 0.3, seed=3)
    est = GraphParameterEstimator(G, model="WS")
    assert est.is_multi_parameter_model() is True


def test_is_multi_param_false_for_others():
    G = _er(15, 0.3, seed=4)
    for model in ["ER", "BA"]:
        est = GraphParameterEstimator(G, model=model)
        assert est.is_multi_parameter_model() is False


# -------------------------------------------------------------------
# calculate_gic returns finite numbers
# -------------------------------------------------------------------

def test_calculate_gic_returns_finite_value():
    G = _er(20, 0.2, seed=5)
    est = GraphParameterEstimator(G, model="ER")
    val = est.calculate_gic(0.2)
    assert math.isfinite(val)
    assert val >= 0.0  # GIC = 2*KL + 2*k, both non-negative


# -------------------------------------------------------------------
# end-to-end: estimate() returns a dict with param + gic
# -------------------------------------------------------------------

def test_estimate_er_returns_param_and_gic():
    G = _er(20, 0.2, seed=6)
    # Use coarse eps for a quick test
    est = GraphParameterEstimator(G, model="ER", eps=0.2, search="grid")
    out = est.estimate()
    assert "param" in out and "gic" in out
    assert math.isfinite(out["gic"])
