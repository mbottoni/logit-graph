"""Deeper coverage for GraphModelSelection's averaging / dispatch internals.

The three selector classes' basic surface is covered in
``test_model_selection_core.py``. This file targets the untested paths:
``_generate_graph`` dispatch, ``_run_seed`` semantics, ``calculate_average_*``,
the avg-GIC selector, the SBM branch, ``validate_input`` failures, and
determinism under a fixed ``random_state``.
"""
import io
import contextlib

import networkx as nx
import numpy as np
import pytest

from logit_graph.model_selection import GraphModelSelection


def _er(n, p, seed):
    return nx.erdos_renyi_graph(n, p, seed=seed)


def _silent(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def _selection(models, parameters=None, n=24, seed=0, random_state=None, n_runs=2):
    G = _er(n, 0.2, seed=seed)
    return GraphModelSelection(
        graph=G,
        log_graphs=[G],
        log_params=[0.1],
        models=models,
        parameters=parameters,
        n_runs=n_runs,
        grid_points=3,
        random_state=random_state,
    )


# -------------------------------------------------------------------
# _run_seed
# -------------------------------------------------------------------

def test_run_seed_returns_none_when_random_state_unset():
    sel = _selection(["ER"], random_state=None)
    assert sel._run_seed(0) is None
    assert sel._run_seed(5) is None


def test_run_seed_is_offset_from_random_state():
    sel = _selection(["ER"], random_state=7)
    assert sel._run_seed(0) == 7 + 10_000
    assert sel._run_seed(3) == 7 + 10_000 + 3
    assert sel._run_seed(0) != sel._run_seed(1)


# -------------------------------------------------------------------
# _generate_graph dispatch
# -------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,params",
    [("ER", 0.2), ("BA", 2), ("WS", 0.3), ("KR", 1), ("GRG", 0.4), ("SBM", None)],
)
def test_generate_graph_returns_graph_with_n_nodes(model, params):
    sel = _selection([model])
    g = sel._generate_graph(model, params, seed_offset=0)
    assert isinstance(g, nx.Graph)
    assert g.number_of_nodes() == sel.graph.number_of_nodes()


def test_generate_graph_unknown_model_raises():
    sel = _selection(["ER"])
    with pytest.raises(ValueError):
        sel._generate_graph("NOPE", 0.2, seed_offset=0)


def test_generate_graph_ws_multiparam_uses_k_and_p():
    sel = _selection(["WS"])
    g = sel._generate_graph("WS", [4, 0.3], seed_offset=0)
    assert g.number_of_nodes() == sel.graph.number_of_nodes()
    # k=4 ⇒ each node wired to 4 neighbors before rewiring ⇒ n*k/2 edges.
    assert g.number_of_edges() == sel.graph.number_of_nodes() * 4 // 2


# -------------------------------------------------------------------
# model_function dispatch
# -------------------------------------------------------------------

def test_model_function_ws_wrapper_is_callable():
    sel = _selection(["WS"])
    fn = sel.model_function("WS")
    assert callable(fn)
    g = fn(sel.graph.number_of_nodes(), [4, 0.3])
    assert g.number_of_nodes() == sel.graph.number_of_nodes()


def test_model_function_sbm_wrapper_is_callable():
    sel = _selection(["SBM"])
    fn = sel.model_function("SBM")
    assert callable(fn)
    g = fn(sel.graph.number_of_nodes(), None)
    assert g.number_of_nodes() == sel.graph.number_of_nodes()


def test_model_function_lg_returns_none():
    sel = _selection(["LG"])
    # LG is handled specially upstream; model_function falls through to None.
    assert sel.model_function("LG") is None


# -------------------------------------------------------------------
# calculate_average_spectrum / calculate_average_gic
# -------------------------------------------------------------------

def test_average_spectrum_er_returns_50_bin_density():
    sel = _selection(["ER"])
    sp = _silent(sel.calculate_average_spectrum, "ER", 0.2)
    assert sp.shape == (50,)
    assert np.all(np.isfinite(sp))


def test_average_spectrum_lg_uses_log_graphs():
    sel = _selection(["LG"])
    sp = _silent(sel.calculate_average_spectrum, "LG", None)
    assert sp.shape == (50,)


def test_average_gic_er_returns_finite_float():
    sel = _selection(["ER"])
    val = _silent(sel.calculate_average_gic, "ER", 0.2)
    assert np.isfinite(val)


# -------------------------------------------------------------------
# select_model_avg_gic
# -------------------------------------------------------------------

def test_select_model_avg_gic_lg_and_sbm():
    # Both LG and SBM skip the param-estimator branch.
    sel = _selection(["LG", "SBM"])
    out = _silent(sel.select_model_avg_gic)
    assert out["model"] in {"LG", "SBM"}
    assert len(out["estimates"]) == 2
    assert set(out["estimates"]["model"]).issubset({"LG", "SBM"})


# -------------------------------------------------------------------
# select_model_avg_spectrum — SBM branch and single-model ensemble
# -------------------------------------------------------------------

def test_select_avg_spectrum_includes_sbm_branch():
    sel = _selection(["ER", "SBM"], parameters=[{"lo": 0.05, "hi": 0.3}, None])
    out = _silent(sel.select_model_avg_spectrum)
    assert set(out["estimates"]["model"]).issubset({"ER", "SBM"})
    assert len(out["estimates"]) == 2


def test_select_avg_spectrum_single_model_ensemble():
    sel = _selection(["ER"], parameters=[{"lo": 0.1, "hi": 0.3}])
    out = _silent(sel.select_model_avg_spectrum)
    assert out["model"] == "ER"
    assert len(out["estimates"]) == 1


# -------------------------------------------------------------------
# Determinism under a fixed random_state
# -------------------------------------------------------------------

def test_select_avg_spectrum_deterministic_under_random_state():
    params = [{"lo": 0.1, "hi": 0.3}, {"lo": 1, "hi": 3}]
    a = _selection(["ER", "BA"], parameters=params, random_state=11)
    b = _selection(["ER", "BA"], parameters=params, random_state=11)
    out_a = _silent(a.select_model_avg_spectrum)
    out_b = _silent(b.select_model_avg_spectrum)
    np.testing.assert_array_equal(out_a["estimates"]["GIC"], out_b["estimates"]["GIC"])
    assert out_a["model"] == out_b["model"]


# -------------------------------------------------------------------
# validate_input failure modes
# -------------------------------------------------------------------

def test_validate_input_rejects_non_graph():
    with pytest.raises(ValueError):
        GraphModelSelection(graph="not a graph", log_graphs=[_er(10, 0.2, 0)], log_params=[0.1])


def test_validate_input_rejects_empty_log_graphs():
    with pytest.raises(ValueError):
        GraphModelSelection(graph=_er(10, 0.2, 0), log_graphs=[], log_params=[0.1])
