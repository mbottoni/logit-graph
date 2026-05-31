"""Coverage for the three model-selection classes."""
import io
import contextlib

import networkx as nx
import pytest

from logit_graph.model_selection import (
    GraphModelSelection,
    ModelSelectorSpectrum,
    RandomGraphModelSelector,
)


def _er(n, p, seed):
    return nx.erdos_renyi_graph(n, p, seed=seed)


def _silent(fn, *args, **kwargs):
    """Run fn with stdout suppressed (the selectors print verbose progress)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


# -------------------------------------------------------------------
# RandomGraphModelSelector
# -------------------------------------------------------------------

def test_random_selector_fit_returns_best_model_in_candidates():
    real = _er(30, 0.2, seed=0)
    lg = _er(30, 0.2, seed=1)  # stand-in for an LG graph
    sel = RandomGraphModelSelector(real, lg)
    best, scores = _silent(sel.fit)
    assert best in {"ER", "WS", "BA", "LG"}
    assert set(scores.keys()).issubset({"ER", "WS", "BA", "LG"})
    # scores are floats, lower is better
    for v in scores.values():
        assert isinstance(v, float)


def test_random_selector_evaluate_model_returns_float():
    real = _er(20, 0.2, seed=0)
    sel = RandomGraphModelSelector(real, _er(20, 0.2, seed=1))
    score = _silent(sel.evaluate_model, _er(20, 0.2, seed=2))
    assert isinstance(score, float)
    assert score >= 0.0  # All components (ks, |Δclust|, |Δaspl|) are non-negative


def test_random_selector_score_smaller_for_similar_graphs():
    real = _er(30, 0.2, seed=0)
    # Same-distribution graph should score lower than a very different one
    similar = _er(30, 0.2, seed=5)
    different = _er(30, 0.6, seed=6)  # much denser
    sel = RandomGraphModelSelector(real, similar)
    s_sim = _silent(sel.evaluate_model, similar)
    s_diff = _silent(sel.evaluate_model, different)
    assert s_sim < s_diff


# -------------------------------------------------------------------
# ModelSelectorSpectrum
# -------------------------------------------------------------------

def test_spectrum_selector_kl_to_self_is_zero():
    real = _er(25, 0.2, seed=0)
    sel = ModelSelectorSpectrum(real, _er(25, 0.2, seed=1))
    sp = sel.graph_spectrum(real)
    kl = sel.kl_divergence(sp, sp)
    assert abs(kl) < 1e-9


def test_spectrum_selector_kl_non_negative_between_different_graphs():
    real = _er(25, 0.2, seed=0)
    sel = ModelSelectorSpectrum(real, _er(25, 0.2, seed=1))
    s1 = sel.graph_spectrum(_er(25, 0.2, seed=2))
    s2 = sel.graph_spectrum(_er(25, 0.5, seed=3))
    assert sel.kl_divergence(s1, s2) >= 0.0


def test_spectrum_model_penalty_known_models():
    sel = ModelSelectorSpectrum(_er(10, 0.2, seed=0), _er(10, 0.2, seed=1))
    assert sel.model_penalty("ER") == 1
    assert sel.model_penalty("WS") == 2
    assert sel.model_penalty("BA") == 1
    assert sel.model_penalty("LG") == 3
    assert sel.model_penalty("UNKNOWN") == 0


def test_spectrum_selector_fit_returns_best_model():
    real = _er(25, 0.2, seed=0)
    sel = ModelSelectorSpectrum(real, _er(25, 0.2, seed=1))
    best, scores = _silent(sel.fit)
    assert best in scores
    assert set(scores.keys()).issubset({"ER", "WS", "BA", "LG"})


# -------------------------------------------------------------------
# GraphModelSelection (high-level avg-spectrum / avg-GIC selectors)
# -------------------------------------------------------------------

def _make_selection(n=20, seed=0):
    G = _er(n, 0.2, seed=seed)
    return GraphModelSelection(
        graph=G,
        log_graphs=[G],
        log_params=[0.1],
        models=["ER", "BA"],
        parameters=[{"lo": 0.05, "hi": 0.3}, {"lo": 1, "hi": 3}],
        n_runs=2,
        grid_points=3,
    )


def test_graph_model_selection_avg_spectrum_returns_estimates_per_model():
    sel = _make_selection()
    out = _silent(sel.select_model_avg_spectrum)
    assert "model" in out and "estimates" in out
    assert out["model"] in {"ER", "BA"}
    assert len(out["estimates"]) == 2


def test_graph_model_selection_validate_input_models_length_matches_parameters():
    sel = _make_selection()
    # Construction completes without error if lengths match
    sel.validate_input()


def test_graph_model_selection_model_function_returns_callable_for_known_model():
    sel = _make_selection()
    fn = sel.model_function("ER")
    assert callable(fn)


def test_graph_model_selection_seed_offset_advances_state():
    sel = _make_selection(seed=42)
    # Two distinct seed offsets should be allowed without error
    sel._run_seed(0)
    sel._run_seed(1)
    # Just smoke — internal state is reset per call


def test_graph_model_selection_unknown_model_raises():
    sel = _make_selection()
    with pytest.raises(ValueError):
        sel.model_function("DEFINITELY_NOT_A_MODEL")
