"""Edge cases for the offset-logit MLE not covered in test_offset_logit_core.py.

Core file covers: empty input, all-neg/all-pos, zero-offset density recovery,
large mean-offset regression, statsmodels match, AIC formula. This file adds:
the ``max_iter`` knob, extreme offsets, degenerate labels with nonzero offsets,
the ``_offsets_are_zero`` detector, single observation, and combined AIC terms.
"""
import math

import numpy as np

from logit_graph.offset_logit import (
    _offsets_are_zero,
    aic_from_offset_fit,
    fit_offset_logit_numba,
)


def _score_at(sigma, offsets, labels):
    linpred = sigma + offsets
    p = np.where(linpred >= 0,
                 1.0 / (1.0 + np.exp(-linpred)),
                 np.exp(linpred) / (1.0 + np.exp(linpred)))
    return float(np.sum(labels - p))


# -------------------------------------------------------------------
# max_iter knob
# -------------------------------------------------------------------

def test_single_newton_step_is_finite():
    rng = np.random.default_rng(0)
    off = rng.normal(0.5, 1.0, size=200).astype(np.float64)
    lab = (rng.random(200) < 0.3).astype(np.int8)
    sigma, ll = fit_offset_logit_numba(off, lab, max_iter=1)
    assert math.isfinite(sigma) and math.isfinite(ll)


def test_more_iterations_reach_score_zero():
    rng = np.random.default_rng(1)
    off = rng.normal(0.0, 1.0, size=400).astype(np.float64)
    lab = (rng.random(400) < 0.4).astype(np.int8)
    sigma, _ = fit_offset_logit_numba(off, lab, max_iter=100)
    # Well-conditioned ⇒ first-order condition sum(y - p) ≈ 0 at the MLE.
    assert abs(_score_at(sigma, off, lab)) < 1e-4


# -------------------------------------------------------------------
# Extreme offsets stay finite
# -------------------------------------------------------------------

def test_extreme_positive_offsets_stay_finite():
    n = 300
    off = np.full(n, 100.0, dtype=np.float64)
    rng = np.random.default_rng(2)
    lab = (rng.random(n) < 0.5).astype(np.int8)
    sigma, ll = fit_offset_logit_numba(off, lab)
    assert math.isfinite(sigma) and math.isfinite(ll)


def test_extreme_negative_offsets_stay_finite():
    n = 300
    off = np.full(n, -100.0, dtype=np.float64)
    rng = np.random.default_rng(3)
    lab = (rng.random(n) < 0.5).astype(np.int8)
    sigma, ll = fit_offset_logit_numba(off, lab)
    assert math.isfinite(sigma) and math.isfinite(ll)


# -------------------------------------------------------------------
# Degenerate labels with nonzero offsets (Hessian → 0 path)
# -------------------------------------------------------------------

def test_all_ones_labels_with_nonzero_offsets_finite():
    rng = np.random.default_rng(4)
    off = rng.normal(2.0, 0.5, size=150).astype(np.float64)
    lab = np.ones(150, dtype=np.int8)
    sigma, ll = fit_offset_logit_numba(off, lab)
    assert math.isfinite(sigma) and math.isfinite(ll)


def test_all_zeros_labels_with_nonzero_offsets_finite():
    rng = np.random.default_rng(5)
    off = rng.normal(-2.0, 0.5, size=150).astype(np.float64)
    lab = np.zeros(150, dtype=np.int8)
    sigma, ll = fit_offset_logit_numba(off, lab)
    assert math.isfinite(sigma) and math.isfinite(ll)


# -------------------------------------------------------------------
# Single observation
# -------------------------------------------------------------------

def test_single_observation_is_finite():
    off = np.array([0.5], dtype=np.float64)
    lab = np.array([1], dtype=np.int8)
    sigma, ll = fit_offset_logit_numba(off, lab)
    assert math.isfinite(sigma) and math.isfinite(ll)


# -------------------------------------------------------------------
# _offsets_are_zero detector
# -------------------------------------------------------------------

def test_offsets_are_zero_exact():
    assert _offsets_are_zero(np.zeros(10, dtype=np.float64))


def test_offsets_are_zero_within_tolerance():
    # Below the 1e-15 default tolerance ⇒ treated as zero.
    assert _offsets_are_zero(np.full(10, 1e-16, dtype=np.float64))


def test_offsets_not_zero_above_tolerance():
    off = np.zeros(10, dtype=np.float64)
    off[3] = 1e-10  # above 1e-15 ⇒ not zero
    assert not _offsets_are_zero(off)


# -------------------------------------------------------------------
# AIC with combined k and extra_penalty
# -------------------------------------------------------------------

def test_aic_combines_k_and_extra_penalty():
    res = aic_from_offset_fit(-2.0, -80.0, extra_penalty=3.0, k=2.0)
    # -2*ll + 2*k + extra_penalty
    assert math.isclose(res["aic"], 160.0 + 4.0 + 3.0)
    assert res["k"] == 2.0
