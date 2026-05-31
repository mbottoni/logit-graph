"""Edge cases + regression coverage for the offset-logit MLE."""
import math

import numpy as np
import statsmodels.api as sm

from logit_graph.offset_logit import (
    aic_from_offset_fit,
    fit_offset_logit_fast,
    fit_offset_logit_numba,
)


def _make_offsets_labels(n, sigma_true, offset_mean, offset_std, seed=0):
    """Synthetic (offsets, labels) for logit(p) = sigma + offset."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(offset_mean, offset_std, size=n)
    logits = sigma_true + offsets
    p = 1.0 / (1.0 + np.exp(-logits))
    labels = (rng.random(n) < p).astype(np.int8)
    return offsets.astype(np.float64), labels


# -------------------------------------------------------------------
# Empty / degenerate inputs
# -------------------------------------------------------------------

def test_fit_empty_returns_zeros():
    off = np.zeros(0, dtype=np.float64)
    lab = np.zeros(0, dtype=np.int8)
    sigma, ll = fit_offset_logit_numba(off, lab)
    assert sigma == 0.0
    assert ll == 0.0


def test_fit_all_negatives_pushes_sigma_very_negative():
    n = 100
    off = np.zeros(n, dtype=np.float64)
    lab = np.zeros(n, dtype=np.int8)
    sigma, _ = fit_offset_logit_numba(off, lab)
    # All zeros → p_hat clamped to 1e-15 → sigma very negative
    assert sigma < -30.0


def test_fit_all_positives_pushes_sigma_very_positive():
    n = 100
    off = np.zeros(n, dtype=np.float64)
    lab = np.ones(n, dtype=np.int8)
    sigma, _ = fit_offset_logit_numba(off, lab)
    assert sigma > 30.0


def test_fit_zero_offsets_recovers_empirical_density_logit():
    # All offsets 0 → MLE collapses to sigma_hat = logit(empirical density)
    n = 200
    off = np.zeros(n, dtype=np.float64)
    lab = np.zeros(n, dtype=np.int8)
    lab[:50] = 1  # 25% positive
    sigma, _ = fit_offset_logit_numba(off, lab)
    assert math.isclose(sigma, math.log(0.25 / 0.75), abs_tol=1e-6)


# -------------------------------------------------------------------
# Regression: large mean offset used to cause Newton oscillation
# -------------------------------------------------------------------

def _score_at(sigma, offsets, labels):
    """Score function: sum_i (y_i - p_i) where p_i = expit(sigma + offset_i)."""
    linpred = sigma + offsets
    # Use stable expit
    p = np.where(linpred >= 0,
                 1.0 / (1.0 + np.exp(-linpred)),
                 np.exp(linpred) / (1.0 + np.exp(linpred)))
    return float(np.sum(labels - p))


def test_fit_large_positive_mean_offset_converges_to_score_zero():
    """Pre-fix, sigma init = logit(p_hat) ≈ -3 with offsets ~ 4.5 caused
    Newton to overshoot, clamp to ±10, then oscillate without finding the
    MLE. The fix initializes sigma = logit(p_hat) - mean(offsets).

    We don't compare to statsmodels here because the regime (most p_i near 1)
    makes statsmodels' Hessian singular. Instead we verify the score
    sum_i(y_i - p_i) is near zero at the returned sigma.
    """
    off, lab = _make_offsets_labels(
        n=500, sigma_true=-3.0, offset_mean=4.5, offset_std=0.5, seed=42,
    )
    sigma, ll = fit_offset_logit_numba(off, lab)
    assert math.isfinite(sigma) and math.isfinite(ll)
    # MLE first-order condition for logistic: sum(y - p) = 0
    assert abs(_score_at(sigma, off, lab)) < 1e-4


def test_fit_large_negative_mean_offset_converges_to_score_zero():
    off, lab = _make_offsets_labels(
        n=500, sigma_true=-2.0, offset_mean=-3.5, offset_std=0.4, seed=7,
    )
    sigma, ll = fit_offset_logit_numba(off, lab)
    assert math.isfinite(sigma) and math.isfinite(ll)
    assert abs(_score_at(sigma, off, lab)) < 1e-4


def test_fit_moderate_offsets_matches_statsmodels():
    """Sanity check against statsmodels for a well-conditioned case."""
    off, lab = _make_offsets_labels(
        n=300, sigma_true=-1.5, offset_mean=0.5, offset_std=1.0, seed=3,
    )
    sigma, _ = fit_offset_logit_numba(off, lab)
    result = sm.Logit(lab, np.ones((len(lab), 1)), offset=off).fit(disp=False)
    sm_sigma = float(result.params[0])
    assert abs(sigma - sm_sigma) < 0.05


# -------------------------------------------------------------------
# Wrappers and AIC helpers
# -------------------------------------------------------------------

def test_fit_offset_logit_fast_accepts_lists_and_equals_numba():
    off, lab = _make_offsets_labels(
        n=200, sigma_true=-2.5, offset_mean=1.0, offset_std=1.0, seed=1,
    )
    s_fast, ll_fast = fit_offset_logit_fast(off.tolist(), lab.tolist())
    s_num, ll_num = fit_offset_logit_numba(off, lab)
    assert s_fast == s_num
    assert ll_fast == ll_num


def test_aic_from_offset_fit_formula_no_penalty():
    res = aic_from_offset_fit(sigma_hat=-3.0, log_likelihood=-100.0, k=1.0)
    assert res["aic"] == 200.0 + 2.0  # -2*ll + 2*k
    assert res["ll"] == -100.0
    assert res["k"] == 1.0
    assert res["sigma_hat"] == -3.0


def test_aic_extra_penalty_added():
    res_no = aic_from_offset_fit(-3.0, -100.0, extra_penalty=0.0)
    res_yes = aic_from_offset_fit(-3.0, -100.0, extra_penalty=4.5)
    assert math.isclose(res_yes["aic"] - res_no["aic"], 4.5)


def test_aic_different_k_changes_penalty_term():
    res_k1 = aic_from_offset_fit(-3.0, -50.0, k=1.0)
    res_k3 = aic_from_offset_fit(-3.0, -50.0, k=3.0)
    # difference is 2*(3 - 1) = 4
    assert math.isclose(res_k3["aic"] - res_k1["aic"], 4.0)
