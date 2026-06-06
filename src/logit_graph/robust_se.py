"""Dyadic-cluster-robust standard error for the offset-logit sigma estimate.

A single observed graph gives one sigma_hat; its uncertainty cannot come from
re-subsampling that one graph (pseudo-replication). For the intercept-only offset
logistic regression ``logit(y_ij) = sigma + offset_ij`` the pairs share nodes and
are not independent, so the model-based (independence) SE ``1/sqrt(A)`` is wrong.

This module provides the dyadic-cluster-robust (sandwich) SE of Aronow, Samii &
Assenova (2015): with score residuals ``s_ij = y_ij - p_ij``, bread
``A = sum p_ij(1-p_ij)`` and node sums ``T_m = sum_{j!=m} s_mj``,
``B = sum_m T_m^2 - sum_ij s_ij^2`` and ``Var(sigma_hat) = B / A^2``. Distinct
dyads share at most one node, which collapses the O(n^4) double sum to O(n^2).
At d=0 (ER, independent dyads) it reduces to the independence SE.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.special import expit

from .lg_features import FeatureMode, build_pair_dataset
from .offset_logit import fit_offset_logit_fast


def dyadic_robust_sigma_se(
    offsets: np.ndarray,
    labels: np.ndarray,
    sigma_hat: float,
    n: int,
) -> tuple[float, float]:
    """Return ``(se_robust, se_naive)`` for the offset-logit intercept sigma_hat.

    ``offsets`` and ``labels`` are the per-pair feature offsets and 0/1 edge
    indicators for all upper-triangle pairs, enumerated in row-major order
    ``(0,1),(0,2),...,(1,2),...`` (the order produced by
    :func:`logit_graph.lg_features.build_pair_dataset`).
    """
    p = expit(sigma_hat + np.asarray(offsets, dtype=np.float64))
    s = np.asarray(labels, dtype=np.float64) - p
    A = float(np.sum(p * (1.0 - p)))
    if A <= 0:
        return float("nan"), float("nan")
    sum_s2 = float(np.sum(s * s))

    # T_m = sum of score residuals over dyads incident to node m, via row slices
    # (avoids materializing the n^2 (i,j) index arrays).
    T = np.zeros(n, dtype=np.float64)
    start = 0
    for i in range(n - 1):
        cnt = n - 1 - i
        s_row = s[start:start + cnt]
        T[i] += float(s_row.sum())
        T[i + 1:] += s_row
        start += cnt

    B = float(np.sum(T * T) - sum_s2)
    se_robust = math.sqrt(max(B, 0.0)) / A
    se_naive = 1.0 / math.sqrt(A)
    return se_robust, se_naive


def fit_sigma(
    adj: np.ndarray,
    d: int,
    feature_mode: FeatureMode = "incremental",
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Offset-logit MLE sigma_hat at radius ``d`` on all upper-triangle pairs.

    Returns ``(sigma_hat, log_likelihood, offsets, labels)``.
    """
    offsets, labels = build_pair_dataset(adj, d=d, mode=feature_mode, layer2=True)
    sigma, ll = fit_offset_logit_fast(offsets, labels)
    return float(sigma), float(ll), offsets, labels


def fit_sigma_with_robust_se(
    adj: np.ndarray,
    d: int,
    feature_mode: FeatureMode = "incremental",
) -> tuple[float, float, float]:
    """Fit sigma_hat at ``d`` and return ``(sigma_hat, se_robust, se_naive)``."""
    sigma, _ll, offsets, labels = fit_sigma(adj, d, feature_mode=feature_mode)
    se_r, se_n = dyadic_robust_sigma_se(offsets, labels, sigma, adj.shape[0])
    return sigma, se_r, se_n


def select_d_aic(
    adj: np.ndarray,
    d_candidates: list[int],
    feature_mode: FeatureMode = "incremental",
) -> tuple[int, float, np.ndarray, np.ndarray, dict[int, float]]:
    """AIC d-selection on the full graph (AIC = -2*ll + 2, one parameter sigma).

    Returns ``(d_hat, sigma_hat, offsets, labels, aic_by_d)`` where the offsets /
    labels / sigma_hat correspond to the selected ``d_hat`` (so the caller can
    feed them straight to :func:`dyadic_robust_sigma_se` without recomputing).
    """
    best: Optional[tuple[float, int, float, np.ndarray, np.ndarray]] = None
    aic_by_d: dict[int, float] = {}
    for d in d_candidates:
        sigma, ll, offsets, labels = fit_sigma(adj, d, feature_mode=feature_mode)
        aic = -2.0 * ll + 2.0
        aic_by_d[d] = aic
        if best is None or aic < best[0]:
            best = (aic, d, sigma, offsets, labels)
    assert best is not None
    _, d_hat, sigma_hat, offsets, labels = best
    return d_hat, sigma_hat, offsets, labels, aic_by_d
