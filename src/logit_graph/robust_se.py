"""Dyadic-cluster-robust (sandwich) standard error for the offset-logit sigma_hat
(Aronow, Samii & Assenova 2015): a single graph's pairs share nodes, so the independence
SE 1/sqrt(A) is wrong; Var = B/A^2 (O(n^2) double sum). At d=0 it reduces to independence."""
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
    ``offsets``/``labels`` are the per-pair offsets and 0/1 edge indicators for all
    upper-triangle pairs in row-major order (as build_pair_dataset produces)."""
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
    Returns ``(sigma_hat, log_likelihood, offsets, labels)``."""
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
    """AIC d-selection on the full graph (AIC = -2*ll + 2, one parameter sigma). Returns
    ``(d_hat, sigma_hat, offsets, labels, aic_by_d)`` with offsets/labels/sigma_hat for the
    selected ``d_hat`` (ready to feed to dyadic_robust_sigma_se without recomputing)."""
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
