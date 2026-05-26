"""Fast scalar MLE for offset-only logistic regression: logit(y) = sigma + h."""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _logit(p: float) -> float:
    p = min(max(p, 1e-15), 1.0 - 1e-15)
    return math.log(p / (1.0 - p))


@njit(cache=True)
def _expit(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@njit(cache=True)
def _log1pexp(x: float) -> float:
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


@njit(cache=True)
def _offsets_are_zero(offsets: np.ndarray, tol: float = 1e-15) -> bool:
    for i in range(offsets.shape[0]):
        if abs(offsets[i]) > tol:
            return False
    return True


@njit(cache=True)
def _loglik_grad_hess(
    sigma: float,
    offsets: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    ll = 0.0
    grad = 0.0
    hess = 0.0
    for i in range(labels.shape[0]):
        eta = sigma + offsets[i]
        if labels[i]:
            ll -= _log1pexp(-eta)
        else:
            ll -= _log1pexp(eta)
        p = _expit(eta)
        grad += labels[i] - p
        hess -= p * (1.0 - p)
    return ll, grad, hess


@njit(cache=True)
def fit_offset_logit_numba(
    offsets: np.ndarray,
    labels: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[float, float]:
    """Return (sigma_hat, log_likelihood) for logit(y) = sigma + offsets."""
    n = labels.shape[0]
    if n == 0:
        return 0.0, 0.0

    if _offsets_are_zero(offsets):
        y_sum = 0.0
        for i in range(n):
            y_sum += labels[i]
        p_hat = y_sum / n
        p_hat = min(max(p_hat, 1e-15), 1.0 - 1e-15)
        sigma = _logit(p_hat)
        ll, _, _ = _loglik_grad_hess(sigma, offsets, labels)
        return sigma, ll

    y_sum = 0.0
    for i in range(n):
        y_sum += labels[i]
    p_hat = y_sum / n
    p_hat = min(max(p_hat, 1e-15), 1.0 - 1e-15)
    sigma = _logit(p_hat)

    for _ in range(max_iter):
        ll, grad, hess = _loglik_grad_hess(sigma, offsets, labels)
        if abs(grad) < tol:
            break
        if abs(hess) < 1e-14:
            sigma += 0.1 if grad > 0.0 else -0.1
            continue
        step = grad / hess
        if step > 10.0:
            step = 10.0
        elif step < -10.0:
            step = -10.0
        sigma -= step
        if abs(step) < tol:
            break
    ll, _, _ = _loglik_grad_hess(sigma, offsets, labels)
    return sigma, ll


def fit_offset_logit_fast(
    offsets: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    """NumPy wrapper around :func:`fit_offset_logit_numba`."""
    off = np.asarray(offsets, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int8)
    return fit_offset_logit_numba(off, lab)


def aic_from_offset_fit(
    sigma_hat: float,
    log_likelihood: float,
    *,
    extra_penalty: float = 0.0,
    k: float = 1.0,
) -> dict[str, float]:
    aic = -2.0 * log_likelihood + 2.0 * k + extra_penalty
    return {
        "aic": aic,
        "ll": log_likelihood,
        "k": k,
        "sigma_hat": sigma_hat,
    }
