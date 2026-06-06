"""Temporal / growth Logit-Graph: generation + estimation (degree-only).

The equilibrium Logit-Graph (graph.py / simulate_graph) samples a graph at
stationarity. There, a *free* coefficient on a node-degree feature is neither
recoverable by logistic regression nor non-degenerate (see FINDINGS). This module
adds the **growth** reformulation, where an edge forms at step t from the
*predetermined* previous snapshot:

    logit( P[edge_ij forms at t] ) = sigma + alpha * D_ij(t-1)

with D = degree feature ("bounded": log(1+S_i)+log(1+S_j)), read from the snapshot
at t-1. Because the predictors are predetermined, each formation is — conditional
on the snapshot — an independent Bernoulli, so the pooled "at-risk dyad" design is
an ordinary logistic regression whose MLE recovers (sigma, alpha) consistently,
with no degeneracy.

(The structural feature has been removed for now; only the degree slope is modeled.)

This is a separate, additive path: the equilibrium model is untouched. Features
reuse logit_graph.lg_features; the fit reuses statsmodels.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
import statsmodels.api as sm

from .lg_features import FeatureMode, build_pair_dataset
from .offset_logit import aic_from_offset_fit

# The feature carrying alpha (the degree slope).
DEGREE_MODE: FeatureMode = "bounded"


@dataclass
class GrowthResult:
    """Output of :func:`grow_graph`.

    adj        final adjacency (n x n, 0/1, symmetric).
    X          (m, 1) pooled design [D] over all at-risk dyads across steps.
    y          (m,) formation outcomes (1 = the at-risk non-edge formed that step).
    snapshots  list of adjacency snapshots G(0), ..., G(n_steps) (empty if not stored).
    params     the generative parameters used.
    """
    adj: np.ndarray
    X: np.ndarray
    y: np.ndarray
    snapshots: list
    params: dict


def _degree_feature(adj, d, degree_mode):
    """Degree feature (D) + current-edge labels over all pairs.

    Pairs are in row-major upper-triangle order, matching np.triu_indices(n, 1).
    """
    D, labels = build_pair_dataset(adj, d=d, mode=degree_mode, layer2=True)
    return np.asarray(D, dtype=np.float64), np.asarray(labels, dtype=np.int8)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def grow_graph(
    n: int,
    d: int,
    sigma: float,
    alpha: float,
    *,
    n_steps: int,
    degree_mode: FeatureMode = DEGREE_MODE,
    seed: int | None = None,
    p0: float = 0.02,
    record_design: bool = True,
    store_snapshots: bool = True,
    allow_removal: bool = False,
) -> GrowthResult:
    """Grow a temporal Logit-Graph from a sparse ER seed (degree-only model).

    At each step, the degree feature D is read from the current snapshot
    (predetermined), every at-risk non-edge (i,j) forms with
    p = expit(sigma + alpha*D), and the formations are applied afterwards (edges
    are only added). With ``record_design`` the per-step at-risk dyads (D, formed?)
    are pooled into ``GrowthResult.X/y`` for estimation.

    With ``allow_removal=True`` the step instead resamples **every** dyad from the
    lagged probability — y_ij(t) ~ Bernoulli(expit(sigma + alpha*D_ij(t-1))) — so
    existing edges can be removed, not only formed. The predictors stay predetermined
    (read from t-1), so the design (now over *all* dyads, outcome = the new state) is
    still an ordinary logistic regression and the MLE stays consistent. This turns
    the monotone growth into an ergodic Markov chain with a stationary distribution
    (no saturation), at the cost of possible ERGM-style bistability for strong alpha.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.triu_indices(n, k=1)

    adj = np.zeros((n, n), dtype=np.float64)
    seed_mask = rng.random(rows.shape[0]) < p0
    adj[rows[seed_mask], cols[seed_mask]] = 1.0
    adj[cols[seed_mask], rows[seed_mask]] = 1.0

    snapshots = [adj.copy()] if store_snapshots else []
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for _ in range(n_steps):
        D, labels = _degree_feature(adj, d, degree_mode)
        p = expit(sigma + alpha * D)
        draw = rng.random(p.shape[0]) < p
        if allow_removal:
            # Resample every dyad from the lagged probability: edges may form OR
            # dissolve. Design is over all dyads, outcome = the new state.
            if record_design:
                Xs.append(D.reshape(-1, 1))
                ys.append(draw.astype(np.int8))
            adj[:] = 0.0
            ki, kj = rows[draw], cols[draw]
            adj[ki, kj] = 1.0
            adj[kj, ki] = 1.0
        else:
            at_risk = labels == 0
            form = at_risk & draw
            if record_design:
                Xs.append(D[at_risk].reshape(-1, 1))
                ys.append(form[at_risk].astype(np.int8))
            fi, fj = rows[form], cols[form]
            adj[fi, fj] = 1.0
            adj[fj, fi] = 1.0
        if store_snapshots:
            snapshots.append(adj.copy())

    X = np.vstack(Xs) if Xs else np.empty((0, 1), dtype=np.float64)
    y = np.concatenate(ys) if ys else np.empty(0, dtype=np.int8)
    return GrowthResult(
        adj=adj, X=X, y=y, snapshots=snapshots,
        params=dict(n=n, d=d, sigma=sigma, alpha=alpha, n_steps=n_steps,
                    degree_mode=degree_mode, p0=p0, allow_removal=allow_removal),
    )


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def growth_design_from_snapshots(
    snapshots: list,
    d: int,
    *,
    degree_mode: FeatureMode = DEGREE_MODE,
    allow_removal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the at-risk logistic-regression design from observed snapshots.

    For each consecutive pair (G(t-1), G(t)): predictor D from G(t-1); the at-risk
    set is the non-edges of G(t-1); the outcome is whether each became an edge in
    G(t). Reproduces the design recorded by :func:`grow_graph`.

    With ``allow_removal=True`` the at-risk set is *all* dyads and the outcome is the
    full new state of each dyad in G(t) — matching ``grow_graph(allow_removal=True)``.
    """
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for t in range(len(snapshots) - 1):
        prev = np.asarray(snapshots[t])
        nxt = np.asarray(snapshots[t + 1])
        n = prev.shape[0]
        rows, cols = np.triu_indices(n, k=1)
        D, labels_prev = _degree_feature(prev, d, degree_mode)
        state = (nxt[rows, cols] > 0)
        if allow_removal:
            Xs.append(D.reshape(-1, 1))
            ys.append(state.astype(np.int8))
        else:
            at_risk = labels_prev == 0
            Xs.append(D[at_risk].reshape(-1, 1))
            ys.append(state[at_risk].astype(np.int8))
    X = np.vstack(Xs) if Xs else np.empty((0, 1), dtype=np.float64)
    y = np.concatenate(ys) if ys else np.empty(0, dtype=np.int8)
    return X, y


def fit_growth_params(X: np.ndarray, labels: np.ndarray) -> dict:
    """Fit logit(P[form]) = sigma + alpha*D by ordinary logistic regression.

    Returns sigma, alpha (and their SEs), the log-likelihood, AIC (k=2), and
    n_params. This is the exact MLE for the growth model (dyad-independent given the
    past), so the estimates are consistent.
    """
    y = np.asarray(labels, dtype=int)
    Xc = sm.add_constant(np.asarray(X, dtype=np.float64), has_constant="add")
    res = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("newton", "bfgs", "lbfgs"):
            try:
                r = sm.Logit(y, Xc).fit(method=method, disp=0, maxiter=200)
                if np.isfinite(r.llf):
                    res = r
                    break
            except Exception:
                continue
        if res is None:
            res = sm.Logit(y, Xc).fit_regularized(method="l1", alpha=1e-4, disp=0)

    params = np.asarray(res.params, dtype=np.float64)
    try:
        bse = np.asarray(res.bse, dtype=np.float64)
    except Exception:
        bse = np.full(2, np.nan)
    ll = float(res.llf)
    return {
        "sigma": float(params[0]),
        "alpha": float(params[1]),
        "se_sigma": float(bse[0]),
        "se_alpha": float(bse[1]),
        "ll": ll,
        "aic": float(aic_from_offset_fit(params[0], ll, k=2.0)["aic"]),
        "n_params": 2,
    }


def fit_growth_from_result(result: GrowthResult) -> dict:
    """Convenience: fit on the design recorded by :func:`grow_graph`."""
    return fit_growth_params(result.X, result.y)
