"""Temporal / growth Logit-Graph: generation + estimation (degree-only).

The equilibrium Logit-Graph (graph.py / simulate_graph) samples a graph at
stationarity. There, a *free* coefficient on a node-degree feature is neither
recoverable by logistic regression nor non-degenerate (see FINDINGS). This module
adds the **temporal** reformulation, where each dyad's state at step t is drawn from
the *predetermined* previous snapshot:

    logit( P[edge_ij at t] ) = sigma + alpha * D_ij(t-1)

with D = degree feature ("bounded": log(1+S_i)+log(1+S_j)), read from the snapshot
at t-1. By default (``allow_removal=True``) every dyad is resampled each step, so
edges form AND dissolve and the process is an ergodic Markov chain with a stationary
distribution at moderate density. Setting ``allow_removal=False`` recovers the
add-only growth variant (edges only formed, drifting to saturation). Either way the
predictors are predetermined, so conditional on the snapshot each draw is an
independent Bernoulli and the pooled dyad design is an ordinary logistic regression
whose MLE recovers (sigma, alpha) consistently, with no degeneracy.

(The structural feature has been removed for now; only the degree slope is modeled.)

This is a separate, additive path: the equilibrium model is untouched. Features
reuse logit_graph.lg_features; the fit reuses statsmodels.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import entropy
import statsmodels.api as sm

from .lg_features import FeatureMode, build_pair_dataset
from .offset_logit import aic_from_offset_fit

# The feature carrying alpha (the degree slope).
DEGREE_MODE: FeatureMode = "bounded"


def _adjacency_esd_kl(cur: np.ndarray, prev: np.ndarray, nbins: int = 50) -> float:
    """KL divergence between two graphs' adjacency eigenvalue spectral densities.

    Both eigenvalue spectra are histogrammed on common bins spanning their combined
    range; KL is computed with a 1e-10 floor (as in the convergence diagnostics).
    """
    ec = np.linalg.eigvalsh(cur)
    ep = np.linalg.eigvalsh(prev)
    lo = float(min(ec.min(), ep.min()))
    hi = float(max(ec.max(), ep.max()))
    if hi <= lo:
        return 0.0
    bins = np.linspace(lo, hi, nbins + 1)
    hc, _ = np.histogram(ec, bins=bins, density=True)
    hp, _ = np.histogram(ep, bins=bins, density=True)
    return float(entropy(hc + 1e-10, hp + 1e-10))


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
    allow_removal: bool = True,
    until_convergence: bool = False,
    esd_tol: float = 1e-2,
    patience: int = 3,
    esd_nbins: int = 50,
) -> GrowthResult:
    """Grow a temporal Logit-Graph from a sparse ER seed (degree-only model).

    By default (``allow_removal=True``) each step resamples **every** dyad from the
    lagged probability — y_ij(t) ~ Bernoulli(expit(sigma + alpha*D_ij(t-1))), D read
    from the current snapshot — so edges form AND dissolve. The design is over *all*
    dyads with the new state as outcome. Because predictors are predetermined, it is
    an ordinary logistic regression and the MLE stays consistent. This is an ergodic
    Markov chain with a stationary distribution (no saturation), at the cost of
    possible ERGM-style bistability for strong positive alpha.

    With ``allow_removal=False`` the step instead only forms at-risk non-edges with
    p = expit(sigma + alpha*D) (edges are never removed); the design is over the
    at-risk non-edges (outcome = formed?) and the graph grows monotonically toward
    saturation. With ``record_design`` the per-step dyads (D, outcome) are pooled into
    ``GrowthResult.X/y`` for estimation.

    Stopping. By default the chain runs for exactly ``n_steps``. With
    ``until_convergence=True`` it instead runs *until mixed* — ``n_steps`` becomes a
    safety cap and the loop stops early once the adjacency-ESD KL divergence between
    consecutive snapshots stays below ``esd_tol`` for ``patience`` consecutive steps
    (the spectrum has stopped changing). ``GrowthResult.params`` then carries
    ``n_steps_run``, ``converged``, and the per-step ``esd_kl_trace``. The criterion
    is reliable for moderately large graphs (n ≳ 300); for small graphs the ESD itself
    fluctuates between independent draws, so the noise floor is high and convergence
    may not trigger before the cap.
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

    esd_kl_trace: list[float] = []
    below = 0
    converged = False
    n_steps_run = n_steps

    for step in range(n_steps):
        prev_adj = adj.copy() if until_convergence else None
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

        if until_convergence:
            kl = _adjacency_esd_kl(adj, prev_adj, esd_nbins)
            esd_kl_trace.append(kl)
            below = below + 1 if kl < esd_tol else 0
            if below >= patience:
                converged = True
                n_steps_run = step + 1
                break

    X = np.vstack(Xs) if Xs else np.empty((0, 1), dtype=np.float64)
    y = np.concatenate(ys) if ys else np.empty(0, dtype=np.int8)
    params = dict(n=n, d=d, sigma=sigma, alpha=alpha, n_steps=n_steps,
                  degree_mode=degree_mode, p0=p0, allow_removal=allow_removal)
    if until_convergence:
        params.update(until_convergence=True, esd_tol=esd_tol, patience=patience,
                      n_steps_run=n_steps_run, converged=converged,
                      esd_kl_trace=esd_kl_trace)
    return GrowthResult(adj=adj, X=X, y=y, snapshots=snapshots, params=params)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def growth_design_from_snapshots(
    snapshots: list,
    d: int,
    *,
    degree_mode: FeatureMode = DEGREE_MODE,
    allow_removal: bool = True,
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
