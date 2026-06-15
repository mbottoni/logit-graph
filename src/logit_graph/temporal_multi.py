"""Multi-feature (unified) Temporal Logit-Graph: the degree-only temporal model extended with
fixed exogenous dyad covariates, logit P[edge_ij at t] = sigma + alpha*D(t-1) + sum_k beta_k*F_k.
Every F_k is predetermined, so the pooled dyad design is an ordinary logistic regression whose
MLE recovers (sigma, alpha, beta_1..beta_k) consistently — no degeneracy."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.special import expit
import statsmodels.api as sm

from .lg_features import FeatureMode, build_pair_dataset

# The feature carrying alpha (the degree slope), read from the previous snapshot.
DEGREE_MODE: FeatureMode = "bounded"


@dataclass
class MultiGrowthResult:
    """Output of :func:`grow_graph_multi`: final adjacency ``adj`` (n x n, 0/1, symmetric),
    pooled design ``X`` (m, 1+k) = [D, F_1..F_k] / outcomes ``y``, the ``feature_names`` for the
    extra covariates, the ``snapshots`` G(0..n_steps) (empty if not stored), and ``params``."""
    adj: np.ndarray
    X: np.ndarray
    y: np.ndarray
    feature_names: list
    snapshots: list
    params: dict


def _degree_feature(adj, d, degree_mode):
    """Degree feature (D) + current-edge labels over all upper-triangle pairs, in row-major
    order (matching np.triu_indices(n, 1))."""
    D, labels = build_pair_dataset(adj, d=d, mode=degree_mode, layer2=True)
    return np.asarray(D, dtype=np.float64), np.asarray(labels, dtype=np.int8)


# ---------------------------------------------------------------------------
# Exogenous dyad-feature builders (how to construct F_k on a real graph)
# ---------------------------------------------------------------------------

def _as_adj(graph):
    """Accept a numpy adjacency or a networkx graph, returning a dense 0/1 ndarray."""
    if isinstance(graph, np.ndarray):
        return graph
    import networkx as nx
    return nx.to_numpy_array(graph)


def community_feature(graph, *, resolution: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Same-community indicator over upper-triangle dyads from a fixed Louvain partition of the
    graph at the given resolution (higher resolution -> finer communities). An exogenous node
    covariate, like SBM uses, so it stays identifiable. Returns a (n*(n-1)/2,) float array."""
    import networkx as nx
    A = _as_adj(graph)
    n = A.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    part = nx.community.louvain_communities(nx.from_numpy_array(A), seed=seed, resolution=resolution)
    blk = np.empty(n, dtype=int)
    for i, com in enumerate(part):
        for v in com:
            blk[v] = i
    return (blk[rows] == blk[cols]).astype(np.float64)


def latent_feature(graph, k: int = 4, *, kind: str = "dot") -> np.ndarray:
    """Latent proximity over upper-triangle dyads from a rank-k adjacency spectral embedding z
    (eigvecs scaled by sqrt|eigval|). kind="dot": z_i . z_j (RDPG inner product); kind="dist":
    -||z_i - z_j|| (Hoff latent-space distance). Standardized; returns a (n*(n-1)/2,) array."""
    A = _as_adj(graph)
    n = A.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    w, U = np.linalg.eigh(A)
    idx = np.argsort(-np.abs(w))[:k]
    z = U[:, idx] * np.sqrt(np.abs(w[idx]))
    if kind == "dist":
        L = -np.sqrt(((z[rows] - z[cols]) ** 2).sum(1))
    else:
        L = (z[rows] * z[cols]).sum(1)
    return (L - L.mean()) / (L.std() + 1e-9)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def grow_graph_multi(
    n: int,
    d: int,
    sigma: float,
    alpha: float,
    features: np.ndarray,
    coefs,
    *,
    n_steps: int,
    feature_names: list | None = None,
    degree_mode: FeatureMode = DEGREE_MODE,
    allow_removal: bool = True,
    seed: int | None = None,
    p0: float = 0.02,
    record_design: bool = True,
    store_snapshots: bool = True,
) -> MultiGrowthResult:
    """Grow a multi-feature temporal Logit-Graph from a sparse ER seed. ``features`` is an
    (n*(n-1)/2, k) array of FIXED exogenous dyad covariates (upper-triangle order) with
    coefficients ``coefs``; each step draws dyads from expit(sigma + alpha*D + features @ coefs),
    D read from the previous snapshot. allow_removal (default) resamples every dyad (ergodic
    chain); allow_removal=False grows add-only. Returns the pooled design and outcomes."""
    features = np.atleast_2d(np.asarray(features, dtype=np.float64))
    if features.shape[0] != n * (n - 1) // 2:
        features = features.T
    coefs = np.asarray(coefs, dtype=np.float64).ravel()
    k = features.shape[1]
    if coefs.shape[0] != k:
        raise ValueError(f"coefs has length {coefs.shape[0]} but features has {k} columns")
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(k)]
    fixed_lo = features @ coefs

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
        p = expit(sigma + alpha * D + fixed_lo)
        draw = rng.random(p.shape[0]) < p
        design = np.column_stack([D, features])
        if allow_removal:
            if record_design:
                Xs.append(design)
                ys.append(draw.astype(np.int8))
            adj[:] = 0.0
            adj[rows[draw], cols[draw]] = 1.0
            adj[cols[draw], rows[draw]] = 1.0
        else:
            at_risk = labels == 0
            form = at_risk & draw
            if record_design:
                Xs.append(design[at_risk])
                ys.append(form[at_risk].astype(np.int8))
            adj[rows[form], cols[form]] = 1.0
            adj[cols[form], rows[form]] = 1.0
        if store_snapshots:
            snapshots.append(adj.copy())

    X = np.vstack(Xs) if Xs else np.empty((0, 1 + k), dtype=np.float64)
    y = np.concatenate(ys) if ys else np.empty(0, dtype=np.int8)
    params = dict(n=n, d=d, sigma=sigma, alpha=alpha, coefs=coefs.tolist(),
                  feature_names=list(feature_names), n_steps=n_steps,
                  degree_mode=degree_mode, allow_removal=allow_removal, p0=p0)
    return MultiGrowthResult(adj=adj, X=X, y=y, feature_names=list(feature_names),
                             snapshots=snapshots, params=params)


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def multi_design_from_snapshots(
    snapshots: list,
    d: int,
    features: np.ndarray,
    *,
    degree_mode: FeatureMode = DEGREE_MODE,
    allow_removal: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the pooled [D, F_1..F_k] design from observed snapshots: D from G(t-1), the extra
    fixed ``features`` repeated each step, outcome from G(t). With allow_removal the design is
    over all dyads (else the at-risk non-edges). Reproduces grow_graph_multi's recorded design."""
    features = np.atleast_2d(np.asarray(features, dtype=np.float64))
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for t in range(len(snapshots) - 1):
        prev = np.asarray(snapshots[t])
        nxt = np.asarray(snapshots[t + 1])
        n = prev.shape[0]
        if features.shape[0] != n * (n - 1) // 2:
            features = features.T
        rows, cols = np.triu_indices(n, k=1)
        D, labels_prev = _degree_feature(prev, d, degree_mode)
        state = (nxt[rows, cols] > 0)
        design = np.column_stack([D, features])
        if allow_removal:
            Xs.append(design)
            ys.append(state.astype(np.int8))
        else:
            at_risk = labels_prev == 0
            Xs.append(design[at_risk])
            ys.append(state[at_risk].astype(np.int8))
    k = features.shape[1]
    X = np.vstack(Xs) if Xs else np.empty((0, 1 + k), dtype=np.float64)
    y = np.concatenate(ys) if ys else np.empty(0, dtype=np.int8)
    return X, y


def fit_multi_params(X: np.ndarray, labels: np.ndarray, feature_names: list | None = None) -> dict:
    """Fit logit(P) = sigma + alpha*D + sum_k beta_k*F_k by pooled logistic regression — the exact,
    consistent MLE for the multi-feature temporal model. Returns sigma, alpha, the per-feature
    coefficients and their SEs, the log-likelihood, AIC, and n_params (solver fallbacks for robustness)."""
    y = np.asarray(labels, dtype=int)
    X = np.asarray(X, dtype=np.float64)
    k = X.shape[1] - 1
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(k)]
    Xc = sm.add_constant(X, has_constant="add")
    res = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("newton", "bfgs", "lbfgs"):
            try:
                r = sm.Logit(y, Xc).fit(method=method, disp=0, maxiter=300)
                if np.isfinite(r.llf):
                    res = r
                    break
            except Exception:
                continue
        if res is None:
            res = sm.Logit(y, Xc).fit_regularized(method="l1", alpha=1e-4, disp=0)

    b = np.asarray(res.params, dtype=np.float64)
    try:
        se = np.asarray(res.bse, dtype=np.float64)
    except Exception:
        se = np.full(b.shape[0], np.nan)
    ll = float(res.llf)
    n_params = 2 + k
    out = {
        "sigma": float(b[0]),
        "alpha": float(b[1]),
        "se_sigma": float(se[0]),
        "se_alpha": float(se[1]),
        "coefs": {feature_names[j]: float(b[2 + j]) for j in range(k)},
        "se_coefs": {feature_names[j]: float(se[2 + j]) for j in range(k)},
        "ll": ll,
        "aic": 2.0 * n_params - 2.0 * ll,
        "n_params": n_params,
    }
    return out
