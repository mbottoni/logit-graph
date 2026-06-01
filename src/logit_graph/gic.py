from __future__ import annotations

import os

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.stats import entropy
from scipy.spatial.distance import euclidean, cityblock
from typing import Any, Callable, Optional, Union

_MODEL_N_PARAMS: dict[str, int] = {
    "ER": 1,
    "BA": 1,
    "WS": 2,
    "GRG": 1,
    "KR": 1,
    "LG": 1,
    # SBM has k(k+1)/2 block probabilities; the exact count is graph-
    # dependent. ``_get_n_params`` returns 1 as a safe default and callers
    # that care about the real count read it from ``sbm.fit_sbm_from_graph``.
    "SBM": 1,
}

# Switch to KPM (Kernel Polynomial Method) approximation of the normalized
# Laplacian spectral density once n > KPM_THRESHOLD. Exact dense eigvalsh
# is O(n³) / O(n²) memory; KPM is O(M·P·nnz) with M moments and P probes,
# so it stays cheap for sparse social networks at n in the thousands.
KPM_THRESHOLD = int(os.environ.get("LG_GIC_KPM_THRESHOLD", "500"))
KPM_N_MOMENTS = int(os.environ.get("LG_GIC_KPM_MOMENTS", "60"))
KPM_N_PROBES = int(os.environ.get("LG_GIC_KPM_PROBES", "20"))
KPM_SEED = int(os.environ.get("LG_GIC_KPM_SEED", "0"))


def _jackson_kernel(M: int) -> np.ndarray:
    """Jackson damping coefficients of length M; suppresses Gibbs ringing."""
    k = np.arange(M)
    Mp1 = M + 1
    return (
        (Mp1 - k) * np.cos(np.pi * k / Mp1)
        + np.sin(np.pi * k / Mp1) / np.tan(np.pi / Mp1)
    ) / Mp1


def kpm_spectral_density(
    laplacian: sp.spmatrix,
    n_bins: int = 50,
    n_moments: int = KPM_N_MOMENTS,
    n_probes: int = KPM_N_PROBES,
    seed: int = KPM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic Chebyshev estimator of the normalized-Laplacian density.

    Returns ``(hist, bin_edges)`` matching the signature of
    ``np.histogram(eigvals, bins=n_bins, range=(0, 2), density=True)`` so it
    is a drop-in replacement. Eigenvalues of the normalized Laplacian lie in
    [0, 2]; we rescale to [-1, 1] before invoking Chebyshev recurrences.
    """
    n = laplacian.shape[0]
    # Rescale to [-1, 1]: x = λ − 1
    H = (laplacian - sp.eye(n, format="csr")).astype(np.float64).tocsr()

    rng = np.random.default_rng(seed)
    moments = np.zeros(n_moments)
    for _ in range(n_probes):
        v0 = rng.choice([-1.0, 1.0], size=n)
        # T_0(H) v = v
        v_prev = v0.copy()
        moments[0] += float(np.dot(v0, v_prev))
        if n_moments > 1:
            v_curr = H @ v0
            moments[1] += float(np.dot(v0, v_curr))
            for k in range(2, n_moments):
                v_next = 2.0 * (H @ v_curr) - v_prev
                moments[k] += float(np.dot(v0, v_next))
                v_prev, v_curr = v_curr, v_next
    moments /= float(n_probes * n)

    g = _jackson_kernel(n_moments)
    bin_edges = np.linspace(0.0, 2.0, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    x = centers - 1.0
    # Sum_k c_k g_k μ_k T_k(x), with c_0=1 and c_k=2 for k>=1
    Tprev = np.ones_like(x)
    Tcurr = x.copy()
    f = g[0] * moments[0] * Tprev
    if n_moments > 1:
        f += 2.0 * g[1] * moments[1] * Tcurr
    for k in range(2, n_moments):
        Tnext = 2.0 * x * Tcurr - Tprev
        f += 2.0 * g[k] * moments[k] * Tnext
        Tprev, Tcurr = Tcurr, Tnext
    weight = 1.0 / (np.pi * np.sqrt(np.clip(1.0 - x * x, 1e-12, None)))
    density = np.maximum(weight * f, 0.0)
    # Normalize so the histogram integrates to 1 over [0, 2]
    integral = float(np.trapezoid(density, centers))
    if integral > 0:
        density /= integral
    return density, bin_edges


class GraphInformationCriterion:
    def __init__(
        self,
        graph: nx.Graph,
        model: Union[str, Callable[..., nx.Graph]],
        log_graph: Optional[nx.Graph] = None,
        p: Optional[Union[float, list]] = None,
        dist: str = 'KL',
        **kwargs: Any,
    ) -> None:
        self.graph = graph
        self.log_graph =  log_graph
        self.model = model
        self.parameter = p
        self.dist_type = dist
        self.kwargs = kwargs
        self.n = graph.number_of_nodes()

    def compute_spectral_density(self, graph: nx.Graph) -> tuple[np.ndarray, np.ndarray]:
        n = graph.number_of_nodes()
        laplacian = nx.normalized_laplacian_matrix(graph)
        if n > KPM_THRESHOLD:
            # KPM avoids the O(n³) dense eigendecomposition for large sparse
            # graphs; spectral density is reconstructed from Chebyshev moments.
            return kpm_spectral_density(laplacian, n_bins=50)
        eigenvalues = np.linalg.eigvalsh(laplacian.todense())
        hist, bin_edges = np.histogram(eigenvalues, bins=50, range=(0, 2), density=True)
        return hist, bin_edges

    def generate_model_graph(self) -> nx.Graph:
        if isinstance(self.model, str):
            if self.model == "ER":
                return nx.erdos_renyi_graph(self.n, self.parameter)
            elif self.model == "GRG":
                return nx.random_geometric_graph(self.n, self.parameter)
            elif self.model == "KR":
                return nx.random_regular_graph(int(self.parameter), self.n)
            elif self.model == "WS":
                k = int(np.ceil(np.sqrt(self.n)))
                return nx.watts_strogatz_graph(self.n, k, self.parameter)
            elif self.model == "BA":
                m = max(1, int(self.parameter))  # Ensure m is at least 1
                return nx.barabasi_albert_graph(self.n, m)
            elif self.model == "SBM":
                from .sbm import generate_sbm_from_real

                G_sbm, _ = generate_sbm_from_real(self.graph)
                return G_sbm
            elif self.model == "LG":
                if isinstance(self.log_graph, tuple):
                    return self.log_graph[0]
                else:
                    return self.log_graph

        elif callable(self.model):
            return self.model(self.n, self.parameter)
        else:
            raise ValueError(f"{self.model}: Model definition is not recognized.")

    def _get_n_params(self) -> int:
        """Return the number of free parameters for the current model."""
        if isinstance(self.model, str):
            return _MODEL_N_PARAMS.get(self.model, 1)
        return 1

    def calculate_spectral_distance(self, model_den: Optional[np.ndarray] = None) -> float:
        """Raw spectral distance (KL / L1 / L2) without complexity penalty."""
        graph_den, _ = self.compute_spectral_density(self.graph)
        if model_den is None:
            model_graph = self.generate_model_graph()
            model_den, _ = self.compute_spectral_density(model_graph)

        if self.dist_type == 'KL':
            distance = entropy(graph_den + 1e-10, model_den + 1e-10)
        elif self.dist_type == 'L1':
            distance = cityblock(graph_den, model_den)
        elif self.dist_type == 'L2':
            distance = euclidean(graph_den, model_den)
        else:
            raise ValueError("Unsupported distance type specified.")

        return distance

    def calculate_gic(
        self,
        model_den: Optional[np.ndarray] = None,
        n_params: Optional[int] = None,
    ) -> float:
        """GIC = 2 * spectral_distance + 2 * |theta|  (Eq. 4 in the paper)."""
        distance = self.calculate_spectral_distance(model_den=model_den)
        if n_params is None:
            n_params = self._get_n_params()
        return 2.0 * distance + 2.0 * n_params
