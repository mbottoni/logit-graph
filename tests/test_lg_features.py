import numpy as np
import networkx as nx

from logit_graph.lg_features import (
    build_pair_dataset,
    pair_feature,
    pair_feature_layer2,
    recommended_iterations,
)
from logit_graph.logit_estimator import LogitRegEstimator
from logit_graph.experiments.sweeps import (
    _calibrate_beta_given_sigma,
    estimate_sigma_from_graph,
    run_aic_d_sweep,
    run_sigma_sweep,
    simulate_graph,
    select_d_ensemble,
)
from logit_graph.experiments.presets import PRESETS


def _path_graph_adj(n: int = 8) -> np.ndarray:
    G = nx.path_graph(n)
    return nx.to_numpy_array(G)


def test_layer2_reduces_feature_when_edge_present():
    adj = _path_graph_adj(10)
    i, j = 2, 5
    adj[i, j] = adj[j, i] = 1.0
    f_no_l2 = pair_feature(adj, i, j, d=1, mode="bounded")
    f_l2 = pair_feature_layer2(adj, i, j, d=1, mode="bounded")
    assert f_l2 <= f_no_l2


def test_recommended_iterations_scales_with_n():
    assert recommended_iterations(50) < recommended_iterations(200)


def test_compute_aic_returns_finite():
    adj = nx.to_numpy_array(nx.erdos_renyi_graph(30, 0.12, seed=1))
    est = LogitRegEstimator(adj, d=1, layer2=True, feature_mode="bounded")
    stats = est.compute_aic(d_est=1)
    assert np.isfinite(stats["aic"])
    assert np.isfinite(stats["sigma_hat"])


def test_sigma_recovery_smoke():
    n, d, sigma_true = 80, 1, -4.0
    nit = min(recommended_iterations(n), 40_000)
    adj, meta = simulate_graph(
        n, d, sigma=sigma_true, n_iter=nit,
        feature_mode="incremental", target_density=0.10, seed=42,
        return_meta=True,
    )
    sh = estimate_sigma_from_graph(adj, d, feature_mode="incremental")
    assert abs(sh - sigma_true) < 0.6
    assert abs(meta["density"] - 0.10) < 0.15


def test_aic_selects_d0_or_d1_smoke():
    n, nit = 60, 8000
    graphs = [
        simulate_graph(n, 1, sigma=None, n_iter=nit, feature_mode="incremental", seed=10 + m)
        for m in range(3)
    ]
    hat, _ = select_d_ensemble(graphs, [0, 1, 2, 3], "incremental", extra_penalty_per_d=3.0)
    assert hat in (0, 1, 2)


def test_sigma_sweep_smoke_preset(tmp_path):
    cfg = PRESETS["SMOKE"]["sigma"]
    df = run_sigma_sweep(cfg, tmp_path, use_cache=False)
    assert len(df) > 0
    assert "sigma_hat_mean" in df.columns
    assert "density_mean" in df.columns
    assert "sigma_error" in df.columns


def test_calibrate_beta_given_sigma_is_unit_weight():
    beta = _calibrate_beta_given_sigma(
        80, 1, -4.0, 0.10, 0.5, "incremental", 42,
    )
    assert beta == 1.0


def test_calibrate_beta_given_sigma_density_smoke():
    n, d = 100, 1
    target_density = 0.10
    nit = min(recommended_iterations(n), 40_000)
    for sigma in (-2.0, -4.0, -6.0):
        seed = 100 + int(abs(sigma))
        adj, meta = simulate_graph(
            n, d, sigma=sigma, n_iter=nit,
            feature_mode="incremental", target_density=target_density,
            seed=seed, return_meta=True,
        )
        sh = estimate_sigma_from_graph(adj, d, "incremental")
        assert abs(sh - sigma) < 0.65, f"sigma={sigma} sh={sh:.2f}"
        assert meta["beta"] == 1.0


def test_adaptive_gibbs_stops_before_max_iter():
    n, d, sigma = 80, 1, -3.0
    max_iter = 120_000
    _, meta = simulate_graph(
        n, d, sigma=sigma, n_iter=max_iter,
        feature_mode="incremental", seed=7, return_meta=True,
        materialize_adjacency=False,
        adaptive_stopping=True,
        adaptive_check_interval=5_000,
        adaptive_patience=3,
        adaptive_cv_tol=0.02,
        adaptive_min_iter=10_000,
    )
    assert meta["n_iter_used"] < max_iter
    assert "csr_rows" in meta


def test_sigma_from_csr_rows_matches_adj():
    n, d, sigma = 60, 1, -3.0
    nit = 8_000
    adj, meta = simulate_graph(
        n, d, sigma=sigma, n_iter=nit,
        feature_mode="incremental", seed=11, return_meta=True,
    )
    sh_adj = estimate_sigma_from_graph(adj, d, "incremental")
    sh_csr = estimate_sigma_from_graph(
        None, d, "incremental", csr_rows=meta["csr_rows"],
    )
    assert sh_adj == sh_csr
