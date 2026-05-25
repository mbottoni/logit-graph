"""Experiment presets for INSIGHT / SMOKE / DEV / PAPER tiers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SigmaSweepConfig:
    sigma_values: list[float] = field(default_factory=lambda: [-2.0, -4.0, -6.0, -8.0])
    d_values: list[int] = field(default_factory=lambda: [0, 1, 2])
    n_values: list[int] = field(default_factory=lambda: [50, 100])
    n_reps: int = 4
    iter_cap: Optional[int] = 80_000
    target_density: float = 0.10
    signal: float = 0.5
    feature_mode_gen: str = "incremental"
    feature_mode_est: str = "incremental"
    seed_base: int = 0


@dataclass
class ROCSweepConfig:
    """ANOVA-on-sigma_hat ROC sweeps (paper fig:roc_effect / fig:roc_sample)."""

    sigma1: float = -1.0
    d_values: list[int] = field(default_factory=lambda: [0, 1, 2])
    sigma2_values: list[float] = field(default_factory=lambda: [-1.0, -1.5, -2.0, -2.5])
    n_effect: int = 500
    sigma2_fixed: float = -1.5
    n_values: list[int] = field(default_factory=lambda: [10, 100, 500, 1000, 2000])
    n_reps: int = 30
    n_experiments: int = 500
    iter_cap: Optional[int] = 80_000
    target_density: float = 0.10
    signal: float = 0.5
    feature_mode_gen: str = "incremental"
    feature_mode_est: str = "incremental"
    seed_base: int = 2000


@dataclass
class AICSweepConfig:
    d_true_values: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    d_est_values: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    n_sizes: list[int] = field(default_factory=lambda: [100])
    n_runs: int = 4
    m_ensemble: int = 3
    iter_cap: Optional[int] = 30_000
    target_density: float = 0.10
    signal: float = 0.5
    aic_penalty_per_d: float = 3.0
    feature_mode_gen: str = "incremental"
    feature_mode_est: str = "incremental"
    seed_base: int = 1000
    # Paper-strict: fixed sigma + beta=1 in generation (matches estimator).
    # Sigma is also used for d=0 ER baseline.
    sigma_gen: float = -3.0


PRESETS: dict[str, dict[str, SigmaSweepConfig | AICSweepConfig | ROCSweepConfig]] = {
    "INSIGHT_SCALING": {
        "roc": ROCSweepConfig(
            n_effect=200,
            n_values=[50, 100, 200],
            n_reps=10,
            n_experiments=100,
            iter_cap=50_000,
        ),
        "sigma": SigmaSweepConfig(
            sigma_values=[-2.0, -4.0, -6.0],
            d_values=[0, 1, 2],
            n_values=[50, 100, 200],
            n_reps=3,
            iter_cap=200_000,
        ),
        "aic": AICSweepConfig(
            d_true_values=[0, 1, 2],
            d_est_values=[0, 1, 2],
            n_sizes=[50, 100, 200],
            n_runs=3,
            m_ensemble=3,
            iter_cap=20_000,
            aic_penalty_per_d=1.0,
            sigma_gen=-3.0,
        ),
    },
    "INSIGHT": {
        "roc": ROCSweepConfig(
            n_effect=100,
            sigma2_values=[-1.0, -1.5, -2.0],
            n_values=[10, 100, 500],
            n_reps=8,
            n_experiments=60,
            iter_cap=30_000,
        ),
        "sigma": SigmaSweepConfig(
            n_values=[80, 100],
            n_reps=2,
            iter_cap=20_000,
        ),
        "aic": AICSweepConfig(
            d_true_values=[0, 1, 2, 3],
            d_est_values=[0, 1, 2, 3],
            n_sizes=[80],
            n_runs=3,
            m_ensemble=5,
            iter_cap=30_000,
            aic_penalty_per_d=1.0,
            sigma_gen=-3.0,
        ),
    },
    "SMOKE": {
        "roc": ROCSweepConfig(
            n_effect=80,
            sigma2_values=[-1.0, -1.5, -2.0],
            n_values=[10, 80, 200],
            d_values=[0, 1],
            n_reps=5,
            n_experiments=25,
            iter_cap=20_000,
        ),
        "sigma": SigmaSweepConfig(
            n_values=[50, 100],
            n_reps=4,
            iter_cap=80_000,
        ),
        "aic": AICSweepConfig(
            n_sizes=[100],
            n_runs=4,
            m_ensemble=3,
            iter_cap=30_000,
        ),
    },
    "DEV": {
        "roc": ROCSweepConfig(
            n_effect=300,
            n_values=[10, 100, 500, 1000],
            n_reps=15,
            n_experiments=200,
            iter_cap=120_000,
        ),
        "sigma": SigmaSweepConfig(
            n_values=[10, 50, 100, 200],
            n_reps=8,
            iter_cap=200_000,
        ),
        "aic": AICSweepConfig(
            n_sizes=[100, 200],
            n_runs=8,
            m_ensemble=5,
            iter_cap=120_000,
        ),
    },
    "PAPER": {
        "roc": ROCSweepConfig(
            n_effect=500,
            sigma2_values=[-1.0, -1.5, -2.0, -2.5],
            n_values=[10, 100, 500, 1000, 2000],
            n_reps=30,
            n_experiments=500,
            iter_cap=None,
        ),
        "sigma": SigmaSweepConfig(
            n_values=[10, 50, 100, 200, 500, 1000, 2000],
            n_reps=15,
            iter_cap=None,
        ),
        "aic": AICSweepConfig(
            n_sizes=[100, 500, 1000],
            n_runs=10,
            m_ensemble=3,
            iter_cap=None,
        ),
    },
}
