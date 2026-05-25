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


PRESETS: dict[str, dict[str, SigmaSweepConfig | AICSweepConfig]] = {
    "INSIGHT_SCALING": {
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
        "sigma": SigmaSweepConfig(
            n_values=[10, 50, 100, 200, 500, 1000, 2000],
            n_reps=15,
            iter_cap=None,
        ),
        "aic": AICSweepConfig(
            n_sizes=[100, 500, 1000],
            n_runs=25,
            m_ensemble=5,
            iter_cap=None,
        ),
    },
}
