import os
import sys
import pathlib

import pytest

from logit_graph.experiments.presets import AICSweepConfig, SigmaSweepConfig


@pytest.fixture()
def tiny_sigma_cfg() -> SigmaSweepConfig:
    """Minimal sigma sweep for unit tests (serial/parallel smoke)."""
    return SigmaSweepConfig(
        sigma_values=[-4.0],
        d_values=[0, 1],
        n_values=[30],
        n_reps=1,
        iter_cap=500,
        seed_base=0,
    )


@pytest.fixture()
def tiny_aic_cfg() -> AICSweepConfig:
    """Minimal AIC sweep for unit tests (serial/parallel smoke)."""
    return AICSweepConfig(
        d_true_values=[0, 1],
        d_est_values=[0, 1],
        n_sizes=[30],
        n_runs=1,
        m_ensemble=1,
        iter_cap=500,
        seed_base=0,
    )


def pytest_sessionstart(session):
    # Ensure `src` is on sys.path so `import logit_graph` works in tests
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_path = repo_root / 'src'
    if src_path.exists():
        sys.path.insert(0, str(src_path))


