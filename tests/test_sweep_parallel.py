"""Serial vs parallel sweep identity tests."""
from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from logit_graph.experiments.presets import AICSweepConfig, SigmaSweepConfig
from logit_graph.experiments.sweeps import run_aic_d_sweep, run_sigma_sweep


@pytest.fixture()
def tiny_sigma_cfg() -> SigmaSweepConfig:
    return SigmaSweepConfig(
        sigma_values=[-2.0],
        d_values=[0, 1],
        n_values=[30],
        n_reps=2,
        iter_cap=500,
        seed_base=0,
    )


@pytest.fixture()
def tiny_aic_cfg() -> AICSweepConfig:
    return AICSweepConfig(
        d_true_values=[0, 1],
        d_est_values=[0, 1],
        n_sizes=[30],
        n_runs=2,
        m_ensemble=2,
        iter_cap=500,
        seed_base=0,
    )


def _sort_sigma(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["d", "sigma_true", "n"]
    return df.sort_values(cols).reset_index(drop=True)


def test_sigma_sweep_serial_vs_parallel(tiny_sigma_cfg, tmp_path):
    serial = run_sigma_sweep(
        tiny_sigma_cfg, tmp_path / "serial", use_cache=False, n_jobs=1,
    )
    parallel = run_sigma_sweep(
        replace(tiny_sigma_cfg, seed_base=0),
        tmp_path / "parallel", use_cache=False, n_jobs=2,
    )
    pd.testing.assert_frame_equal(
        _sort_sigma(serial), _sort_sigma(parallel), check_exact=False, rtol=1e-9,
    )


def test_aic_sweep_serial_vs_parallel(tiny_aic_cfg, tmp_path):
    _, conf_serial = run_aic_d_sweep(
        tiny_aic_cfg, tmp_path / "serial", use_cache=False, n_jobs=1,
    )
    df_parallel, conf_parallel = run_aic_d_sweep(
        tiny_aic_cfg, tmp_path / "parallel", use_cache=False, n_jobs=2,
    )
    assert conf_serial == conf_parallel
    assert len(df_parallel) == tiny_aic_cfg.n_runs * len(tiny_aic_cfg.d_true_values)
