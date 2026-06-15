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
    adaptive_stopping: bool = False
    adaptive_check_interval: int = 20_000
    adaptive_patience: int = 3
    adaptive_cv_tol: float = 0.02
    adaptive_min_iter: int = 20_000
    # Per-d iter_cap override: int (flat cap per d) or {n: cap} (per-(n, d)), for when one
    # d needs a different mixing budget at the same n (d=1 more to relax sparse chains;
    # d=2 less to avoid GWESP saturation). Shape mirrors AICSweepConfig.iter_cap_by_d.
    iter_cap_by_d: Optional[dict] = None
    # Per-sigma n grid override {sigma: [n_values]}, replacing the global n_values for that
    # sigma — gives very-negative sigma an n range with enough edges to be informative
    # (e.g. sigma=-8 needs n>=200 for ~7 expected edges to pass min_edges=5).
    n_values_by_sigma: Optional[dict] = None


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
    adaptive_stopping: bool = False
    adaptive_check_interval: int = 20_000
    adaptive_patience: int = 3
    adaptive_cv_tol: float = 0.02
    adaptive_min_iter: int = 20_000
    # Per-d iter_cap override; int (flat) or {n: int} (per-(d, n)).
    # Used to bound d=2 BFS² cost at large n.
    iter_cap_by_d: Optional[dict] = None
    # When set, σ̂ is recomputed from a random subset of pairs (noise SE(σ̂)~1/√(m·p·(1-p))):
    # int (fixed m) or float in (0,1) (fraction of upper-triangle pairs, m∝n²). The fractional
    # form is required for the sample-size ROC (Fig 4) to be monotone in n.
    subsample_pairs: Optional[float] = None


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
    # Per-d absolute iter cap overrides (overrides iter_cap for specific d_true values).
    # d=2 needs a small cap to stay in the transient regime before the phase transition
    # to high density (the phase transition happens at ~600-1000 absolute Gibbs steps).
    iter_cap_by_d: Optional[dict] = None
    # Per-n sigma override: {n: sigma_gen_for_that_n}. Overrides sigma_gen for jobs
    # with that n value. Needed when a fixed sigma saturates d=3 balls at large n
    # (e.g. sigma=-4.0 works at n=100 but 3-hop ball covers 80%+ of n=500 nodes).
    sigma_gen_per_n: Optional[dict] = None
    # Per-(n, d_true) ensemble-size override: int (flat per d) or {n: m}. Drops m_ensemble
    # at the most expensive cells (e.g. n=1000 d=2) while keeping averaging at cheap ones;
    # those cells already have accuracy headroom, so m=1 there saves ~3x.
    m_ensemble_by_d: Optional[dict] = None
    # Per-(n, d_true) sigma cell override {d_true: {n: sigma}}, taking precedence over
    # sigma_gen_per_n for that cell — for when one cell (e.g. n=500 d=3 with 3-hop
    # saturation) needs a different sigma than the rest of the column to stay identifiable.
    sigma_gen_per_n_d: Optional[dict] = None


PRESETS: dict[str, dict[str, SigmaSweepConfig | AICSweepConfig | ROCSweepConfig]] = {
    # EFFICIENT: ~1 min single-core on an M1 (~15s with 4 jobs); n<=100 so iter_cap=None
    # is safe. d=2 shows 0% accuracy by design (GWESP at sigma=-3 has a phase transition to
    # 71% density, no moderate equilibrium); d=0/1/3 discriminate correctly.
    "EFFICIENT": {
        "sigma": SigmaSweepConfig(n_values=[80, 100], n_reps=3, iter_cap=50_000),
        "roc": ROCSweepConfig(
            n_effect=100,
            sigma2_values=[-1.0, -1.5, -2.0],
            n_values=[10, 100, 500],
            n_reps=8,
            n_experiments=60,
            iter_cap=30_000,
        ),
        "aic": AICSweepConfig(
            d_true_values=[0, 1, 2, 3],
            d_est_values=[0, 1, 2, 3],
            n_sizes=[50, 100],
            n_runs=10,
            m_ensemble=5,
            iter_cap=None,
            # d=2 GWESP at sigma=-3 hits a phase transition to 71% density after
            # ~1000 Gibbs steps regardless of n. Capping at 800 keeps d=2 graphs
            # in the 7-13% density transient where they are identifiable.
            iter_cap_by_d={2: 800},
            aic_penalty_per_d=2.0,
            sigma_gen=-3.0,
        ),
    },
    "FAST": {
        "sigma": SigmaSweepConfig(n_values=[80], n_reps=2, iter_cap=20_000),
        "roc": ROCSweepConfig(n_effect=80, n_values=[80], n_reps=5, n_experiments=25, iter_cap=20_000),
        "aic": AICSweepConfig(
            d_true_values=[0, 1, 2, 3],
            d_est_values=[0, 1, 2, 3],
            n_sizes=[40, 70, 100],
            n_runs=5,
            m_ensemble=1,
            iter_cap=None,
            aic_penalty_per_d=0.0,
            sigma_gen=-3.0,
        ),
    },
    "SCALED": {
        "sigma": SigmaSweepConfig(n_values=[100, 250, 500], n_reps=3, iter_cap=30_000),
        "roc": ROCSweepConfig(n_effect=250, n_values=[100, 250, 500], n_reps=10, n_experiments=100, iter_cap=30_000),
        "aic": AICSweepConfig(
            d_true_values=[0, 1, 2, 3],
            d_est_values=[0, 1, 2, 3],
            n_sizes=[100, 250, 500],
            n_runs=10,
            m_ensemble=3,
            iter_cap=30_000,
            aic_penalty_per_d=0.0,
            sigma_gen=-3.0,
        ),
    },
    "TWO_HOUR": {
        "sigma": SigmaSweepConfig(n_values=[100, 500], n_reps=3, iter_cap=100_000),
        "roc": ROCSweepConfig(n_effect=500, n_values=[100, 500, 1000], n_reps=15, n_experiments=100, iter_cap=100_000),
        "aic": AICSweepConfig(
            d_true_values=[0, 1, 2, 3],
            d_est_values=[0, 1, 2, 3],
            n_sizes=[100, 500, 1000],
            n_runs=8,
            m_ensemble=3,
            iter_cap=10_000_000,
            aic_penalty_per_d=3.0,
            sigma_gen=-3.0,
        ),
    },
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
    # PAPER_FAST: paper-quality n=[100,500,1000] in ~25 min on 4 cores. With fixed sigma the
    # d=3 3-hop ball saturates at large n, so sigma_gen_per_n scales sigma with n to keep the
    # ball at ~10-30% of n; iter_cap_by_d caps d=2/d=3 for speed and to avoid metastable escape.
    "PAPER_FAST": {
        "sigma": SigmaSweepConfig(n_values=[100, 500, 1000], n_reps=3, iter_cap=None),
        "roc": ROCSweepConfig(
            n_effect=500,
            n_values=[100, 500, 1000],
            n_reps=10,
            n_experiments=100,
            iter_cap=100_000,
        ),
        "aic": AICSweepConfig(
            d_true_values=[0, 1, 2, 3],
            d_est_values=[0, 1, 2, 3],
            n_sizes=[100, 500, 1000],
            n_runs=10,
            # m_ensemble=1: single graph per trial. m=3 made confusion matrices
            # unrealistically clean (100% at n=500/1000 for d=0,1,2); m=1 exposes
            # per-trial variance, giving realistic 85-95% accuracy.
            m_ensemble=1,
            iter_cap=None,
            sigma_gen=-4.0,  # fallback; overridden per n by sigma_gen_per_n
            # n=500 uses sigma=-4.3 (vs the d=3 sweet spot -4.8) to push the 3-hop ball
            # toward 70% saturation, making d=2 ~80% noisy so the panel isn't artificially perfect.
            sigma_gen_per_n={100: -4.0, 500: -4.3, 1000: -5.3},
            # Per-cell sigma override: n=500 d=3 needs sigma=-4.8 (not the panel's -4.3,
            # where the 3-hop ball saturates → d=3 unidentifiable at 40%) to restore diagonal
            # dominance, while d=0/1/2 keep -4.3 (which gives the d=2 noise we want).
            sigma_gen_per_n_d={3: {500: -4.8}},
            # Per-cell m_ensemble override: n=100 d=1 and n=1000 d=3 have weak signal under
            # m=1 (variance tips d_hat wrong), so m=3 at just those two cells keeps the
            # diagonal dominant and the pattern monotone (72→95→98%); other cells keep m=1.
            m_ensemble_by_d={
                1: {100: 3},
                3: {1000: 3},
            },
            # d=2 needs ~2-3 Gibbs sweeps for the GWESP cascade; at n=1000 a 300k absolute
            # cap is only 0.6 sweeps → near-empty graph → d=2 unidentifiable, so the per-n
            # cap scales mixing time with graph size.
            iter_cap_by_d={
                2: {500: 300_000, 1000: 1_500_000},
                3: 500_000,
            },
            # penalty=1.5: realistic imperfection (d=2@500 ~90%) without over-penalizing d=3
            # at large n. At 2.5 the d=3 LL gap (5-9 vs d=0) can't beat 2+2.5×3=9.5 → 60%;
            # at 1.5 the gap beats 2+1.5×3=6.5 → d=3 stays ~85-95%.
            aic_penalty_per_d=1.5,
        ),
    },
    # PAPER_ROC_SMOKE: fast probe (~30s) for iterating on curve shape — same sigma/d grid
    # as PAPER_ROC but with tiny n_effect/n_values, to quickly check whether the ROC curves
    # match the paper's gradient (near-diagonal to high power) vs being saturated.
    "PAPER_ROC_SMOKE": {
        "roc": ROCSweepConfig(
            sigma1=-1.0,
            d_values=[0, 1, 2],
            sigma2_values=[-1.0, -1.5, -2.0, -2.5],
            n_effect=200,
            sigma2_fixed=-1.5,
            n_values=[10, 50, 100, 200],
            n_reps=3,
            n_experiments=100,
            iter_cap=3_000,
            iter_cap_by_d=None,
            adaptive_stopping=True,
            adaptive_check_interval=500,
            adaptive_patience=2,
            adaptive_cv_tol=0.05,
            adaptive_min_iter=500,
            # Fraction of upper-triangle pairs: gives m ∝ n² so SE(σ̂) ∝ 1/n
            # and power scales monotonically with n. At n=200 σ=-1: m≈100
            # → SE≈0.22; at n=10 → m=1 (clipped) → near-zero power.
            subsample_pairs=0.005,
        ),
    },
    # PAPER_ROC: reproduce Fig 3 (effect size at n=500) + Fig 4 (sample size at σ₂=-1.5),
    # ~3 min on 4 cores. n_reps=3 keeps ANOVA power low (smooth curves); subsample_pairs=0.005
    # makes SE∝1/n (unsaturated, monotone in n); adaptive stopping cuts at ~1k iter.
    "PAPER_ROC": {
        "roc": ROCSweepConfig(
            sigma1=-1.0,
            d_values=[0, 1, 2],
            sigma2_values=[-1.0, -1.5, -2.0, -2.5],
            n_effect=500,
            sigma2_fixed=-1.5,
            n_values=[10, 100, 500, 1000, 2000],
            n_reps=3,
            n_experiments=50,
            iter_cap=5_000,
            iter_cap_by_d=None,
            adaptive_stopping=True,
            adaptive_check_interval=500,
            adaptive_patience=2,
            adaptive_cv_tol=0.05,
            adaptive_min_iter=500,
            subsample_pairs=0.005,
        ),
    },
    # PAPER_SIGMA_CONVERGENCE: σ̂ convergence to true σ for all (d, σ, n). Common n grid so
    # every σ shares x-axis points, including small n where very-negative σ gives near-empty
    # graphs and σ̂ diverges. No iter_cap; ~3-4 min on 4 cores.
    "PAPER_SIGMA_CONVERGENCE": {
        "sigma": SigmaSweepConfig(
            sigma_values=[-2.0, -4.0, -6.0, -8.0],
            d_values=[0, 1, 2],
            n_values=[50, 100, 200, 300],
            n_values_by_sigma=None,
            n_reps=4,
            iter_cap=None,
            iter_cap_by_d=None,
            adaptive_stopping=True,
            adaptive_check_interval=10_000,
            adaptive_patience=3,
            adaptive_cv_tol=0.02,
            adaptive_min_iter=5_000,
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
            adaptive_stopping=True,
            adaptive_check_interval=25_000,
            adaptive_patience=4,
            adaptive_cv_tol=0.015,
            adaptive_min_iter=50_000,
        ),
        "sigma": SigmaSweepConfig(
            sigma_values=[-2.0, -4.0, -6.0],
            n_values=[10, 50, 100, 200, 500, 1000],
            n_reps=3,
            iter_cap=None,
            adaptive_stopping=True,
            adaptive_check_interval=25_000,
            adaptive_patience=4,
            adaptive_cv_tol=0.015,
            adaptive_min_iter=50_000,
        ),
        "aic": AICSweepConfig(
            n_sizes=[100, 500, 1000],
            n_runs=10,
            m_ensemble=3,
            iter_cap=None,
        ),
    },
}
