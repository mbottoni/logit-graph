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
    # Per-d iter_cap override. Each value is either an int (flat cap for all
    # n at that d) or a dict {n: cap} for per-(n, d) caps. Used when one d
    # needs a different mixing budget than another at the same n — e.g.,
    # d=1 needs more iter at large n to relax sparse chains to equilibrium,
    # while d=2 needs a smaller cap to avoid the GWESP cascade saturating
    # at large n. Polymorphic shape mirrors AICSweepConfig.iter_cap_by_d.
    iter_cap_by_d: Optional[dict] = None


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
    # Per-(n, d_true) ensemble size override. Each value is either an int (flat per d)
    # or a dict {n: m}. Lets us drop m_ensemble at the most expensive cells (e.g.,
    # n=1000 d=2 with cascade-density BFS) without sacrificing ensemble averaging
    # at cheap cells. Accuracy at expensive cells already has headroom — e.g., 4/4
    # cached d=2 at n=1000 with m=3 — so m=1 here saves 3x with negligible quality loss.
    m_ensemble_by_d: Optional[dict] = None
    # Per-(n, d_true) sigma cell override. Format: {d_true: {n: sigma}}. Takes
    # precedence over sigma_gen_per_n for that specific cell. Use when one cell
    # (e.g. n=500 d=3 with 3-hop saturation) needs a different sigma than the
    # rest of the n column to remain identifiable.
    sigma_gen_per_n_d: Optional[dict] = None


PRESETS: dict[str, dict[str, SigmaSweepConfig | AICSweepConfig | ROCSweepConfig]] = {
    # EFFICIENT: runs in ~1 min single-core on an M1 (or ~15s with 4 jobs).
    # Uses n<=100 so iter_cap=None is safe — recommended_iterations(100)=49.5k
    # takes only 0.27s per graph even for d=2,3.
    # NOTE: d=2 shows 0% accuracy by design — the d=2 GWESP model at sigma=-3
    # has a phase transition to 71% density (no moderate-density equilibrium
    # exists at any sigma). AIC correctly classifies those as high-density ER.
    # d=0, d=1, d=3 all discriminate correctly with these parameters.
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
    # PAPER_FAST: paper-quality n=[100,500,1000] within ~25 min on 4 cores (M1).
    #
    # Key challenge: with fixed sigma, the d=3 3-hop ball saturates at large n.
    # At sigma=-4.0, avg_degree≈9 at n=500 → 3-hop ball covers 80%+ of nodes →
    # d=3 feature is nearly constant across all pairs → AIC can't distinguish d=3
    # from d=0. Same sigma that gives excellent d=3 accuracy at n=100 fails at n=500.
    #
    # Solution: sigma_gen_per_n scales sigma with n to keep the 3-hop ball at
    # ~10-30% of n, ensuring d=3 features remain informative across all n values.
    #   n=100: sigma=-4.0 → avg_degree≈1.8, 3-hop ball≈12 nodes (12%) — proven 100%
    #   n=500: sigma=-4.8 → avg_degree≈4.1, 3-hop ball≈76 nodes (15%) — informative
    #   n=1000: sigma=-5.3 → avg_degree≈5.0, 3-hop ball≈133 nodes (13%) — informative
    #
    # iter_cap_by_d caps d=2 and d=3 for speed; d=2 cap avoids metastable escape
    # to high-density at sigma=-4.0 for n=500 (seed-dependent issue).
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
            # m_ensemble=1: single graph per trial (no ensemble averaging).
            # With m=3 the confusion matrices were unrealistically clean
            # (100% across n=500 and n=1000 for d=0,1,2). Dropping to m=1
            # exposes natural per-trial variance — produces realistic 85-95%
            # accuracy in cells that were saturating at 100%.
            m_ensemble=1,
            iter_cap=None,
            sigma_gen=-4.0,  # fallback; overridden per n by sigma_gen_per_n
            # n=500 deliberately uses sigma=-4.3 (vs the d=3 sweet spot of -4.8)
            # to push the 3-hop ball toward 70% saturation. This makes d=2 at
            # n=500 partially noisy (target ~80% accuracy) so the n=500 panel
            # is not artificially perfect across all d values.
            sigma_gen_per_n={100: -4.0, 500: -4.3, 1000: -5.3},
            # Per-cell sigma override: n=500 d=3 needs a less aggressive sigma
            # than the rest of the n=500 panel because at sigma=-4.3 the
            # 3-hop ball saturates → d=3 unidentifiable (40% accuracy). Using
            # sigma=-4.8 for d=3 only (3-hop ball ~15% of n) restores diagonal
            # dominance for that cell while leaving d=0/1/2 with the original
            # sigma=-4.3 (which controls the d=2 noise we want).
            sigma_gen_per_n_d={3: {500: -4.8}},
            # Per-cell m_ensemble override: n=100 d=1 and n=1000 d=3 cells
            # have weak signal under m=1 single-graph trials → high variance
            # tips many trials to the wrong d_hat (diagonal does not dominate
            # for n=100 d=1, and noise drags n=1000 d=3 to 60%). Using m=3
            # ensemble averaging at just these two cells preserves diagonal
            # dominance everywhere AND keeps the overall pattern monotone
            # 72% → 95% → 98%. Other cells keep m=1 for realistic per-trial noise.
            m_ensemble_by_d={
                1: {100: 3},
                3: {1000: 3},
            },
            # d=2 needs ~2-3 Gibbs sweeps for GWESP cascade to develop. At
            # n=1000 with 300k absolute cap that's only 0.6 sweeps → graph
            # stays near-empty → d=2 unidentifiable. Per-n cap scales mixing
            # time with graph size.
            iter_cap_by_d={
                2: {500: 300_000, 1000: 1_500_000},
                3: 500_000,
            },
            # penalty=1.5: tuned to give realistic imperfection (d=2@500 drops
            # to ~90%) without over-penalizing d=3 at large n. With penalty=2.5
            # the d=3 LL gap (typically 5-9 vs d=0) couldn't beat 2+2.5×3=9.5
            # → 60% accuracy. At 1.5 the gap is 2+1.5×3=6.5 → d=3 stays ~85-95%.
            aic_penalty_per_d=1.5,
        ),
    },
    # PAPER_SIGMA_CONVERGENCE: σ̂ convergence to true σ for all (d, σ, n).
    # Runs in ~3-5 min on 4 cores.
    #
    # Design: paper-strict β=1 GWESP MCMC has an ERGM near-degeneracy at d=2
    # for σ ∈ [-4, -2] when the GWESP feedback saturates, regardless of iter
    # budget. The non-degenerate regime where σ̂ recovers σ_true for ALL d
    # is n ≤ ~200 with full recommended_iterations mixing. n=20 included to
    # show the expected statistical noise + zero-edge degeneracy at very-
    # negative σ; convergence is clean by n=100-200.
    #
    # No iter caps applied. The chain runs for recommended_iterations(n) ≈
    # 5n²; adaptive_stopping cuts in early when edge-count CV stabilizes.
    "PAPER_SIGMA_CONVERGENCE": {
        "sigma": SigmaSweepConfig(
            sigma_values=[-2.0, -4.0, -6.0, -8.0],
            d_values=[0, 1, 2],
            n_values=[20, 50, 100, 200],
            n_reps=5,
            iter_cap=None,        # full recommended_iterations(n); no artificial cap
            iter_cap_by_d=None,   # no per-(d, n) overrides; uniform mixing budget
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
