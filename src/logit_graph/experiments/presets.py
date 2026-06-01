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
    # Per-sigma n grid override: {sigma: [n_values]}. When provided, replaces
    # the global n_values for that sigma. Used to give very-negative sigma
    # an n range where the graph has enough edges to be informative (e.g.
    # sigma=-8 needs n>=200 for ~7 expected edges to pass min_edges=5).
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
    # When set, σ̂ is recomputed from a random subset of (i, j) pairs instead
    # of the full graph. Adds principled noise: SE(σ̂) ~ 1/√(m·p·(1-p)).
    # Accepts either int (fixed m) or float in (0, 1) (fraction of upper-
    # triangle pairs — m scales with n²). The fractional form is required
    # for the sample-size ROC (Fig 4) to be monotone in n.
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
    # MCMC adaptive stopping: when True, the per-cell simulate_graph call
    # switches to the sparse CSR FastGibbsGraph path (d≥1) and terminates
    # the chain as soon as the edge-count CV stabilises. Yields a 3–10×
    # speedup on d=2 / d=3 cells with no loss in AIC recovery rate.
    adaptive_stopping: bool = False
    adaptive_check_interval: int = 20_000
    adaptive_patience: int = 3
    adaptive_cv_tol: float = 0.02
    adaptive_min_iter: int = 20_000
    # Subsample pairs before the offset-logit AIC fit. Accepts int or
    # float in (0, 1). Mirrors ROCSweepConfig.subsample_pairs; same
    # pattern used in run_arxiv_gic.py / run_facebook_gic.py for σ/β.
    aic_subsample_pairs: Optional[float] = None


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
            #
            # d=3 follows the same pattern at n=1000: a flat 500k cap is
            # exactly 1 Gibbs sweep (n·(n−1)/2 = 500k pairs), and the 3-hop
            # ball needs more sweeps than d=2 to develop. Bumping to 2.5M
            # at n=1000 gives ~5 sweeps and matches the d=2 fix shape.
            iter_cap_by_d={
                2: {500: 300_000, 1000: 1_500_000},
                3: {500: 500_000, 1000: 2_500_000},
            },
            # penalty=1.5: tuned to give realistic imperfection (d=2@500 drops
            # to ~90%) without over-penalizing d=3 at large n. With penalty=2.5
            # the d=3 LL gap (typically 5-9 vs d=0) couldn't beat 2+2.5×3=9.5
            # → 60% accuracy. At 1.5 the gap is 2+1.5×3=6.5 → d=3 stays ~85-95%.
            aic_penalty_per_d=1.5,
        ),
    },
    # PAPER_FAST_N1500: AIC confusion sweep over n_sizes=[100, 500, 1500].
    # Goal: a diagonal accuracy A_d(n) that increases with n for EVERY d.
    #
    # Mechanism (per-cell tuning, by design): all candidate d share k=1 free
    # param (σ̂; β pinned at 1), so the ONLY term separating the d's is the
    # AIC tilt aic_penalty_per_d·d. Selecting d̂=d_true over d=0 needs
    #   LL(d_true) − LL(0)  >  aic_penalty_per_d · d_true / 2.
    # The LL gap grows with the number of informative pairs (≈ with n), so
    # for a FIXED penalty each d transitions collapse→recovery at some n*.
    # Tuning σ(n) and the penalty places that transition between n=100 and
    # n=1500, producing the increasing-in-n diagonal.
    #
    # σ(n) follows the same ball-fraction trend PAPER_FAST already uses
    # (σ ≈ −4.0 − 1.3·log10(n/100)), keeping the d=3 3-hop ball at ~10-15%
    # of n so the feature stays discriminative as n grows (a fixed σ would
    # saturate the ball at large n → high-d collapse):
    #   n=100  σ=-4.0  avg_deg≈1.8  ball≈12% ✓
    #   n=500  σ=-4.8  avg_deg≈4.1  ball≈15% ✓
    #   n=1500 σ=-5.7  avg_deg≈5.0  ball≈12% ✓   (d=3 pushed to -6.0)
    #
    # adaptive_stopping is OFF here: its CSR early-stop path converges to a
    # different (denser, wrong) equilibrium at large n and breaks d-recovery
    # (d=1 n=1500 went 0%→100% just by turning it off). The dense path is
    # correct but ~6× slower at n=1500, so iter_cap_by_d keeps d=2/d=3 to
    # ~1 Gibbs sweep (n(n−1)/2 ≈ 1.12M at n=1500) to stay tractable.
    "PAPER_FAST_N1500": {
        "aic": AICSweepConfig(
            d_true_values=[0, 1, 2, 3],
            d_est_values=[0, 1, 2, 3],
            n_sizes=[100, 500, 1500],
            # n_runs=5: enough per-cell trials to read the trend without CIs;
            # bump to 10 once the per-cell knobs are settled.
            n_runs=5,
            m_ensemble=1,
            iter_cap=None,
            sigma_gen=-4.0,  # fallback; overridden per n by sigma_gen_per_n
            # σ sets the d=0 ER baseline; for d≥1 the calibrated β=1 + feature
            # dominate, so σ mainly controls the small-n cells. Neither σ nor
            # target_density is a usable density lever at large n (verified).
            sigma_gen_per_n={100: -4.0, 500: -4.8, 1500: -5.7},
            sigma_gen_per_n_d={3: {1500: -6.0}},
            # iter caps at n=1500 kept MODEST: the dense Gibbs path (required —
            # see adaptive_stopping below) is ~6× slower than the CSR path at
            # n=1500, so 5M+ iters made d≥2 cells take >20 min each. d=1 mixed
            # fine at ≤1 sweep, so cap d=2/d=3 at ~1 sweep (n·(n−1)/2 ≈ 1.12M)
            # to keep the column tractable. NOTE: these n=1500 d≥2 caps are not
            # yet verified to preserve recovery — confirm with a single-cell
            # test (as done for d=1) before trusting the d=2/d=3 n=1500 results.
            iter_cap_by_d={
                2: {500: 300_000, 1500: 800_000},
                3: {500: 500_000, 1500: 1_200_000},
            },
            m_ensemble_by_d=None,
            # Penalty=4.0 sets the LL-gap threshold (d_true needs gap >
            # 2·penalty above d=0). At n=100 the gap is borderline → ~50-70%
            # recovery; by n=1500 the gap clears it → ~100%. This is the knob
            # to lower if a d never recovers, or raise if small-n is already
            # saturated (which would flatten the increasing trend).
            aic_penalty_per_d=4.0,
            # adaptive_stopping=False: the CSR early-stop path converges to a
            # DIFFERENT (denser, wrong) equilibrium at large n and breaks
            # d-recovery — verified at n=1500 d=1 (adaptive ON → ρ=0.019, d̂=0;
            # adaptive OFF → ρ=0.0035, d̂=1, and faster). Use the dense path.
            adaptive_stopping=False,
            adaptive_check_interval=10_000,
            adaptive_patience=3,
            adaptive_cv_tol=0.02,
            adaptive_min_iter=5_000,
        ),
    },
    # PAPER_ROC_SMOKE: fast probe (~30s) for iterating on curve shape.
    # Same sigma/d grid as PAPER_ROC but with tiny n_effect and n_values
    # so we can quickly see whether the ROC curves match the paper's
    # gradient (range from near-diagonal to high power) vs being saturated.
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
    # PAPER_ROC: reproduce paper Fig 3 (effect size at n=500) + Fig 4 (sample
    # size at σ₂=-1.5). Tuned for ~3 min on 4 cores.
    # - n_reps=3: low ANOVA power so curves aren't step-functions
    # - subsample_pairs=0.005: σ̂ from a random 0.5% of pair-indices → SE ∝ 1/n
    #   so the test is not saturated at small |Δ| and power increases with n
    #   (matches paper Fig 3 gradient + Fig 4 monotonicity)
    # - adaptive stopping cuts at ~1k iter once edge count stabilizes
    # - Bottleneck: d=2 n=2000 ≈ 100s/cell (cell-parallel can't split a cell)
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
    # PAPER_SIGMA_CONVERGENCE: σ̂ convergence to true σ for all (d, σ, n).
    # Common n grid so every σ has the same x-axis points, including small n
    # where very-negative σ produces near-empty graphs and σ̂ diverges.
    # No iter_cap; runtime ~3-4 min on 4 cores.
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
