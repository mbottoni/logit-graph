"""Paper experiment sweeps (INSIGHT / SMOKE / DEV / PAPER presets)."""
from .presets import PRESETS, AICSweepConfig, SigmaSweepConfig
from .sweeps import (
    plot_aic_confusion,
    plot_convergence_sigma,
    run_aic_d_sweep,
    run_sigma_sweep,
    run_sigma_estimator_ablation,
    flag_sigma_sweep_issues,
    summarize_aic_insights,
    summarize_sigma_insights,
    simulate_graph,
    estimate_sigma_from_graph,
)

__all__ = [
    "PRESETS",
    "AICSweepConfig",
    "SigmaSweepConfig",
    "run_sigma_sweep",
    "run_aic_d_sweep",
    "run_sigma_estimator_ablation",
    "flag_sigma_sweep_issues",
    "summarize_aic_insights",
    "summarize_sigma_insights",
    "plot_convergence_sigma",
    "plot_aic_confusion",
    "simulate_graph",
    "estimate_sigma_from_graph",
]
