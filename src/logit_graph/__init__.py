"""Logit Graph: paper-consistent random-graph model with Layer-2 estimation.
Default exports are paper-consistent (Layer-2, incremental feature, beta=1 offset logit,
d=0 direct-ER, warm-started Gibbs); legacy estimator classes stay importable for repro."""

from .graph import GraphModel
from .lg_features import (
    FeatureMode,
    build_pair_dataset,
    incremental_h,
    pair_feature,
    pair_feature_layer2,
    recommended_iterations,
)
from .logit_estimator import LogitRegEstimator
from .simulation import (
    GraphModelComparator,
    LogitGraphFitter,
    LogitGraphSimulation,
    calculate_graph_attributes,
    estimate_sigma_many,
    estimate_sigma_only,
)

# Re-export the paper-consistent sampler / estimator helpers at the top level, e.g.
#   from logit_graph import simulate_graph, estimate_sigma_from_graph
from .experiments.sweeps import (
    estimate_sigma_from_graph,
    select_d_ensemble,
    simulate_graph,
)
from .experiments.presets import (
    AICSweepConfig,
    PRESETS,
    SigmaSweepConfig,
)
# Temporal / growth Logit-Graph (additive; the equilibrium model is unchanged).
from .temporal import (
    GrowthResult,
    fit_growth_from_result,
    fit_growth_params,
    grow_graph,
    growth_design_from_snapshots,
)
# Multi-feature (unified) temporal LG: degree + arbitrary fixed exogenous dyad covariates.
from .temporal_multi import (
    MultiGrowthResult,
    community_feature,
    fit_multi_params,
    grow_graph_multi,
    latent_feature,
    multi_design_from_snapshots,
)

__all__ = [
    # Core model
    "GraphModel",
    "LogitRegEstimator",
    # Features
    "FeatureMode",
    "build_pair_dataset",
    "incremental_h",
    "pair_feature",
    "pair_feature_layer2",
    "recommended_iterations",
    # High-level API (sklearn-style)
    "LogitGraphFitter",
    "LogitGraphSimulation",
    "GraphModelComparator",
    "calculate_graph_attributes",
    "estimate_sigma_only",
    "estimate_sigma_many",
    # Paper-consistent sampler / estimator (canonical entry points)
    "simulate_graph",
    "estimate_sigma_from_graph",
    "select_d_ensemble",
    # Experiment configs
    "AICSweepConfig",
    "SigmaSweepConfig",
    "PRESETS",
    # Temporal / growth Logit-Graph (generation + estimation)
    "grow_graph",
    "GrowthResult",
    "growth_design_from_snapshots",
    "fit_growth_params",
    "fit_growth_from_result",
    # Multi-feature (unified) temporal Logit-Graph
    "grow_graph_multi",
    "MultiGrowthResult",
    "multi_design_from_snapshots",
    "fit_multi_params",
    "community_feature",
    "latent_feature",
]
