"""Logit Graph: paper-consistent random-graph model with Layer-2 estimation.

The default exports here are all paper-consistent (Layer-2 conditioning,
``feature_mode="incremental"``, ``beta=1`` offset logit, d=0 direct ER
sampling, warm-started Gibbs for sparse regimes). Legacy classes
(:class:`MLEGraphModelEstimator`, :class:`NegativeLogLikelihoodLoss`) are
deprecated but kept importable for reproducibility of the original notebooks.
"""

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

# Re-export the corrected sampler / estimator helpers so users get the
# paper-consistent path directly from the top level:
#
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
]
