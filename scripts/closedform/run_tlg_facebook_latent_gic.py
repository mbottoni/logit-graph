#!/usr/bin/env python3
"""Latent-TLG GIC sweep over the Facebook SNAP ego networks (all).

Fits EVERY qualifying network of the facebook dataset with the unified, identifiable TLG
(degree + coarse/fine Louvain community + latent adjacency-spectral-embedding feature) and
ranks it against ER/BA/WS/KR/GRG/SBM by raw KL (fair ensemble-mean estimator). Runs in
parallel across LG_SWEEP_WORKERS processes, caches each finished network under
runs/tlg_latent_gic/sweep_cache/facebook/ (rerun resumes; only unfinished networks are fit),
and streams the full family KL ranking per network. See tlg_latent_gic_common.run_sweep.

Output: runs/tlg_latent_gic/sweep/facebook_{per_graph,summary}.csv. Env: LG_SWEEP_WORKERS,
LG_SWEEP_NMIN/NMAX (ego filters), LG_SWEEP_HUMAN_SCALE, LG_SWEEP_ARXIV_CAP, + LG_TLM_*.

  make tlg-facebook-latent-gic
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlg_latent_gic_common as C  # noqa: E402

if __name__ == "__main__":
    C.run_sweep("facebook")
