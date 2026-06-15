#!/usr/bin/env python3
"""Latent-TLG GIC sweep over the Facebook SNAP ego networks (all): fit every qualifying network with the unified TLG (degree + community +
latent ASE) and rank vs ER/BA/WS/KR/GRG/SBM by raw KL (ensemble-mean). Output
runs/tlg_latent_gic/sweep/facebook_{per_graph,summary}.csv; `make tlg-facebook-latent-gic`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlg_latent_gic_common as C  # noqa: E402

if __name__ == "__main__":
    C.run_sweep("facebook")
