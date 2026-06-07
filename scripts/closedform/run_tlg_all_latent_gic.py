#!/usr/bin/env python3
"""Run the latent-TLG GIC sweep across ALL datasets and print the overall cross-dataset
family KL ranking.

Every network of every dataset (with the same per-dataset selection the individual
run_tlg_<dataset>_latent_gic.py scripts use) is fit in ONE global parallel pool, so work is
parallelized BOTH across datasets and across networks within a dataset, with load balancing
(human has ~975 graphs, twitch 6). Each finished network is cached, so the run is resumable
and reproducible (fixed seeds). Results are written per dataset under
runs/tlg_latent_<dataset>_gic/ and the overall ranking under runs/tlg_latent_overall_gic/.

Select a subset with LG_SWEEP_DATASETS=twitter,gplus,...; tune with LG_SWEEP_WORKERS,
LG_SWEEP_NMIN/NMAX (twitter/gplus band), LG_SWEEP_HUMAN_SCALE, LG_SWEEP_ARXIV_CAP, LG_TLM_*.

  make tlg-all-latent-gic
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlg_latent_gic_common as C  # noqa: E402

if __name__ == "__main__":
    sel = os.environ.get("LG_SWEEP_DATASETS")
    datasets = sel.split(",") if sel else C.SWEEP_DATASETS
    C.run_sweep_multi(datasets)
