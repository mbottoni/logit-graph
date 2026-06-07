#!/usr/bin/env python3
"""Run the latent-TLG GIC experiment across ALL datasets and write a combined ranking.

Convenience orchestrator over the per-dataset entry points (run_tlg_<dataset>_latent_gic.py):
runs every dataset in tlg_latent_gic_common.DATASETS (or LG_TLM_DATASETS) and writes, in
addition to each dataset's table/plot, a cross-dataset KL-rank summary and a TLG-vs-SBM
head-to-head under runs/tlg_latent_gic/.

  make tlg-all-latent-gic        full run (all datasets + summary)
  make tlg-all-latent-gic-quick  smoke
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlg_latent_gic_common as C  # noqa: E402

if __name__ == "__main__":
    C.run_all()
