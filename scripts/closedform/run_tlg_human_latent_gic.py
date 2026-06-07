#!/usr/bin/env python3
"""Latent-TLG GIC experiment on the human connectome (OASIS3 brain, graphml) (one representative graph).

Fits the unified, identifiable TLG (degree + coarse/fine Louvain community + latent
adjacency-spectral-embedding feature) and ranks it against ER/BA/WS/KR/GRG/SBM by the
spectral GIC / raw KL, using the fair ensemble-mean estimator. Model, optimizations and
estimator live in tlg_latent_gic_common; this entry point only selects the dataset.

Writes runs/tlg_latent_gic/human_table.csv + human_gic_bar.png (gitignored). Env knobs
(shared): LG_TLM_{QUICK,NRUNS,CAP,K,SEARCH,KLIST,KERNELS,FINE_RES,FAST_SPECTRAL,SEED}.

  make tlg-human-latent-gic        full run
  make tlg-human-latent-gic-quick  smoke
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlg_latent_gic_common as C  # noqa: E402

if __name__ == "__main__":
    C.run_one("human", C.DATASETS["human"])
