#!/usr/bin/env python3
"""Run the latent-TLG GIC sweep across ALL datasets in one global parallel pool (resumable,
cached, fixed seeds) and print the overall cross-dataset family KL ranking. Per-dataset output
under runs/tlg_latent_<dataset>_gic/, overall under runs/tlg_latent_overall_gic/; `make tlg-all-latent-gic`."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tlg_latent_gic_common as C  # noqa: E402

if __name__ == "__main__":
    sel = os.environ.get("LG_SWEEP_DATASETS")
    datasets = sel.split(",") if sel else C.SWEEP_DATASETS
    C.run_sweep_multi(datasets)
