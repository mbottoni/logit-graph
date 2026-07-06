#!/usr/bin/env python3
"""Run the additional-baselines experiment over one or more datasets.

Usage:
    python scripts/more_baselines/run_more_baselines.py connectome
    python scripts/more_baselines/run_more_baselines.py connectome twitch
    python scripts/more_baselines/run_more_baselines.py all

Datasets: twitch, facebook, twitter, gplus, connectome, human, arxiv (same names/loaders as the
main latent-TLG sweep). Results are cached per network (resumable) under
scripts/more_baselines/runs/more_baselines_<dataset>/, with per_graph.csv + summary.csv written at
the end. Only the new baselines are fitted; the existing families are untouched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import more_baselines_common as MB  # noqa: E402

ALL = ["twitch", "facebook", "twitter", "gplus", "connectome", "human", "arxiv"]


def main():
    args = sys.argv[1:] or ["connectome"]
    datasets = ALL if args == ["all"] else args
    for ds in datasets:
        MB.run_dataset(ds)


if __name__ == "__main__":
    main()
