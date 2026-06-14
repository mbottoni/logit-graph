#!/usr/bin/env python3
"""Convergence-dynamics figure for the connectome case study (thesis rhesus_iteration.png).

Companion to run_rhesus_case_study_figure.py. Reproduces the three-panel convergence plot of the
LG spectral-fitting process — Edge Differences, Spectrum Differences, and GIC Values vs. iteration,
each shown raw + a 10-point moving average — using the repo's own LG fitter
(logit_graph.LogitGraphFitter), exactly the object whose metadata the thesis notebook
(notebooks/connectomes_datasets/17-1-connectomes-analysis.ipynb) plotted: the traces are
metadata['edge_diffs'], ['spectrum_diffs'], ['gic_values'].

The fit is the iterative add/remove spectral-minimization process, which only exists for a
neighborhood radius d >= 1 (the d = 0 path generates a graph in closed form, with no iteration).
The default network is therefore the rhesus cerebral cortex (d = 1), the same connectome used for
the thesis convergence figure; rhesus_brain_1 (the LG-best case-study default) has d = 0 and is
skipped with a clear message. Set LG_CASE_NET to any d >= 1 connectome id to regenerate for it.

Output under scripts/experiments/runs/rhesus_case_study/ (gitignored):
  rhesus_convergence.png / .pdf

Run:  .venv/bin/python scripts/experiments/run_rhesus_convergence_figure.py
Env:  LG_CASE_NET (rhesus_cerebral.cortex_1), LG_CONV_ITERS (6000), LG_CONV_CHECK (20),
      LG_CONV_WARMUP (500).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
_repo = _here.parents[1]
for p in (_repo / "src", _repo / "scripts" / "closedform"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tlg_latent_gic_common as C   # noqa: E402  (connectome loader, cached d selection)
from logit_graph import LogitGraphFitter  # noqa: E402

NET_ID = os.environ.get("LG_CASE_NET", "rhesus_cerebral.cortex_1")
OUT = _here / "runs" / "rhesus_case_study"
ITERS = int(os.environ.get("LG_CONV_ITERS", "6000"))
CHECK = int(os.environ.get("LG_CONV_CHECK", "20"))
WARMUP = int(os.environ.get("LG_CONV_WARMUP", "500"))
SEED = 12345

DISPLAY = {"rhesus_brain_1": "Rhesus macaque brain",
           "rhesus_cerebral.cortex_1": "Rhesus macaque cerebral cortex"}


def _net_display(nid):
    return DISPLAY.get(nid, nid.replace("_", " "))


def _cached_d(nid):
    cpath = C._out_dir("connectome") / "cache" / f"{nid}.json"
    if cpath.is_file():
        return int(json.loads(cpath.read_text())["tlg_selected"]["d"])
    return 1


def main():
    np.random.seed(SEED)
    G = C._graphml(C.DATA / "connectomes" / f"{NET_ID}.graphml")
    n, m = G.number_of_nodes(), G.number_of_edges()
    d = _cached_d(NET_ID)
    C.log(f"{_net_display(NET_ID)} ({NET_ID}): n={n} m={m}  cached d={d}")
    if d == 0:
        C.log("  d=0 selected for this network -> the LG fit is closed-form (direct ER), so there "
              "is no iterative convergence trace.\n  Pick a d>=1 connectome via LG_CASE_NET "
              "(default rhesus_cerebral.cortex_1).")
        return

    # The iterative LG spectral-fitting process. min_gic_threshold is set low and patience high so
    # the fit keeps iterating (recording the full trace) rather than stopping at the GIC gate.
    fitter = LogitGraphFitter(d=d, n_iteration=ITERS, warm_up=WARMUP, patience=10**9,
                              dist_type="KL", min_gic_threshold=0.0, check_interval=CHECK,
                              verbose=False)
    fitter.fit(G)
    md = fitter.metadata
    edge_diff = md.get("edge_diffs") or []
    spec_diff = md.get("spectrum_diffs") or []
    gic_values = md.get("gic_values") or []
    C.log(f"  trace points: edge={len(edge_diff)} spectrum={len(spec_diff)} gic={len(gic_values)}; "
          f"best_iteration={md.get('best_iteration')}")
    if len(spec_diff) < 3:
        C.log("  too few trace points to plot — increase LG_CONV_ITERS."); return

    window = 10
    series = [("Edge Differences", "Difference", edge_diff),
              ("Spectrum Differences", "Difference", spec_diff),
              ("GIC Values", "GIC", gic_values)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, ylab, raw) in zip(axes, series):
        x = np.arange(len(raw)) * (1 if title.startswith("Edge") else CHECK)
        ma = pd.Series(raw).rolling(window=window).mean()
        ax.plot(x, raw, color="#9ecae1", alpha=0.6, lw=1.2, label="Raw")
        ax.plot(x, ma, "r-", lw=2.0, label=f"{window}-point Moving Avg")
        ax.set_title(title, fontsize=15)
        ax.set_xlabel("Iteration", fontsize=13); ax.set_ylabel(ylab, fontsize=13)
        ax.legend(fontsize=11)
    fig.suptitle(f"{_net_display(NET_ID)}: convergence of the LG spectral-fitting process "
                 f"(d={d})", fontsize=15, y=1.03)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rhesus_convergence.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    C.log(f"Wrote {OUT}/rhesus_convergence.png/.pdf")


if __name__ == "__main__":
    main()
