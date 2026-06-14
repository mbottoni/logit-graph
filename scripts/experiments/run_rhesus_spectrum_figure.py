#!/usr/bin/env python3
"""Spectral-density figure for the connectome case study (thesis rhesus_spectrum.png).

Companion to run_rhesus_case_study_figure.py. Reproduces the normalized-Laplacian eigenvalue
spectral density (histogram + KDE) of the observed connectome and the best-fit graph from each
model — the spectra whose KL divergence the case-study ranking is built on.

It reuses the case-study script's graph generation (run_rhesus_case_study_figure.compute), so the
panels show exactly the same best-fit instances the KL ranking uses, with the same network default
(LG_CASE_NET, default rhesus_brain_1 where LG is the best fit). The eigenvalue spectrum and the
seaborn histogram+KDE plot match the thesis notebook
(notebooks/connectomes_datasets/17-1-connectomes-analysis.ipynb); the only correction is that all
seven models are shown (the old figure showed only Original/LG/ER/BA/WS, omitting GRG/KR/SBM), and
panels are ordered by KL rank so the closest-matching spectra come first.

Output under scripts/experiments/runs/rhesus_case_study/ (gitignored):
  rhesus_spectrum.png / .pdf

Run:  .venv/bin/python scripts/experiments/run_rhesus_spectrum_figure.py
Env:  LG_CASE_NET (rhesus_brain_1)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

_here = Path(__file__).resolve().parent
_repo = _here.parents[1]
for p in (_repo / "src", _repo / "scripts" / "closedform", _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_rhesus_case_study_figure as CS  # noqa: E402  (compute(), NET_ID, _net_display, OUT)

BAR = "#5b9bd5"


def _laplacian_eigs(G):
    """Real eigenvalues of the normalized Laplacian (in [0, 2]), as in the thesis notebook."""
    L = nx.normalized_laplacian_matrix(G)
    return np.real(np.linalg.eigvals(L.toarray()))


def main():
    G, results = CS.compute()
    ranked = sorted(results, key=lambda f: results[f]["kl"])     # best (lowest KL) first
    panels = [("Original", G)] + [(m, results[m]["graph"]) for m in ranked]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (name, graph) in zip(axes.flat, panels):
        eigs = _laplacian_eigs(graph)
        sns.histplot(x=eigs, kde=True, ax=ax, stat="density", bins=40,
                     binrange=(0.0, 2.0), color=BAR, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{name} Spectrum", fontsize=15)
        ax.set_xlabel("Eigenvalue", fontsize=13)
        ax.set_ylabel("Density", fontsize=13)
        ax.set_xlim(0.0, 2.0)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(f"{CS._net_display(CS.NET_ID)}: normalized-Laplacian spectral density — observed "
                 "connectome vs. each model's best fit (panels ordered by KL rank)",
                 fontsize=16, y=1.01)
    fig.tight_layout()
    CS.OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(CS.OUT / f"rhesus_spectrum.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {CS.OUT}/rhesus_spectrum.png/.pdf  (panels: "
          f"{', '.join(n for n, _ in panels)})")


if __name__ == "__main__":
    main()
