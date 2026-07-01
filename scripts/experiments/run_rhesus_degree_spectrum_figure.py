#!/usr/bin/env python3
"""Combined degree-distribution + spectral-density figure for the connectome case study.

For the observed connectome and each model's best fit, this plots two aligned views per network:
the degree distribution (top row) and the normalized-Laplacian eigenvalue density (bottom row),
with the columns ordered by KL spectral rank. It reuses run_rhesus_case_study_figure.compute() so
the fitted graphs (and their KL divergences) are exactly those of the case-study/spectrum figures.

Output: <OUT>/rhesus_degree_spectrum.png/.pdf, where OUT is the case-study run directory.
Set LG_CASE_NET to another connectome id to regenerate for it (default: rhesus_brain_1).
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

DEG_BAR = "#e6975b"   # degree distribution (warm)
SPEC_BAR = "#5b9bd5"  # spectral density (cool), matching run_rhesus_spectrum_figure


def _laplacian_eigs(G):
    """Real eigenvalues of the normalized Laplacian (in [0, 2]), as in the spectrum figure."""
    L = nx.normalized_laplacian_matrix(G)
    return np.real(np.linalg.eigvals(L.toarray()))


def _degrees(G):
    return np.asarray([d for _, d in G.degree()], dtype=float)


def main():
    G, results = CS.compute()
    ranked = sorted(results, key=lambda f: results[f]["kl"])       # best (lowest KL) first
    panels = [("Original", G, None)] + [(m, results[m]["graph"], results[m]["kl"]) for m in ranked]

    ncols = len(panels)
    # Shared degree axis so the columns are visually comparable.
    kmax = max(_degrees(g).max() for _, g, _ in panels)

    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 8))
    for col, (name, graph, kl) in enumerate(panels):
        ax_deg, ax_spec = axes[0, col], axes[1, col]

        # Top: degree distribution.
        deg = _degrees(graph)
        sns.histplot(x=deg, kde=True, ax=ax_deg, stat="density", bins=30,
                     color=DEG_BAR, edgecolor="white", linewidth=0.3)
        title = name if kl is None else f"{name}\nKL $= {kl:.3f}$"
        ax_deg.set_title(title, fontsize=13, fontweight="bold")
        ax_deg.set_xlabel("Degree", fontsize=11)
        ax_deg.set_ylabel("Density" if col == 0 else "", fontsize=11)
        ax_deg.set_xlim(0.0, kmax)

        # Bottom: normalized-Laplacian spectral density.
        eigs = _laplacian_eigs(graph)
        sns.histplot(x=eigs, kde=True, ax=ax_spec, stat="density", bins=40,
                     binrange=(0.0, 2.0), color=SPEC_BAR, edgecolor="white", linewidth=0.3)
        ax_spec.set_xlabel("Eigenvalue", fontsize=11)
        ax_spec.set_ylabel("Density" if col == 0 else "", fontsize=11)
        ax_spec.set_xlim(0.0, 2.0)

    axes[0, 0].annotate("Degree distribution", xy=(-0.42, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")
    axes[1, 0].annotate("Spectral density", xy=(-0.42, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", ha="center", fontsize=13, fontweight="bold")

    fig.suptitle(f"{CS._net_display(CS.NET_ID)}: degree distribution and normalized-Laplacian "
                 "spectral density — observed connectome vs. each model's best fit "
                 "(columns ordered by KL rank)", fontsize=15, y=1.02)
    fig.tight_layout()
    CS.OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(CS.OUT / f"rhesus_degree_spectrum.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {CS.OUT}/rhesus_degree_spectrum.png/.pdf  (columns: "
          f"{', '.join(n for n, _, _ in panels)})")


if __name__ == "__main__":
    main()
