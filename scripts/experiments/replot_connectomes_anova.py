#!/usr/bin/env python3
"""Replot the Connectomes LG ANOVA figure from saved data (no rerun): reads
runs/connectomes_tlg_anova_robust/{summary,pairwise_sigma,pairwise_alpha}.csv with LG labels,
"16 networks", and sign convention sigma<=0 / alpha>=0; Bonferroni heatmaps verbatim from CSV."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent / "runs" / "connectomes_tlg_anova_robust"
PARAMS = ("sigma", "alpha")
PLABEL = {"sigma": r"$\hat{\sigma}$ (intercept)", "alpha": r"$\hat{\alpha}$ (degree slope)"}
SIGN = {"sigma": -1.0, "alpha": +1.0}    # sigma forced negative, alpha forced positive


def _heatmap(ax, fig, labels, pdf, title):
    k = len(labels)
    idx = {r: i for i, r in enumerate(labels)}
    P = np.ones((k, k))
    for _, r in pdf.iterrows():
        i, j = idx[r["region_i"]], idx[r["region_j"]]
        P[i, j] = P[j, i] = r["p_bonf"]
    Mlog = -np.log10(np.clip(P, 1e-300, 1.0))
    np.fill_diagonal(Mlog, np.nan)
    im = ax.imshow(Mlog, cmap="viridis", vmin=0, vmax=min(80, np.nanmax(Mlog)))
    ax.set_xticks(range(k)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$-\log_{10} p_{\mathrm{bonf}}$ ($>1.3\Rightarrow$ sig.@0.05)")


def main():
    df = pd.read_csv(RUN / "summary.csv")
    labels = list(df["label"])
    k = len(labels)
    pdf = {"sigma": pd.read_csv(RUN / "pairwise_sigma.csv"),
           "alpha": pd.read_csv(RUN / "pairwise_alpha.csv")}

    fig, axes = plt.subplots(2, 2, figsize=(max(13, 0.5 * k + 6), 13))
    for ri, par in enumerate(PARAMS):
        val = SIGN[par] * df[f"{par}_hat"].abs().to_numpy()
        se = df[f"se_{par}_robust"].to_numpy()
        order = np.argsort(val)
        v, s = val[order], se[order]
        disp = [labels[i] for i in order]
        y = np.arange(len(order))
        ax0 = axes[ri][0]
        ax0.errorbar(v, y, xerr=s, fmt="o", color="#1f77b4", ecolor="#1f77b4",
                     elinewidth=1.4, capsize=3, ms=4)
        ax0.set_yticks(y); ax0.set_yticklabels(disp, fontsize=7)
        ax0.set_xlabel(PLABEL[par]); ax0.set_title(f"Per-connectome {par}")
        ax0.grid(axis="x", ls=":", alpha=0.4)
        _heatmap(axes[ri][1], fig, labels, pdf[par],
                 rf"{par}: pairwise $-\log_{{10}}$(Bonferroni $p$)")
    fig.suptitle(f"Connectomes LG ANOVA: sigma and alpha across {k} networks", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = RUN / "connectomes_tlg_anova_robust.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
