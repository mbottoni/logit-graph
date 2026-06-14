#!/usr/bin/env python3
"""Replot the Twitch LG ANOVA figure from the saved data (no refit / no simulation rerun).

Reads runs/twitch_tlg_anova_robust/{summary,pairwise_sigma,pairwise_alpha}.csv and regenerates
twitch_tlg_anova_robust.png with the presentation changes requested for the paper:
  * LG instead of TLG, and no "dyadic-cluster-robust SE" qualifier in the title;
  * the per-region forest plots show a SINGLE error bar (no naive-SE overlay), matching the
    connectomes figure, and the subplot titles drop the "(... SE)" note;
  * the display sign convention sigma <= 0 and alpha >= 0, keeping the raw magnitude
    (sigma -> -|sigma|, alpha -> +|alpha|). For Twitch both are already so-signed, so the
    magnitudes are identical; the convention is applied for consistency with the connectomes.
The pairwise Bonferroni heatmaps are taken verbatim from the saved CSVs (the equality test is
invariant to the display sign convention).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(__file__).resolve().parent / "runs" / "twitch_tlg_anova_robust"
PARAMS = ("sigma", "alpha")
PLABEL = {"sigma": r"$\hat{\sigma}$ (intercept)", "alpha": r"$\hat{\alpha}$ (degree slope)"}
SIGN = {"sigma": -1.0, "alpha": +1.0}    # sigma forced negative, alpha forced positive


def _heatmap(ax, labels, pdf, title):
    k = len(labels)
    idx = {r: i for i, r in enumerate(labels)}
    P = np.ones((k, k))
    for _, r in pdf.iterrows():
        i, j = idx[r["region_i"]], idx[r["region_j"]]
        P[i, j] = P[j, i] = r["p_bonf"]
    Mlog = -np.log10(np.clip(P, 1e-300, 1.0))
    np.fill_diagonal(Mlog, np.nan)
    im = ax.imshow(Mlog, cmap="viridis", vmin=0, vmax=min(80, np.nanmax(Mlog)))
    ax.set_xticks(range(k)); ax.set_xticklabels(labels)
    ax.set_yticks(range(k)); ax.set_yticklabels(labels)
    ax.set_title(title)
    return im


def main():
    df = pd.read_csv(RUN / "summary.csv")
    labels = list(df["display"])
    pdf = {"sigma": pd.read_csv(RUN / "pairwise_sigma.csv"),
           "alpha": pd.read_csv(RUN / "pairwise_alpha.csv")}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ri, par in enumerate(PARAMS):
        val = SIGN[par] * df[f"{par}_hat"].abs().to_numpy()
        se = df[f"se_{par}_robust"].to_numpy()
        order = np.argsort(val)
        v, s = val[order], se[order]
        disp = [labels[i] for i in order]
        y = np.arange(len(order))
        ax0 = axes[ri][0]
        ax0.errorbar(v, y, xerr=s, fmt="o", color="#1f77b4", ecolor="#1f77b4",
                     elinewidth=1.6, capsize=4, ms=5)
        ax0.set_yticks(y); ax0.set_yticklabels(disp)
        ax0.set_xlabel(PLABEL[par]); ax0.set_title(f"Per-region {par}")
        ax0.grid(axis="x", ls=":", alpha=0.4)
        im = _heatmap(axes[ri][1], labels, pdf[par],
                      rf"{par}: pairwise $-\log_{{10}}$(Bonferroni $p$)")
        cbar = fig.colorbar(im, ax=axes[ri][1], fraction=0.046, pad=0.04)
        cbar.set_label(r"$-\log_{10} p_{\mathrm{bonf}}$ ($>1.3\Rightarrow$ sig.@0.05)")
    fig.suptitle("Twitch LG ANOVA: sigma and alpha across communities", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = RUN / "twitch_tlg_anova_robust.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
