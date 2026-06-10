"""Re-plot the TLG convergence diagnostics from the cached CSV (no experiment rerun).

Reads runs/tlg_convergence/convergence.csv and regenerates the 3-panel figure with an
updated, cleaner title, writing straight to the paper's image folder.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV = Path(__file__).parent / "runs" / "tlg_convergence" / "convergence.csv"
OUT = Path(sys.argv[1])  # destination PNG path

# Simulation parameters (from the cached run; not re-estimated here).
N, D, SIGMA, ALPHA = 200, 0, -2, 0.05

df = pd.read_csv(CSV)
cmap = plt.cm.viridis
markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
p0s = sorted(df["p0"].unique())
colors = {p: cmap(i / max(1, len(p0s) - 1)) for i, p in enumerate(p0s)}

panels = [("spec_dist", "Spectral distance to reference", "(a) Laplacian spectrum"),
          ("ks", "KS statistic (degree dist.)", "(b) Degree distribution"),
          ("esd_kl", r"$D_{\mathrm{KL}}(\rho_t \,\|\, \rho_{\mathrm{ref}})$",
           "(c) Adjacency ESD divergence")]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (key, ylabel, title) in zip(axes, panels):
    for i, p in enumerate(p0s):
        sub = df[df["p0"] == p].sort_values("step")
        ax.plot(sub["step"], sub[key], color=colors[p],
                marker=markers[i % len(markers)], markevery=max(1, len(sub) // 8),
                ms=5, alpha=0.85, label=f"$p_0={p:g}$")
    ax.set_xlabel("growth step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title="Initial ER $p_0$", title_fontsize=8)

fig.suptitle(
    f"MCMC convergence: $n={N}$, $d={D}$, $\\sigma={SIGMA:g}$, $\\alpha={ALPHA:g}$; "
    f"{len(p0s)} chains from different initial densities", y=1.03)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT}")
