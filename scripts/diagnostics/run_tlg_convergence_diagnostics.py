#!/usr/bin/env python3
"""Convergence diagnostics for the Temporal Logit-Graph (TLG) with edge removal.

The TLG analog of notebooks/base/22-3-convergence-diagnostics.ipynb (the LG MCMC
version). With ``allow_removal=True`` each step resamples EVERY dyad from the lagged
probability expit(sigma + alpha*D(t-1)), so edges form AND dissolve — the process is
a genuine ergodic Markov chain with a stationary distribution at moderate density
(no saturation). This experiment shows the chains **mix** to that stationary
distribution independently of where they start.

  1. Build a long-run reference graph: one draw from the stationary distribution
     (grown with removal well past mixing).
  2. Grow ``len(ER_PS)`` chains from ER seeds at varied densities p_0; at every
     checkpoint record three diagnostics vs the reference:
       (a) Laplacian spectral distance        -> noise floor
       (b) KS statistic of the degree dist.    -> noise floor
       (c) adjacency-ESD KL divergence         -> noise floor
     Diagnostics plateau at the (small, non-zero) distance between two independent
     draws from the stationary distribution, not at exactly 0 — the correct MCMC
     interpretation of convergence to a *distribution*.
  3. Plot the three panels.

Model: logit(P[edge i-j at step t]) = sigma + alpha * D_ij(t-1), with D the
"bounded" degree feature (see logit_graph.temporal). x-axis = growth STEP (each
step is one full sweep / resample over all dyads), not single edge flips.

Env knobs (all optional):
  LG_TLGC_N (750)      LG_TLGC_D (0)       LG_TLGC_SIGMA (-2.0)   LG_TLGC_ALPHA (0.05)
  LG_TLGC_STEPS (20)   growth steps per chain
  LG_TLGC_CHECK (1)    checkpoint every k steps
  LG_TLGC_SEED (42)    LG_TLGC_QUICK (1 -> n=200, fewer steps)

  make tlg-convergence-diagnostics         full run
  make tlg-convergence-diagnostics-quick   smoke
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp

_repo_root = Path(__file__).resolve().parents[2]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from logit_graph import graph as lg_graph  # noqa: E402  (calculate_spectrum)
from logit_graph.temporal import grow_graph  # noqa: E402

ER_PS = (0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.075)
OUT_DIR = Path(__file__).resolve().parent / "runs" / "tlg_convergence"


def _int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _float(name, default):
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _adjacency_esd(adj, bin_edges):
    eig = np.linalg.eigvalsh(adj)
    eig = np.clip(eig, bin_edges[0], bin_edges[-1])
    hist, _ = np.histogram(eig, bins=bin_edges, density=True)
    return hist


def _esd_kl(cur_den, ref_den):
    return float(entropy(cur_den + 1e-10, ref_den + 1e-10))


def _diagnostics(adj, ref_spec, ref_deg, ref_esd, esd_bins):
    spec = lg_graph.GraphModel.calculate_spectrum(adj)
    ks, _ = ks_2samp(adj.sum(axis=1), ref_deg)
    return {
        "spec_dist": float(np.linalg.norm(spec - ref_spec)),
        "edges": int(adj.sum() / 2),
        "ks": float(ks),
        "esd_kl": _esd_kl(_adjacency_esd(adj, esd_bins), ref_esd),
    }


def _grow_snapshots(n, d, sigma, alpha, p0, n_steps, seed):
    res = grow_graph(n, d=d, sigma=sigma, alpha=alpha, n_steps=n_steps,
                     seed=seed, p0=p0, store_snapshots=True, allow_removal=True)
    return res.snapshots


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(df, *, n, d, sigma, alpha, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.suptitle(f"TLG convergence (edge add+remove): ergodic mixing to the "
                 f"stationary distribution — $n={n}$, $d={d}$, "
                 f"$\\sigma={sigma:g}$, $\\alpha={alpha:g}$ — "
                 f"{len(p0s)} chains from different initial densities", y=1.03)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "convergence_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'convergence_diagnostics.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    quick = os.environ.get("LG_TLGC_QUICK", "0") == "1"
    n = _int("LG_TLGC_N", 200 if quick else 750)
    d = _int("LG_TLGC_D", 0)
    # sigma=-2, alpha=0.05 is a mild regime: the add+remove chain mixes to a single
    # moderate-density stationary distribution (no ERGM-style bistability), so all
    # chains converge to the common reference within the plotted window.
    sigma = _float("LG_TLGC_SIGMA", -2.0)
    alpha = _float("LG_TLGC_ALPHA", 0.05)
    steps = _int("LG_TLGC_STEPS", 12 if quick else 20)
    check = _int("LG_TLGC_CHECK", 1)
    seed = _int("LG_TLGC_SEED", 42)
    print(f"TLG convergence  mode={'QUICK' if quick else 'FULL'}  n={n} d={d} "
          f"sigma={sigma} alpha={alpha} steps={steps} check={check}")

    # Reference: one draw from the stationary distribution — grow with removal well
    # past mixing — the common limit the chains converge to (up to the noise floor
    # between independent stationary draws).
    print("Building long-run reference (one stationary draw) ...")
    ref_snaps = _grow_snapshots(n, d, sigma, alpha, p0=0.05,
                                n_steps=int(1.5 * steps), seed=seed)
    ref = ref_snaps[-1]
    ref_spec = lg_graph.GraphModel.calculate_spectrum(ref)
    ref_deg = ref.sum(axis=1)
    eig = np.linalg.eigvalsh(ref)
    lo, hi = float(eig.min()), float(eig.max())
    pad = 0.05 * (hi - lo) + 1e-9
    esd_bins = np.linspace(lo - pad, hi + pad, 51)
    ref_esd = _adjacency_esd(ref, esd_bins)
    ref_edges = int(ref.sum() / 2)
    print(f"Reference: {ref_edges} edges, density "
          f"{ref_edges / (n * (n - 1) / 2):.4f}, mean degree {ref_deg.mean():.1f}")

    rows = []
    for ci, p0 in enumerate(ER_PS):
        snaps = _grow_snapshots(n, d, sigma, alpha, p0=p0, n_steps=steps,
                                seed=seed + 1 + ci)
        for t in range(0, len(snaps), check):
            diag = _diagnostics(snaps[t], ref_spec, ref_deg, ref_esd, esd_bins)
            rows.append({"p0": p0, "step": t, **diag})
        last = rows[-1]
        print(f"  chain p0={p0:<6g}: final spec={last['spec_dist']:.2f} "
              f"edges={last['edges']} KS={last['ks']:.3f} ESD_KL={last['esd_kl']:.4f}")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "convergence.csv", index=False)
    _plot(df, n=n, d=d, sigma=sigma, alpha=alpha, out_dir=OUT_DIR)
    print(f"Wrote {OUT_DIR}/ (convergence.csv, convergence_diagnostics.png)")


if __name__ == "__main__":
    main()
