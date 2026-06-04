#!/usr/bin/env python3
"""Diagnose LG Gibbs convergence by tracking GIC vs iteration on gplus nets.

For a few representative ego networks (small / mid / large n) we run the LG
Gibbs chain (warm-started, layer-2, incremental) and recompute the *full* GIC
( 2*KL(real, current) + 2 ) every `CHECK` steps. This shows whether the chain
has converged by iteration 2000 (the value used in the comparison run) or is
still improving, and what a GIC-plateau stopping rule would pick.

Read-only w.r.t. the library. Writes plots/CSV under runs/lg_convergence/.

Env: LG_DIAG_BUDGET (12000)  LG_DIAG_CHECK (200)  LG_DIAG_D (1)
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
from scipy.stats import entropy

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
for p in (_src, _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from logit_graph.gic import GraphInformationCriterion  # noqa: E402
from logit_graph.graph import GraphModel  # noqa: E402
from logit_graph.simulation import estimate_sigma_only, _warm_start_er_p  # noqa: E402

BUDGET = int(os.environ.get("LG_DIAG_BUDGET", "12000"))
CHECK = int(os.environ.get("LG_DIAG_CHECK", "200"))
D = int(os.environ.get("LG_DIAG_D", "1"))
SEED = 12345
REF_ITER = 2000  # the cap used in the comparison run


def pick_representative(files, targets=((50, 70), (120, 145), (280, 300))):
    chosen = {}
    for f in files:
        G = nx.read_edgelist(f, create_using=nx.DiGraph).to_undirected()
        n = G.number_of_nodes()
        for lo, hi in targets:
            if (lo, hi) not in chosen and lo <= n <= hi:
                chosen[(lo, hi)] = (f, n)
        if len(chosen) == len(targets):
            break
    return list(chosen.values())


def gic_trajectory(adj, d):
    """Return (iters, gics) tracking 2*KL+2 every CHECK Gibbs steps."""
    n = adj.shape[0]
    real_nx = nx.from_numpy_array(adj)
    scorer = GraphInformationCriterion(real_nx, model="LG", dist="KL")
    real_den = scorer.compute_spectral_density(real_nx)[0]

    sigma, _ = estimate_sigma_only(adj, d=d, feature_mode="incremental")
    warm = _warm_start_er_p(sigma)
    gm = GraphModel(n=n, d=d, sigma=sigma, er_p=warm, layer2=True,
                    feature_mode="incremental", seed=SEED)

    def cur_gic():
        cur_den = scorer.compute_spectral_density(nx.from_numpy_array(gm.graph))[0]
        return 2.0 * float(entropy(real_den + 1e-10, cur_den + 1e-10)) + 2.0

    iters = [0]
    gics = [cur_gic()]
    done = 0
    while done < BUDGET:
        for _ in range(CHECK):
            gm.add_remove_edge()
        done += CHECK
        iters.append(done)
        gics.append(cur_gic())
    return np.array(iters), np.array(gics), float(sigma)


def plateau_iter(iters, gics, tol=0.02):
    """First iteration whose best-so-far is within `tol` of the global min."""
    best = np.minimum.accumulate(gics)
    target = best[-1] + tol
    idx = int(np.argmax(best <= target))
    return int(iters[idx]), float(best[-1])


def main():
    files = sorted((_repo_root / "data" / "misc" / "gplus").glob("*.edges"))
    out = _here / "runs" / "lg_convergence"
    out.mkdir(parents=True, exist_ok=True)
    reps = pick_representative(files)
    print(f"LG convergence diag  d={D}  budget={BUDGET}  check={CHECK}  "
          f"nets={[(f.stem[:8], n) for f, n in reps]}")

    fig, axes = plt.subplots(1, len(reps), figsize=(5 * len(reps), 4), squeeze=False)
    rows = []
    for ax, (f, n) in zip(axes[0], reps):
        adj = nx.to_numpy_array(nx.convert_node_labels_to_integers(
            nx.read_edgelist(f, create_using=nx.DiGraph).to_undirected()))
        t0 = time.perf_counter()
        iters, gics, sigma = gic_trajectory(adj, D)
        conv_it, best_gic = plateau_iter(iters, gics)
        gic_at_ref = float(gics[np.searchsorted(iters, REF_ITER)])
        rows.append(dict(name=f.stem[:10], n=n, sigma=round(sigma, 3),
                         gic_init=round(float(gics[0]), 3),
                         gic_at_2000=round(gic_at_ref, 3),
                         gic_best=round(best_gic, 3),
                         best_iter=conv_it,
                         improve_after_2000=round(gic_at_ref - best_gic, 3)))
        ax.plot(iters, gics, lw=1.4)
        ax.axvline(REF_ITER, color="red", ls="--", lw=1, label="iter=2000 (cap)")
        ax.axhline(best_gic, color="green", ls=":", lw=1, label=f"best={best_gic:.2f}")
        ax.axvline(conv_it, color="green", ls="-", lw=0.8, alpha=0.5)
        ax.set_title(f"{f.stem[:8]}  n={n}  σ={sigma:.2f}")
        ax.set_xlabel("Gibbs iterations")
        ax.set_ylabel("GIC = 2·KL + 2")
        ax.legend(fontsize=8)
        print(f"  n={n:4d}  GIC: init={gics[0]:.3f}  @2000={gic_at_ref:.3f}  "
              f"best={best_gic:.3f} @iter={conv_it}  "
              f"(still improves {gic_at_ref - best_gic:+.3f} after 2000)  "
              f"[{time.perf_counter() - t0:.0f}s]")

    fig.tight_layout()
    fig.savefig(out / f"lg_convergence_d{D}.png", dpi=110)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / f"lg_convergence_d{D}.csv", index=False)
    print(f"\nWrote {out / f'lg_convergence_d{D}.png'}")


if __name__ == "__main__":
    main()
