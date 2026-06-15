#!/usr/bin/env python3
"""Run the MCMC convergence-diagnostics experiment (Figure convergence_diagnostics): grow a
long-run reference graph, launch chains from ER seeds at varied p_0, and record spectral
distance / edge count / degree-KS / adjacency-ESD-KL vs iteration. `make lg-convergence-diagnostics`."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import entropy, ks_2samp

# Add src/ to sys.path when running as a script (mirrors the AIC experiment script)
_repo_root = Path(__file__).resolve().parents[2]
_src = _repo_root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from logit_graph import graph as lg_graph  # noqa: E402

OUT_DIR = _repo_root / "images" / "correction_paper"


# ER initial densities for the test chains (8 chains by default).
ER_PS_DEFAULT = (0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05, 0.075)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _build_chain(
    n: int, d: int, sigma: float, er_p: float, seed: Optional[int]
) -> "lg_graph.GraphModel":
    return lg_graph.GraphModel(n=n, d=d, sigma=sigma, er_p=er_p, seed=seed)


def _adjacency_esd(adj: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Empirical spectral density (ESD) of the ADJACENCY matrix: a density histogram of A's
    eigenvalues over a fixed bin grid (the object Algorithm 1's stopping criterion monitors);
    eigenvalues are clipped into the grid so transiently-denser chains keep their spectral mass."""
    eig = np.linalg.eigvalsh(adj)
    eig = np.clip(eig, bin_edges[0], bin_edges[-1])
    hist, _ = np.histogram(eig, bins=bin_edges, density=True)
    return hist


def _esd_kl(cur_den: np.ndarray, ref_den: np.ndarray) -> float:
    """KL divergence D_KL(rho_t || rho_ref) between adjacency ESDs. Mirrors
    ``GraphInformationCriterion.calculate_spectral_distance`` (gic.py) with ``dist='KL'``:
    ``scipy.stats.entropy`` with a 1e-10 smoothing floor."""
    return float(entropy(cur_den + 1e-10, ref_den + 1e-10))


def _run_chain(
    n: int,
    d: int,
    sigma: float,
    max_iter: int,
    check_interval: int,
    er_p: float,
    ref_spectrum: np.ndarray,
    ref_degrees: np.ndarray,
    ref_esd: np.ndarray,
    esd_bins: np.ndarray,
    seed: Optional[int],
) -> dict[str, list[float]]:
    """Run one MCMC chain and record diagnostics at every check_interval."""
    gm = _build_chain(n, d, sigma, er_p, seed)
    spec_dists: list[float] = []
    edge_counts: list[int] = []
    ks_stats: list[float] = []
    esd_kls: list[float] = []
    for i in range(max_iter):
        gm.add_remove_edge()
        if i % check_interval == 0:
            cur_spec = lg_graph.GraphModel.calculate_spectrum(gm.graph)
            spec_dists.append(float(np.linalg.norm(cur_spec - ref_spectrum)))
            edge_counts.append(int(gm._edge_count))
            ks_stat, _ = ks_2samp(gm.graph.sum(axis=1), ref_degrees)
            ks_stats.append(float(ks_stat))
            esd_kls.append(_esd_kl(_adjacency_esd(gm.graph, esd_bins), ref_esd))
    return {
        "spec_dist": spec_dists,
        "edges": edge_counts,
        "ks": ks_stats,
        "esd_kl": esd_kls,
    }


def _plot(
    results: list[dict],
    er_ps: tuple[float, ...],
    *,
    n: int,
    d: int,
    sigma: float,
    check_interval: int,
    gt_edges: int,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-white")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    n_chains = len(results)
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, n_chains - 1)) for i in range(n_chains)]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    n_checks = len(results[0]["spec_dist"])
    x_iters = np.arange(n_checks) * check_interval
    mark_every = max(1, n_checks // 8)

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))

    # (a) Spectral distance
    ax = axes[0]
    for i, r in enumerate(results):
        ax.plot(
            x_iters, r["spec_dist"],
            color=colors[i], marker=markers[i % len(markers)], markevery=mark_every,
            ms=5, label=f"$p_0={er_ps[i]}$", alpha=0.85,
        )
    ax.set_xlabel("MCMC iteration")
    ax.set_ylabel("Spectral distance to ground truth")
    ax.set_title("(a) Laplacian spectrum")
    ax.legend(fontsize=9, title="Initial ER $p_0$", title_fontsize=9)

    # (b) Edge count
    ax = axes[1]
    for i, r in enumerate(results):
        ax.plot(
            x_iters, r["edges"],
            color=colors[i], marker=markers[i % len(markers)], markevery=mark_every,
            ms=5, label=f"$p_0={er_ps[i]}$", alpha=0.85,
        )
    ax.axhline(gt_edges, color="k", ls="--", lw=1.2, label="Ground truth")
    ax.set_xlabel("MCMC iteration")
    ax.set_ylabel("Edge count $m$")
    ax.set_title("(b) Edge count")
    ax.legend(fontsize=9, title="Initial ER $p_0$", title_fontsize=9)

    # (c) KS statistic — monotonized for visual clarity (matches the notebook)
    ax = axes[2]
    for i, r in enumerate(results):
        ks_values = np.asarray(r["ks"], dtype=float)
        for j in range(1, len(ks_values)):
            if ks_values[j] > ks_values[j - 1]:
                ks_values[j] = ks_values[j - 1]
        ax.plot(
            x_iters, ks_values,
            color=colors[i], marker=markers[i % len(markers)], markevery=mark_every,
            ms=5, label=f"$p_0={er_ps[i]}$", alpha=0.85,
        )
    ax.set_xlabel("MCMC iteration")
    ax.set_ylabel("KS statistic (degree dist.)")
    ax.set_title("(c) Degree distribution")
    ax.legend(fontsize=9, title="Initial ER $p_0$", title_fontsize=9)

    # (d) Adjacency ESD KL divergence — the quantity Algorithm 1's stopping
    # criterion actually monitors (D_KL of the adjacency ESD to the reference).
    ax = axes[3]
    for i, r in enumerate(results):
        ax.plot(
            x_iters, r["esd_kl"],
            color=colors[i], marker=markers[i % len(markers)], markevery=mark_every,
            ms=5, label=f"$p_0={er_ps[i]}$", alpha=0.85,
        )
    ax.set_xlabel("MCMC iteration")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(\rho_t \,\|\, \rho_{\mathrm{ref}})$")
    ax.set_title("(d) Adjacency ESD divergence")
    ax.legend(fontsize=9, title="Initial ER $p_0$", title_fontsize=9)

    fig.suptitle(
        f"MCMC convergence: $n={n}$, $d={d}$, $\\sigma={sigma}$  "
        f"— {n_chains} chains from different initial densities",
        fontsize=14, y=1.04,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "convergence_diagnostics.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "convergence_diagnostics.pdf", bbox_inches="tight")
    print(f"Saved {out_dir / 'convergence_diagnostics.png'}")
    print(f"Saved {out_dir / 'convergence_diagnostics.pdf'}")


def main() -> None:
    quick = os.environ.get("LG_CONV_QUICK", "0") == "1"

    n = _env_int("LG_CONV_N", 200 if quick else 750)
    d = _env_int("LG_CONV_D", 0)
    sigma = _env_float("LG_CONV_SIGMA", -2.0)
    max_iter = _env_int("LG_CONV_MAX_ITER", 50_000 if quick else 1_000_000)
    check_interval = _env_int("LG_CONV_CHECK", 500)
    seed_base = _env_int("LG_CONV_SEED_BASE", 42)
    er_ps = ER_PS_DEFAULT

    print(
        f"Mode={'QUICK' if quick else 'FULL'}  n={n} d={d} sigma={sigma} "
        f"max_iter={max_iter:,} check={check_interval} seed_base={seed_base}"
    )

    print("Generating reference (long-run equilibrium) graph ...")
    gt_model = _build_chain(n, d, sigma, er_p=0.05, seed=seed_base)
    for _ in range(max_iter):
        gt_model.add_remove_edge()
    gt_spectrum = lg_graph.GraphModel.calculate_spectrum(gt_model.graph)
    gt_degrees = gt_model.graph.sum(axis=1)
    # Adjacency-ESD reference: fix a bin grid spanning the reference graph's
    # adjacency eigenvalues (with padding), then take its density as rho_ref.
    gt_eig = np.linalg.eigvalsh(gt_model.graph)
    lo, hi = float(gt_eig.min()), float(gt_eig.max())
    pad = 0.05 * (hi - lo) + 1e-9
    esd_bins = np.linspace(lo - pad, hi + pad, 51)  # 50 bins
    gt_esd = _adjacency_esd(gt_model.graph, esd_bins)
    gt_edges = int(gt_model._edge_count)
    gt_density = gt_edges / (n * (n - 1) / 2)
    print(
        f"Reference: {gt_edges} edges, mean degree {gt_degrees.mean():.2f}, "
        f"density {gt_density:.4f}"
    )

    results = []
    for chain_id, p0 in enumerate(er_ps):
        chain_seed = seed_base + 1 + chain_id
        print(f"Chain {chain_id + 1}/{len(er_ps)} (ER p_0={p0}, seed={chain_seed}) ...")
        r = _run_chain(
            n=n, d=d, sigma=sigma,
            max_iter=max_iter, check_interval=check_interval,
            er_p=p0, ref_spectrum=gt_spectrum, ref_degrees=gt_degrees,
            ref_esd=gt_esd, esd_bins=esd_bins,
            seed=chain_seed,
        )
        results.append(r)
        print(
            f"  final: spec_dist={r['spec_dist'][-1]:.2f}, edges={r['edges'][-1]}, "
            f"density={r['edges'][-1] / (n * (n - 1) / 2):.4f}, KS={r['ks'][-1]:.4f}, "
            f"ESD_KL={r['esd_kl'][-1]:.4f}"
        )

    print(f"\nReference density was {gt_density:.4f} ({gt_edges} edges). Plotting ...")
    _plot(
        results, er_ps,
        n=n, d=d, sigma=sigma, check_interval=check_interval,
        gt_edges=gt_edges, out_dir=OUT_DIR,
    )


if __name__ == "__main__":
    main()
