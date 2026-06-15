#!/usr/bin/env python3
"""Evaluate ESD-KL convergence stopping for the temporal Logit-Graph: grow_graph(until_convergence)
stops when consecutive-snapshot ESD-KL stays below esd_tol for patience steps — this asks whether
that stops without bias, how it scales with n, and tol sensitivity. `make tlg-esd-stop-eval`."""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_repo_root = Path(__file__).resolve().parents[2]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from logit_graph.temporal import grow_graph, _adjacency_esd_kl  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "runs" / "tlg_esd_stop"


def _int(name, default):
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _float(name, default):
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def _ints(name, default):
    raw = os.environ.get(name)
    return [int(x) for x in raw.split(",")] if raw else default


def _reference(n, d, sigma, alpha, steps, seed):
    """One long fixed-run draw from the stationary distribution."""
    return grow_graph(n, d=d, sigma=sigma, alpha=alpha, n_steps=steps, seed=seed,
                      store_snapshots=False).adj


def main():
    quick = os.environ.get("LG_ESS_QUICK", "0") == "1"
    NS = _ints("LG_ESS_NS", [100, 200] if quick else [100, 200, 400, 750])
    d = _int("LG_ESS_D", 0)
    sigma = _float("LG_ESS_SIGMA", -2.0)
    alpha = _float("LG_ESS_ALPHA", 0.05)
    steps = _int("LG_ESS_STEPS", 30 if quick else 60)
    tol = _float("LG_ESS_TOL", 1e-2)
    patience = _int("LG_ESS_PATIENCE", 3)
    reps = _int("LG_ESS_REPS", 4 if quick else 8)
    seed0 = _int("LG_ESS_SEED", 0)
    print(f"ESD-stop eval  mode={'QUICK' if quick else 'FULL'}  n={NS} d={d} "
          f"sigma={sigma} alpha={alpha} cap={steps} tol={tol} patience={patience} "
          f"reps={reps}")

    rows = []
    traces = {}  # n -> (trace, stop_step) for one example rep
    for n in NS:
        # Two independent long references: ref vs ref2 gives the noise floor.
        ref = _reference(n, d, sigma, alpha, steps, seed0 + 9991)
        ref2 = _reference(n, d, sigma, alpha, steps, seed0 + 9992)
        floor_ks, _ = ks_2samp(ref.sum(1), ref2.sum(1))
        floor_kl = _adjacency_esd_kl(ref, ref2)

        for r in range(reps):
            res = grow_graph(n, d=d, sigma=sigma, alpha=alpha, n_steps=steps,
                             seed=seed0 + 1 + r, until_convergence=True,
                             esd_tol=tol, patience=patience, store_snapshots=False)
            p = res.params
            ks, _ = ks_2samp(res.adj.sum(1), ref.sum(1))
            kl = _adjacency_esd_kl(res.adj, ref)
            rows.append(dict(n=n, rep=r, converged=p["converged"],
                             stop_step=p["n_steps_run"], cap=steps,
                             ks_vs_ref=float(ks), kl_vs_ref=kl,
                             floor_ks=float(floor_ks), floor_kl=floor_kl,
                             density=float(res.adj.sum() / (n * (n - 1)))))
            if r == 0:
                traces[n] = (p["esd_kl_trace"], p["n_steps_run"])
        sub = [x for x in rows if x["n"] == n]
        conv = np.mean([x["converged"] for x in sub])
        steps_arr = np.array([x["stop_step"] for x in sub])
        ks_arr = np.array([x["ks_vs_ref"] for x in sub])
        print(f"  n={n:4d}: converged {conv*100:3.0f}%  stop_step "
              f"{steps_arr.mean():4.1f} [{steps_arr.min()}-{steps_arr.max()}]  "
              f"KS(stop,ref)={ks_arr.mean():.3f}  floor_KS={floor_ks:.3f}")

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "esd_stop_eval.csv", index=False)

    # ---- esd_tol sensitivity at the largest n -------------------------------
    n_sens = NS[-1]
    ref = _reference(n_sens, d, sigma, alpha, steps, seed0 + 9991)
    print(f"\nesd_tol sensitivity at n={n_sens} (patience={patience}):")
    for t in (5e-2, 2e-2, 1e-2, 5e-3, 1e-3):
        ss, kk = [], []
        for r in range(reps):
            res = grow_graph(n_sens, d=d, sigma=sigma, alpha=alpha, n_steps=steps,
                             seed=seed0 + 1 + r, until_convergence=True,
                             esd_tol=t, patience=patience, store_snapshots=False)
            ss.append(res.params["n_steps_run"])
            ks, _ = ks_2samp(res.adj.sum(1), ref.sum(1))
            kk.append(ks)
        print(f"  tol={t:6.4f}: stop_step {np.mean(ss):4.1f}  KS(stop,ref)={np.mean(kk):.3f}")

    _plot(df, traces, n=NS, sigma=sigma, alpha=alpha, tol=tol, out_dir=OUT_DIR)
    print(f"\nWrote {OUT_DIR}/ (esd_stop_eval.csv, esd_stop_eval.png)")


def _plot(df, traces, *, n, sigma, alpha, tol, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    cmap = plt.cm.viridis
    ns = sorted(df["n"].unique())
    colors = {nn: cmap(i / max(1, len(ns) - 1)) for i, nn in enumerate(ns)}

    # (a) stop step vs n
    ax = axes[0]
    g = df.groupby("n")["stop_step"]
    mean = g.mean()
    ax.plot(mean.index, mean.values, "o-", color="#0072B2", lw=1.8)
    ax.fill_between(mean.index, g.min().values, g.max().values, color="#0072B2",
                    alpha=0.15)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(v) for v in ns])
    ax.set_xlabel("n (nodes)"); ax.set_ylabel("stop step")
    ax.set_title("(a) steps to convergence vs n"); ax.grid(alpha=0.25)

    # (b) KS(stop, ref) vs n with noise-floor band
    ax = axes[1]
    g2 = df.groupby("n")
    ks_mean = g2["ks_vs_ref"].mean()
    floor = g2["floor_ks"].first()
    ax.plot(ks_mean.index, ks_mean.values, "o-", color="#D55E00", lw=1.8,
            label="KS(stopped, reference)")
    ax.plot(floor.index, floor.values, "s--", color="#000000", lw=1.2,
            label="noise floor  KS(ref, ref$_2$)")
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(v) for v in ns])
    ax.set_xlabel("n (nodes)"); ax.set_ylabel("degree-distribution KS")
    ax.set_title("(b) stopped graph vs stationary draw"); ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # (c) example ESD-KL traces with stop marked
    ax = axes[2]
    for nn in ns:
        tr, stop = traces[nn]
        x = np.arange(1, len(tr) + 1)
        ax.plot(x, tr, "-", color=colors[nn], lw=1.4, label=f"n={nn}")
        ax.scatter([stop], [tr[stop - 1]], color=colors[nn], s=40, zorder=5,
                   edgecolor="k", linewidth=0.5)
    ax.axhline(tol, color="grey", ls=":", lw=1.0, label=f"tol={tol:g}")
    ax.set_yscale("log"); ax.set_xlabel("growth step")
    ax.set_ylabel(r"consecutive ESD-KL  $D_{\mathrm{KL}}(\rho_t\|\rho_{t-1})$")
    ax.set_title("(c) ESD-KL trace (● = stop)"); ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(f"TLG ESD-KL convergence stopping — $\\sigma={sigma:g}$, "
                 f"$\\alpha={alpha:g}$, tol={tol:g} (● marks early stop)", y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "esd_stop_eval.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'esd_stop_eval.png'}")


if __name__ == "__main__":
    main()
