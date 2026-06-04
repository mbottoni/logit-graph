"""End-to-end incremental-feature LG experiment (Condition D focus + baselines).

Run: python lg_incremental_feature_experiment.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_DENSITY = 0.10
SIGNAL_INC = 0.5
SIGNAL_ADD = 1.5
ALPHA_GWESP = 2.0
AIC_PENALTY_PER_D = 3.0

D_TRUE_VALUES = [0, 1, 2, 3]
D_EST_VALUES = [0, 1, 2, 3]
N_RUNS = 12
M_ENSEMBLE = 5
N_SIZES = [60, 100, 150]
N_ITER_BY_N = {60: 10_000, 100: 15_000, 150: 25_000}

OUT_DIR = (Path(__file__).resolve().parents[2] / "images" / "correction_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-white")
mpl.rcParams.update({"font.family": "serif", "savefig.dpi": 150, "savefig.bbox": "tight"})


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def _ball(nbrs, v, d):
    visited = {v}
    current = [v]
    for _ in range(d):
        nxt = []
        for u in current:
            for nu in nbrs[u]:
                if nu not in visited:
                    visited.add(nu)
                    nxt.append(nu)
        current = nxt
        if not current:
            break
    return visited


def common_dhop(nbrs, i, j, d):
    if d == 0:
        return 0
    bi = _ball(nbrs, i, d)
    bj = _ball(nbrs, j, d)
    inter = bi & bj
    inter.discard(i)
    inter.discard(j)
    return len(inter)


def incremental_h(nbrs, i, j, d):
    if d == 0:
        return 0.0
    if d == 1:
        c = common_dhop(nbrs, i, j, 1)
        return ALPHA_GWESP * (1.0 - (1.0 - 1.0 / ALPHA_GWESP) ** c) if c > 0 else 0.0
    delta = common_dhop(nbrs, i, j, d) - common_dhop(nbrs, i, j, d - 1)
    return math.log(1.0 + max(0, delta))


def bfs_sum_degree(nbrs, vertex, d):
    if d == 0:
        return float(len(nbrs[vertex]))
    visited = {vertex}
    total = float(len(nbrs[vertex]))
    current = [vertex]
    for _ in range(d):
        nxt = []
        for v in current:
            for nv in nbrs[v]:
                if nv not in visited:
                    visited.add(nv)
                    nxt.append(nv)
                    total += float(len(nbrs[nv]))
        current = nxt
        if not current:
            break
    return total


def additive_h(nbrs, i, j, d):
    return math.log(1.0 + bfs_sum_degree(nbrs, i, d)) + math.log(1.0 + bfs_sum_degree(nbrs, j, d))


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
class LGIncremental:
    """Generative: logit = sigma + beta * h_d (single incremental feature at true d)."""

    def __init__(self, n, d, target_density, signal, seed):
        self.n = int(n)
        self.d = int(d)
        self.rng = np.random.default_rng(seed)
        cal = [set() for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.rng.random() < target_density:
                    cal[i].add(j)
                    cal[j].add(i)
        samples = []
        for _ in range(300):
            i = int(self.rng.integers(0, self.n))
            j = int(self.rng.integers(0, self.n))
            if i == j:
                continue
            had = j in cal[i]
            if had:
                cal[i].discard(j)
                cal[j].discard(i)
            samples.append(incremental_h(cal, i, j, self.d))
            if had:
                cal[i].add(j)
                cal[j].add(i)
        if self.d == 0:
            self.beta = 0.0
            self.sigma = math.log(target_density / (1 - target_density))
        else:
            scale = max(0.01, float(np.mean(samples)))
            self.beta = signal / scale
            self.sigma = math.log(target_density / (1 - target_density)) - self.beta * scale
        self.nbrs = cal

    def step(self):
        i = int(self.rng.integers(0, self.n))
        j = int(self.rng.integers(0, self.n - 1))
        if j >= i:
            j += 1
        had = j in self.nbrs[i]
        if had:
            self.nbrs[i].discard(j)
            self.nbrs[j].discard(i)
        f = incremental_h(self.nbrs, i, j, self.d)
        lg = self.sigma + self.beta * f
        p = 1.0 / (1.0 + math.exp(-lg)) if lg >= 0 else math.exp(lg) / (1.0 + math.exp(lg))
        if self.rng.random() < p:
            self.nbrs[i].add(j)
            self.nbrs[j].add(i)

    def run(self, n_iter):
        for _ in range(n_iter):
            self.step()

    def density(self):
        return 2 * sum(len(s) for s in self.nbrs) // 2 / (self.n * (self.n - 1))

    def degree_sequence(self):
        return np.array([len(s) for s in self.nbrs])

    def clustering_coefficient(self):
        tri = trip = 0
        for v in range(self.n):
            deg = len(self.nbrs[v])
            if deg < 2:
                continue
            trip += deg * (deg - 1) // 2
            nv = list(self.nbrs[v])
            for a_idx in range(len(nv)):
                a = nv[a_idx]
                for b in nv[a_idx + 1 :]:
                    if b in self.nbrs[a]:
                        tri += 1
        return (3 * tri) / trip if trip > 0 else 0.0


class LGAdditive:
    def __init__(self, n, d, target_density, signal, seed):
        self.n = int(n)
        self.d = int(d)
        self.rng = np.random.default_rng(seed)
        cal = [set() for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.rng.random() < target_density:
                    cal[i].add(j)
                    cal[j].add(i)
        samples = []
        for _ in range(300):
            i = int(self.rng.integers(0, self.n))
            j = int(self.rng.integers(0, self.n))
            if i == j:
                continue
            had = j in cal[i]
            if had:
                cal[i].discard(j)
                cal[j].discard(i)
            samples.append(additive_h(cal, i, j, self.d))
            if had:
                cal[i].add(j)
                cal[j].add(i)
        scale = max(0.5, float(np.mean(samples)))
        self.beta = signal / scale
        self.sigma = math.log(target_density / (1 - target_density)) - self.beta * scale
        self.nbrs = cal

    def step(self):
        i = int(self.rng.integers(0, self.n))
        j = int(self.rng.integers(0, self.n - 1))
        if j >= i:
            j += 1
        had = j in self.nbrs[i]
        if had:
            self.nbrs[i].discard(j)
            self.nbrs[j].discard(i)
        f = additive_h(self.nbrs, i, j, self.d)
        lg = self.sigma + self.beta * f
        p = 1.0 / (1.0 + math.exp(-lg)) if lg >= 0 else math.exp(lg) / (1.0 + math.exp(lg))
        if self.rng.random() < p:
            self.nbrs[i].add(j)
            self.nbrs[j].add(i)

    def run(self, n_iter):
        for _ in range(n_iter):
            self.step()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _fit_logit(rows, labels, extra_penalty=0.0):
    x = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=int)
    n_obs = len(y)
    if y.sum() == 0 or y.sum() == n_obs:
        return np.nan
    if x.ndim == 1 and x.std() < 1e-9:
        p_hat = y.mean()
        ll = y.sum() * math.log(p_hat) + (n_obs - y.sum()) * math.log(1 - p_hat)
        return -2 * ll + 2 * 1 + extra_penalty
    if x.ndim == 1:
        feats = np.column_stack([np.ones(n_obs), x])
    else:
        feats = np.column_stack([np.ones(n_obs), x])
    result = None
    for method in ("newton", "bfgs", "lbfgs"):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                result = sm.Logit(y, feats).fit(method=method, disp=False, maxiter=300)
            if np.isfinite(result.llf):
                break
        except Exception:
            continue
    if result is None or not np.isfinite(result.llf):
        return np.nan
    k = len(result.params)
    return -2 * float(result.llf) + 2 * k + extra_penalty


def _collect_pairs(graphs, n, feat_fn, d_est):
    rows, labels = [], []
    for nbrs in graphs:
        for i in range(n):
            for j in range(i + 1, n):
                had = j in nbrs[i]
                if had:
                    nbrs[i].discard(j)
                    nbrs[j].discard(i)
                rows.append(feat_fn(nbrs, i, j, d_est))
                labels.append(1 if had else 0)
                if had:
                    nbrs[i].add(j)
                    nbrs[j].add(i)
    return rows, labels


def aic_additive(graphs, n, d_est):
    rows, labels = _collect_pairs(graphs, n, additive_h, d_est)
    return _fit_logit(rows, labels)


def aic_incremental(graphs, n, d_est, penalized=False):
    rows, labels = _collect_pairs(graphs, n, incremental_h, d_est)
    extra = AIC_PENALTY_PER_D * d_est if penalized else 0.0
    return _fit_logit(rows, labels, extra_penalty=extra)


def wilson_ci(k, n):
    if n == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def run_sweep(n, n_iter, sampler_cls, signal, aic_fn, m_ensemble=1, seed_base=0):
    rng = np.random.default_rng(seed_base)
    n_chains = N_RUNS * m_ensemble
    graphs = {d: [] for d in D_TRUE_VALUES}
    t0 = time.perf_counter()
    for d_true in D_TRUE_VALUES:
        for _ in range(n_chains):
            seed = int(rng.integers(0, 2**31 - 1))
            s = sampler_cls(n, d_true, TARGET_DENSITY, signal, seed=seed)
            s.run(n_iter)
            graphs[d_true].append(s.nbrs)
    gen_t = time.perf_counter() - t0

    conf = {dt: {de: 0 for de in D_EST_VALUES} for dt in D_TRUE_VALUES}
    t1 = time.perf_counter()
    for d_true in D_TRUE_VALUES:
        for k in range(N_RUNS):
            if m_ensemble > 1:
                nbrs_list = graphs[d_true][k * m_ensemble : (k + 1) * m_ensemble]
            else:
                nbrs_list = [graphs[d_true][k]]
            aics = {de: aic_fn(nbrs_list, n, de) for de in D_EST_VALUES}
            valid = {d: v for d, v in aics.items() if np.isfinite(v)}
            if valid:
                conf[d_true][min(valid, key=valid.get)] += 1
    aic_t = time.perf_counter() - t1
    return conf, gen_t, aic_t


def print_confusion(conf, label):
    print(f"\n=== {label} ===")
    accs = []
    for dt in D_TRUE_VALUES:
        row = conf[dt]
        total = sum(row.values())
        on = row[dt]
        lo, hi = wilson_ci(on, total)
        acc = on / max(1, total)
        accs.append(acc)
        hats = " ".join(f"hat={de}:{row[de]:2d}" for de in D_EST_VALUES)
        print(f"  true={dt}  {hats}  acc={acc*100:.0f}% [{lo*100:.0f}-{hi*100:.0f}]")
    print(f"  overall={100*np.mean(accs):.0f}%")


def plot_scaling(all_scaling, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    for i, dt in enumerate(D_TRUE_VALUES):
        ys = [all_scaling[n][dt] for n in N_SIZES]
        ax.plot(N_SIZES, ys, "o-", color=colors[i], linewidth=2, markersize=8, label=rf"$d_{{\mathrm{{true}}}}={dt}$")
    ax.set_xlabel(r"Graph size $n$")
    ax.set_ylabel(r"Recovery rate $\mathbb{P}(\hat d = d_{\mathrm{true}})$")
    ax.set_title(rf"Condition D: incremental + penalized AIC, ensemble $M={M_ENSEMBLE}$")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_confusion_panels(all_conf, labels, out_path):
    fig, axes = plt.subplots(1, len(N_SIZES), figsize=(6 * len(N_SIZES), 5.5))
    if len(N_SIZES) == 1:
        axes = [axes]
    im = None
    for ax, n in zip(axes, N_SIZES):
        conf = all_conf[n]
        mat = np.zeros((4, 4))
        for i, dt in enumerate(D_TRUE_VALUES):
            total = sum(conf[dt].values())
            for j, de in enumerate(D_EST_VALUES):
                mat[i, j] = conf[dt][de] / max(1, total)
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4))
        ax.set_xticklabels([str(d) for d in D_EST_VALUES])
        ax.set_yticks(range(4))
        ax.set_yticklabels([str(d) for d in D_TRUE_VALUES])
        ax.set_xlabel(r"$\hat d$")
        ax.set_ylabel(r"$d_{\mathrm{true}}$")
        for i in range(4):
            for j in range(4):
                v = mat[i, j]
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center",
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if i == j else "normal")
        acc = np.mean([mat[i, i] for i in range(4)])
        ax.set_title(rf"$n={n}$, acc={acc*100:.0f}%")
    if im:
        fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    fig.suptitle("Incremental + penalized AIC (Condition D)", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 72)
    print("Incremental-feature LG experiment")
    print(f"N_SIZES={N_SIZES}, N_RUNS={N_RUNS}, M={M_ENSEMBLE}, pen={AIC_PENALTY_PER_D}")
    print("=" * 72)

    # Identifiability quick check at n=60
    print("\n--- Identifiability (incremental sampler, n=60) ---")
    stats = {d: {"density": [], "feat": []} for d in D_TRUE_VALUES}
    for d in D_TRUE_VALUES:
        for rep in range(10):
            s = LGIncremental(60, d, TARGET_DENSITY, SIGNAL_INC, seed=100 + rep)
            s.run(10_000)
            feats = []
            for _ in range(80):
                i = int(s.rng.integers(0, s.n))
                j = int(s.rng.integers(0, s.n))
                if i == j:
                    continue
                feats.append(incremental_h(s.nbrs, i, j, d))
            stats[d]["density"].append(s.density())
            stats[d]["feat"].append(float(np.mean(feats)))
        print(f"  d={d}: density={np.mean(stats[d]['density']):.3f}  "
              f"mean_h={np.mean(stats[d]['feat']):.3f}")

    n_sig = 0
    for a in D_TRUE_VALUES:
        for b in D_TRUE_VALUES:
            if b <= a:
                continue
            ks = ks_2samp(stats[a]["feat"], stats[b]["feat"])
            if ks.pvalue < 0.05:
                n_sig += 1
    print(f"  KS-significant feat pairs: {n_sig}/6")

    # Main sweeps
    all_conf_D = {}
    scaling = {n: {} for n in N_SIZES}
    results_summary = []

    for n in N_SIZES:
        n_iter = N_ITER_BY_N[n]
        print(f"\n--- n={n}, n_iter={n_iter} ---")

        # Condition A: additive single
        conf_a, gt, at = run_sweep(
            n, n_iter, LGAdditive, SIGNAL_ADD, aic_additive,
            m_ensemble=1, seed_base=1000 + n,
        )
        print_confusion(conf_a, f"A additive M=1 [{gt:.0f}s gen, {at:.0f}s aic]")

        # Condition D: incremental penalized ensemble
        conf_d, gt, at = run_sweep(
            n, n_iter, LGIncremental, SIGNAL_INC,
            lambda g, n, d: aic_incremental(g, n, d, penalized=True),
            m_ensemble=M_ENSEMBLE, seed_base=2000 + n,
        )
        print_confusion(conf_d, f"D incremental penalized M={M_ENSEMBLE} [{gt:.0f}s gen, {at:.0f}s aic]")
        all_conf_D[n] = conf_d
        for dt in D_TRUE_VALUES:
            total = sum(conf_d[dt].values())
            acc = conf_d[dt][dt] / max(1, total)
            scaling[n][dt] = acc
            results_summary.append({"n": n, "d_true": dt, "recovery": acc, "condition": "D"})

    # Save artifacts
    scaling_path = OUT_DIR / "aic_d_incremental_scaling.png"
    plot_scaling(scaling, scaling_path)
    conf_path = OUT_DIR / "aic_d_incremental_confusion_n_sweep.png"
    plot_confusion_panels(all_conf_D, ["D"] * len(N_SIZES), conf_path)

    summary_df = pd.DataFrame(results_summary)
    csv_path = OUT_DIR / "aic_d_incremental_scaling.csv"
    summary_df.to_csv(csv_path, index=False)

    print("\n--- Scaling summary (Condition D) ---")
    print(summary_df.pivot(index="d_true", columns="n", values="recovery").round(3).to_string())
    print(f"\nSaved: {scaling_path}")
    print(f"Saved: {conf_path}")
    print(f"Saved: {csv_path}")

    # Write compact JSON for notebook import
    meta = {
        "scaling": {str(n): {str(dt): scaling[n][dt] for dt in D_TRUE_VALUES} for n in N_SIZES},
        "config": {
            "N_SIZES": N_SIZES, "N_RUNS": N_RUNS, "M_ENSEMBLE": M_ENSEMBLE,
            "SIGNAL_INC": SIGNAL_INC, "AIC_PENALTY_PER_D": AIC_PENALTY_PER_D,
        },
    }
    (OUT_DIR / "aic_d_incremental_results.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
