#!/usr/bin/env python3
"""Fit Twitter SNAP ego networks with the Temporal Logit-Graph (TLG) and rank families by GIC.

The Twitter twin of run_tlg_twitch_gic.py. Twitch is six big country graphs; Twitter is
973 small SNAP ego networks (n up to ~247), so instead of a detailed per-graph table we
**sample** ego nets in a size window and **aggregate** the family rankings (mean rank,
mean GIC, win counts) across nets — the same shape as run_lg_twitter_closedform.py.

It reuses the Twitch script's core TLG-GIC machinery (fit + baselines + GIC), so the
methodology is identical:
  * TLG: d by GIC over candidate depths (degenerate/under-overshooting depths excluded),
    sigma/alpha by logistic regression (FIT_MODE=mle) or min-GIC Nelder-Mead (FIT_MODE=gic),
    GIC by edge-gated monitored growth from a very sparse seed. n_params = 2.
  * ER/BA/WS/KR/GRG/SBM by closed-form; SBM n_params = k(k+1)/2.
TLG knobs are shared via the LG_TLGT_* env (read by the imported twitch module).

Output under runs/tlg_twitter_gic/ (gitignored):
  - all_nets.csv     per (ego net, family) rows: GIC, rank, GIC terms, n_params, metrics
  - summary.csv      per family: mean rank, mean GIC, mean KL, #wins, n_nets
  - summary.png      mean rank (and mean KL) by family across the sampled nets

Env knobs (twitter-specific; TLG knobs are LG_TLGT_*):
  LG_TLGTW_MAX_NETS (20)   ego nets to sample      LG_TLGTW_SEED (12345)
  LG_TLGTW_MIN_NODES (80)  size window lower       LG_TLGTW_MAX_NODES (247) upper
  LG_TLGTW_QUICK (0)       1 -> a few small nets

  make tlg-twitter-gic        sample + aggregate over ego nets
  make tlg-twitter-gic-quick  smoke (few nets)
"""
from __future__ import annotations

import os
import random
import sys
import time
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import networkx as nx
import pandas as pd

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
for p in (_src, _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")

from logit_graph.gic import GraphInformationCriterion  # noqa: E402
# Reuse the Twitch script's TLG-GIC core (importing it line-buffers stdout too).
import run_tlg_twitch_gic as tw  # noqa: E402

OUT_DIR = _here / "runs" / "tlg_twitter_gic"
DATA_DIR = _repo_root / "data" / "misc" / "twitter"


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


QUICK = os.environ.get("LG_TLGTW_QUICK", "0") == "1"
MAX_NETS = _int("LG_TLGTW_MAX_NETS", 3 if QUICK else 20)
MIN_NODES = _int("LG_TLGTW_MIN_NODES", 60 if QUICK else 80)
MAX_NODES = _int("LG_TLGTW_MAX_NODES", 247)
SEED = _int("LG_TLGTW_SEED", 12345)

FAMILIES = ["TLG", "ER", "BA", "WS", "KR", "GRG", "SBM"]


def _ensure_data():
    if DATA_DIR.exists() and any(DATA_DIR.glob("*.edges")):
        return
    tarball = _repo_root / "data" / "misc" / "twitter.tar.gz"
    if tarball.exists():
        import tarfile
        tw.log(f"extracting {tarball.relative_to(_repo_root)} ...")
        with tarfile.open(tarball) as tf:
            tf.extractall(_repo_root / "data" / "misc")


def _peek_size(path):
    nodes = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                nodes.add(parts[0])
                nodes.add(parts[1])
    return len(nodes)


def _load_edges(path):
    G = nx.read_edgelist(path, nodetype=int)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def _sample_files():
    _ensure_data()
    allf = sorted(DATA_DIR.glob("*.edges"))
    if not allf:
        return []
    inwin = [f for f in allf if MIN_NODES <= _peek_size(f) <= MAX_NODES]
    rng = random.Random(SEED)
    k = min(MAX_NETS, len(inwin))
    # smallest-first so the test starts on the cheapest nets
    return sorted(rng.sample(inwin, k), key=_peek_size), len(inwin), len(allf)


def score_net(name, G):
    """Per-net family comparison table (reuses the Twitch TLG-GIC core)."""
    adj = nx.to_numpy_array(G)
    real_m = tw._metrics(G)
    scorer = GraphInformationCriterion(G, model="LG", dist="KL")
    real_den, _ = scorer.compute_spectral_density(G)

    rows = []
    tlg = tw.fit_tlg(adj, real_den, scorer)
    tlg_m = (tw._metrics(tlg["graph"]) if tlg["graph"] is not None
             else dict(edges=np.nan, density=np.nan, clustering=np.nan, assortativity=np.nan))
    rows.append(dict(net=name, model="TLG", n_params=tlg["n_params"], gic=tlg["gic"],
                     gic_fit=2.0 * tlg["dist"], gic_penalty=2.0 * tlg["n_params"],
                     kl=tlg["dist"], d_hat=tlg["d_hat"], **tlg_m))

    cf = tw.closed_form_params(G)
    gens = tw._baseline_generators(G, cf)
    for fam in ("ER", "BA", "WS", "KR", "GRG", "SBM"):
        gen_fn, n_params = gens[fam]
        if fam == "SBM":
            _, n_params = tw.generate_sbm_from_real(G, seed=tw.SEED)
        res = tw.baseline_gic(G, fam, gen_fn, n_params, real_den, scorer, tw.SEED)
        if res is None:
            continue
        m = tw._metrics(res["graph"]) if res["graph"] is not None else dict(
            edges=np.nan, density=np.nan, clustering=np.nan, assortativity=np.nan)
        rows.append(dict(net=name, model=fam, n_params=res["n_params"], gic=res["gic"],
                         gic_fit=2.0 * res["dist"], gic_penalty=2.0 * res["n_params"],
                         kl=res["dist"], d_hat=np.nan, **m))

    df = pd.DataFrame(rows).sort_values("gic").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["n"] = G.number_of_nodes()
    df["real_clustering"] = real_m["clustering"]
    return df


def main():
    tw.log(f"TLG Twitter GIC  fit={tw.FIT_MODE} quick={QUICK} max_nets={MAX_NETS} "
           f"window=[{MIN_NODES},{MAX_NODES}] seed={SEED}  (TLG knobs via LG_TLGT_*)")
    sampled = _sample_files()
    if not sampled:
        tw.log(f"No twitter .edges under {DATA_DIR} (gitignored — place the SNAP "
               f"twitter tarball/edges there first).")
        return
    files, n_win, n_total = sampled
    tw.log(f"sampled {len(files)} ego nets (of {n_win} in window, {n_total} total)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_df = []
    for i, f in enumerate(files, 1):
        name = f.stem
        G = _load_edges(f)
        t0 = time.perf_counter()
        df = score_net(name, G)
        all_df.append(df)
        top = df.iloc[0]
        tlg = df[df.model == "TLG"].iloc[0]
        tw.log(f"  [{i}/{len(files)}] {name} n={G.number_of_nodes()} E={G.number_of_edges()} "
               f"clust={df['real_clustering'].iloc[0]:.2f} | TLG d={int(tlg['d_hat'])} "
               f"rank={int(tlg['rank'])} KL={tlg['kl']:.3f} | best={top['model']} "
               f"({top['gic']:.2f})  ({time.perf_counter()-t0:.1f}s)")

    combined = pd.concat(all_df, ignore_index=True)
    combined.to_csv(OUT_DIR / "all_nets.csv", index=False)

    def _finite_mean(s):
        f = s[np.isfinite(s)]
        return float(f.mean()) if len(f) else np.nan

    g = combined.groupby("model")
    summary = pd.DataFrame({
        "mean_rank": g["rank"].mean(),
        "std_rank": g["rank"].std(),
        "mean_gic": g["gic"].apply(_finite_mean),   # finite-only (TLG may not score some nets)
        "mean_kl": g["kl"].apply(_finite_mean),
        "n_scored": g["gic"].apply(lambda s: int(np.isfinite(s).sum())),
        "wins": g.apply(lambda x: int((x["rank"] == 1).sum())),
        "n_nets": g["net"].nunique(),
    }).sort_values("mean_rank")
    summary.to_csv(OUT_DIR / "summary.csv")
    tw.log("\n=== aggregate over sampled ego nets (mean rank, lower = better) ===")
    tw.log(summary.to_string())
    _plot_summary(summary, OUT_DIR / "summary.png")
    tw.log(f"\nWrote {OUT_DIR}/ (all_nets.csv, summary.csv, summary.png)")


def _plot_summary(summary, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary.reset_index()
    colors = [tw.CB.get(m, "#888") for m in s["model"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.bar(s["model"], s["mean_rank"], yerr=s["std_rank"], color=colors, capsize=3)
    ax1.set_ylabel("mean GIC rank (lower = better)")
    ax1.set_title(f"Twitter ego nets: mean GIC rank by family (n={int(s['n_nets'].iloc[0])} nets)")
    ax1.grid(alpha=0.25, axis="y")
    ax2.bar(s["model"], s["mean_kl"], color=colors)
    ax2.set_ylabel("mean KL (spectral fit; lower = better)")
    ax2.set_title("mean spectral fit (2·KL term) by family")
    ax2.grid(alpha=0.25, axis="y")
    fig.suptitle(f"TLG vs closed-form families on Twitter SNAP ego networks "
                 f"(fit={tw.FIT_MODE})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    tw.log(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
