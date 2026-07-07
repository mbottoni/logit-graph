#!/usr/bin/env python3
"""Model-selection recovery across the classic AND modern baseline families.

The thesis validates that the spectral (KL) selection procedure recovers the true generative family
as the graph grows, for the six classic families. This experiment extends that check to include the
modern baselines: for each ground-truth family we generate a synthetic graph, fit ALL candidate
families to it, rank them by the ensemble-mean spectral KL, and record which family is selected.
Aggregated over replicates and sizes, the diagonal of the resulting confusion matrix is the
per-family recovery rate.

Families (generators and candidates): ER, BA, WS, KR, GRG, SBM (classic) and ChungLu, Config, RDPG,
DCSBM, HolmeKim, Hyperbolic (modern). LG is not included as a candidate here (its recovery is
established in the main thesis experiment); the focus is whether the modern families are
distinguishable under the same criterion.

Reproducible (seeded) and cached per (family, n, rep). Outputs recovery_per_graph.csv,
confusion.csv, and confusion.png under runs/model_recovery/.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import entropy

warnings.filterwarnings("ignore")

_here = Path(__file__).resolve().parent
_repo = _here.parents[1]
for p in (_repo / "src", _repo / "scripts" / "closedform", _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tlg_latent_gic_common as C   # noqa: E402  (KL scorer, ensemble baseline_gic, SBM)
import run_tlg_twitch_gic as tw     # noqa: E402  (closed_form_params, classic baseline generators)
import more_baselines_common as MB  # noqa: E402  (modern baseline fitters)

SEED = C.SEED
EVAL_SEEDS = C.EVAL_SEEDS
CACHE_VERSION = 1
NS = [int(x) for x in os.environ.get("MR_NS", "150,300").split(",")]
NREPS = int(os.environ.get("MR_NREPS", "5"))
FAMILIES = ["ER", "BA", "WS", "KR", "GRG", "SBM",
            "ChungLu", "Config", "RDPG", "DCSBM", "HolmeKim", "Hyperbolic"]
KBAR = float(os.environ.get("MR_KBAR", "10"))       # target average degree of the generated graphs


def log(*a):
    print(*a, flush=True)


def _simple(g):
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    g.remove_nodes_from(list(nx.isolates(g)))
    return nx.convert_node_labels_to_integers(g)


def _powerlaw_deg(n, gamma, kmin, seed):
    r = np.random.default_rng(seed)
    d = (kmin * (1 - r.random(n)) ** (-1.0 / (gamma - 1.0))).astype(int)
    d = np.clip(d, 1, n - 1)
    if d.sum() % 2:
        d[0] += 1
    return d.tolist()


# --------------------------------------------------------------------------- ground-truth generators
def _true_graph(family, n, seed):
    """A representative graph of `family` at size n and average degree ~KBAR."""
    m_ba = max(1, round(KBAR / 2))
    if family == "ER":
        return _simple(nx.gnp_random_graph(n, KBAR / (n - 1), seed=seed))
    if family == "BA":
        return _simple(nx.barabasi_albert_graph(n, m_ba, seed=seed))
    if family == "WS":
        return _simple(nx.watts_strogatz_graph(n, int(2 * round(KBAR / 2)), 0.15, seed=seed))
    if family == "KR":
        return _simple(nx.random_regular_graph(int(round(KBAR)), n, seed=seed))
    if family == "GRG":
        return _simple(nx.random_geometric_graph(n, np.sqrt(KBAR / (np.pi * n)), seed=seed))
    if family == "SBM":
        K = 4
        sizes = [n // K] * K
        sizes[-1] += n - sum(sizes)
        pin, pout = 3 * KBAR / (n), 0.3 * KBAR / n
        P = np.full((K, K), pout)
        np.fill_diagonal(P, pin)
        return _simple(nx.stochastic_block_model(sizes, P, seed=seed))
    if family in ("ChungLu", "Config"):
        d = _powerlaw_deg(n, 2.5, max(1, KBAR / 3), seed)
        g = (nx.expected_degree_graph(d, seed=seed, selfloops=False)
             if family == "ChungLu" else nx.configuration_model(d, seed=seed))
        return _simple(g)
    if family == "HolmeKim":
        return _simple(nx.powerlaw_cluster_graph(n, m_ba, 0.5, seed=seed))
    if family == "RDPG":
        r = np.random.default_rng(seed)
        k = 6
        X = np.abs(r.normal(0, 1, (n, k)))
        X *= np.sqrt(KBAR / n) / X.mean()               # scale to target density
        P = np.clip(X @ X.T, 0, 1)
        np.fill_diagonal(P, 0)
        M = np.triu((r.random((n, n)) < P).astype(float), 1)
        return _simple(nx.from_numpy_array(M + M.T))
    if family == "DCSBM":
        r = np.random.default_rng(seed)
        K = 4
        z = r.integers(0, K, n)
        theta = _powerlaw_deg(n, 2.5, 1, seed)
        theta = np.array(theta, float) / np.array([max(1, np.sum(np.array(theta)[z == z[i]]))
                                                   for i in range(n)])
        W = np.full((K, K), 0.3 * KBAR) + np.eye(K) * (3 * KBAR)
        P = np.clip((theta[:, None] * theta[None, :]) * W[np.ix_(z, z)], 0, 1)
        np.fill_diagonal(P, 0)
        M = np.triu((r.random((n, n)) < P).astype(float), 1)
        return _simple(nx.from_numpy_array(M + M.T))
    if family == "Hyperbolic":
        alpha = 0.8
        lo, hi = 2.0, 25.0
        target = KBAR * n / 2
        for _ in range(18):
            mid = (lo + hi) / 2
            e = MB._hyper_gen(n, alpha, mid, seed).number_of_edges()
            lo, hi = (mid, hi) if e > target else (lo, mid)
        return _simple(MB._hyper_gen(n, alpha, (lo + hi) / 2, seed))
    raise ValueError(family)


# --------------------------------------------------------------------------- candidate fits
def _candidate_generators(G):
    """{family: (gen_fn(seed)->Graph, n_params)} fitted to the observed graph G, for all families."""
    A = nx.to_numpy_array(G)
    deg = A.sum(1)
    cf = tw.closed_form_params(G)
    gens = tw._baseline_generators(G, cf)
    out = {}
    for fam in ("ER", "BA", "WS", "KR", "GRG"):
        out[fam] = gens[fam]
    _, sbm_np = C.generate_sbm_from_real(G, seed=SEED)
    out["SBM"] = ((lambda s: C.generate_sbm_from_real(G, seed=s)[0]), int(sbm_np))
    for name, fitter in (("ChungLu", MB.fit_chunglu), ("Config", MB.fit_config),
                         ("RDPG", MB.fit_rdpg), ("DCSBM", MB.fit_dcsbm),
                         ("HolmeKim", MB.fit_holmekim), ("Hyperbolic", MB.fit_hyper)):
        out[name] = fitter(G, A, deg)
    return out


def _select(G):
    """Fit every candidate family to G and return the argmin-KL family + all KLs."""
    scorer = C.GraphInformationCriterion(G, model="LG", dist="KL")
    real_den, _ = scorer.compute_spectral_density(G)
    kls = {}
    for fam, (gen, npar) in _candidate_generators(G).items():
        try:
            res = C.baseline_gic(G, gen, npar, real_den, scorer)
        except Exception:
            continue
        if res is not None:
            kls[fam] = float(res["kl"])
    if not kls:
        return None, {}
    return min(kls, key=kls.get), kls


# --------------------------------------------------------------------------- driver + cache
OUT = _here / "runs" / "model_recovery"


def run():
    cdir = OUT / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    log(f"model-recovery: families={FAMILIES} NS={NS} reps={NREPS} kbar={KBAR} seed={SEED}")
    t0 = time.perf_counter()
    for n in NS:
        for true_fam in FAMILIES:
            for rep in range(NREPS):
                key = f"{true_fam}_n{n}_r{rep}"
                done = cdir / f"{key}.json"
                if done.exists():
                    try:
                        if json.loads(done.read_text()).get("cache_version") == CACHE_VERSION:
                            continue
                    except Exception:
                        pass
                try:
                    G = _true_graph(true_fam, n, SEED + 1000 * rep + hash(true_fam) % 997)
                    if G.number_of_nodes() < 20 or G.number_of_edges() == 0:
                        raise ValueError("degenerate generated graph")
                    pred, kls = _select(G)
                except Exception as ex:
                    log(f"  {key}: FAILED ({ex})")
                    continue
                done.write_text(json.dumps(dict(cache_version=CACHE_VERSION, true=true_fam, n=n,
                                                rep=rep, pred=pred, kls=kls)))
                log(f"  {key}: true={true_fam} -> pred={pred}  "
                    f"({'OK' if pred == true_fam else 'miss'})")
    aggregate()
    log(f"done ({time.perf_counter()-t0:.1f}s)")


def aggregate():
    rows = []
    for jp in sorted(glob.glob(str(OUT / "cache" / "*.json"))):
        try:
            d = json.loads(Path(jp).read_text())
        except Exception:
            continue
        rows.append(dict(true=d["true"], n=d["n"], rep=d["rep"], pred=d["pred"],
                         correct=int(d["pred"] == d["true"])))
    if not rows:
        return
    per = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    per.to_csv(OUT / "recovery_per_graph.csv", index=False)
    conf = (per.groupby(["true", "pred"]).size()
            .unstack(fill_value=0).reindex(index=FAMILIES, columns=FAMILIES, fill_value=0))
    conf.to_csv(OUT / "confusion.csv")
    rec = per.groupby("true")["correct"].mean().reindex(FAMILIES)
    log("\n=== per-family recovery rate (diagonal of confusion; 1 = always recovered) ===")
    log(rec.round(3).to_string())
    log(f"overall recovery rate: {per['correct'].mean():.3f}")
    _plot(conf)


def _plot(conf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frac = conf.div(conf.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(frac.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(FAMILIES))); ax.set_xticklabels(FAMILIES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(FAMILIES))); ax.set_yticklabels(FAMILIES, fontsize=9)
    ax.set_xlabel("selected family (argmin KL)"); ax.set_ylabel("true generative family")
    ax.set_title("Model-selection recovery: classic + modern baselines\n(row-normalized; diagonal = recovery rate)")
    for i in range(len(FAMILIES)):
        for j in range(len(FAMILIES)):
            v = frac.values[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="fraction selected")
    fig.tight_layout()
    fig.savefig(OUT / "confusion.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {OUT/'confusion.png'}, confusion.csv, recovery_per_graph.csv")


if __name__ == "__main__":
    run()
