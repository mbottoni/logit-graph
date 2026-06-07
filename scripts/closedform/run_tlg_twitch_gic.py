#!/usr/bin/env python3
"""Fit the Twitch networks with the Temporal Logit-Graph (TLG) and rank families by GIC.

For each Twitch country graph (default: the smallest, PTBR), we fit several random-graph
families and compare them with the spectral GIC = 2*KL(real||model) + 2*n_params
(KL on the normalized-Laplacian spectral density; lower = better):

  * TLG (our model) — fit to the *whole* real graph (TLG is cheap, no subsampling):
      - d  : the degree-feature depth, chosen by the LOWEST GIC over candidates — each
             d is fit and grown, and the d (and growth iteration) with the smallest GIC
             wins. Since n_params=2 is constant across d, this is the best joint
             (d, iteration) spectral fit, and a degenerate depth (whose generator blows
             past the edge guard and never scores) is excluded automatically. (AIC per
             d is reported for reference; note AIC can prefer the degenerate depth.)
      - sigma, alpha : the intercept and degree coefficient — by logistic regression
             (FIT_MODE=mle, the MLE) or by directly minimizing the GIC with a
             warm-started Nelder-Mead search (FIT_MODE=gic), putting TLG on the same
             min-GIC footing as the grid baselines;
      - GIC: grow a TLG graph at (sigma, alpha, d) seeded at the real graph's density
             (robust for sparse and dense targets), in two
             phases. Phase 1 (edge gate): densify until the TLG reaches a similar edge
             count to the real graph, E >= (1-EDGE_TOL)*E_real. Phase 2 (GIC patience):
             from there, monitor the spectral distance to the real graph every step and
             keep the lowest-GIC iteration, early-stopping after ``patience``
             non-improving steps. Guard: stop if the chain overshoots to
             E > EDGE_FACTOR*E_real.   n_params = 2 (sigma, alpha; d is selected).
  * ER / BA / WS / KR / GRG — closed-form (moment-matched) parameters, ensemble-mean
             spectral density.
  * SBM — Louvain blocks + per-block edge probabilities; n_params = k(k+1)/2.

Per graph it writes a table with, for each family: GIC, its rank, the two GIC terms
(2*KL goodness-of-fit and 2*n_params penalty), n_params, and the structural metrics of
the fitted/representative graph (edges, density, clustering, assortativity) next to a
"Real" reference row.

Output under runs/tlg_twitch_gic/ (gitignored):
  - <region>_table.csv     per-graph family comparison table
  - <region>_gic_bar.png   GIC by family (ranked)
  - <region>_tlg_trace.png TLG GIC vs growth iteration (min marked)
  - summary.csv            rank of each family across regions (+ mean rank)

Env knobs (all optional):
  LG_TLGT_REGIONS (PTBR)   comma list; LG_TLGT_ALL=1 -> all six
  LG_TLGT_DGRID (0,1,2)    candidate depths for AIC d-selection
  LG_TLGT_MAXSTEPS (40)    TLG growth cap     LG_TLGT_PATIENCE (8)   GIC early-stop
  LG_TLGT_NRUNS (5)        baseline ensemble size
  LG_TLGT_P0 (real density) TLG growth seed density; unset -> the real graph's density
  LG_TLGT_EDGE_TOL (0.2)   phase-1 gate: start GIC monitoring once E >= (1-tol)*E_real
  LG_TLGT_EDGE_FACTOR (2)  guard: stop growth once edges exceed this multiple of E_real
  LG_TLGT_FIT (mle)        "mle" (logistic) or "gic" (min-GIC Nelder-Mead search)
  LG_TLGT_GIC_SEEDS (2)    growth seeds averaged per GIC objective eval (gic mode)
  LG_TLGT_GIC_MAXITER (40) Nelder-Mead iteration cap (gic mode)
  LG_TLGT_SEED (12345)
  LG_TLGT_QUICK (0)        1 -> tiny (fewer steps/runs)

  make tlg-twitch-gic        full run (PTBR, the smallest)
  make tlg-twitch-gic-quick  smoke
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import entropy

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)  # stream progress live (no buffering)
except Exception:
    pass


def log(*a):
    print(*a, flush=True)


from logit_graph.gic import GraphInformationCriterion  # noqa: E402
from logit_graph.temporal import grow_graph, fit_growth_params  # noqa: E402
from logit_graph.lg_features import build_pair_dataset  # noqa: E402
from logit_graph.sbm import generate_sbm_from_real  # noqa: E402

OUT_DIR = _here / "runs" / "tlg_twitch_gic"
DATA_DIR = _repo_root / "data" / "twitch" / "graphs_processed"
ALL_REGIONS = ["PTBR", "RU", "ES", "ENGB", "FR", "DE"]  # smallest -> largest
TLG_N_PARAMS = 2  # sigma, alpha (d is a selected hyperparameter, reported separately)


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


def _float(env, default):
    raw = os.environ.get(env)
    return float(raw) if raw is not None else default


QUICK = os.environ.get("LG_TLGT_QUICK", "0") == "1"
if os.environ.get("LG_TLGT_ALL", "0") == "1":
    REGIONS = ALL_REGIONS
else:
    REGIONS = os.environ.get("LG_TLGT_REGIONS", "PTBR").split(",")
DGRID = [int(x) for x in os.environ.get("LG_TLGT_DGRID", "0,1,2").split(",")]
MAXSTEPS = _int("LG_TLGT_MAXSTEPS", 15 if QUICK else 40)
PATIENCE = _int("LG_TLGT_PATIENCE", 6 if QUICK else 8)
NRUNS = _int("LG_TLGT_NRUNS", 3 if QUICK else 5)
_P0_ENV = os.environ.get("LG_TLGT_P0")
P0 = float(_P0_ENV) if _P0_ENV else None  # None -> seed growth at the real graph density
EDGE_FACTOR = _float("LG_TLGT_EDGE_FACTOR", 2.0)  # stop growth if edges > FACTOR*E_real
EDGE_TOL = _float("LG_TLGT_EDGE_TOL", 0.2)  # phase-1 gate: start GIC once E >= (1-tol)*E_real
SEED = _int("LG_TLGT_SEED", 12345)
FIT_MODE = os.environ.get("LG_TLGT_FIT", "mle")  # "mle" (logistic) | "gic" (min-GIC search)
GIC_FIT_SEEDS = _int("LG_TLGT_GIC_SEEDS", 2)  # growth seeds averaged per GIC objective eval
GIC_FIT_MAXITER = _int("LG_TLGT_GIC_MAXITER", 40)  # Nelder-Mead iteration cap

FAMILIES = ["TLG", "ER", "BA", "WS", "KR", "GRG", "SBM"]


def _load_region(path):
    G = nx.read_edgelist(path, comments="#", nodetype=int)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def _metrics(G):
    """Structural metrics of a graph (NaN-safe assortativity)."""
    try:
        assort = float(nx.degree_assortativity_coefficient(G))
    except Exception:
        assort = float("nan")
    return dict(edges=G.number_of_edges(),
                density=float(nx.density(G)),
                clustering=float(nx.average_clustering(G)),
                assortativity=assort)


def closed_form_params(G):
    n, E = G.number_of_nodes(), G.number_of_edges()
    kbar = 2 * E / n
    p_er = 2 * E / (n * (n - 1))
    m_ba = max(1, round(E / n))
    k_ws = max(2, int(round(kbar)))
    if k_ws % 2 == 1:
        k_ws += 1
    C_obs = nx.average_clustering(G)
    if k_ws > 2:
        C0 = 3 * (k_ws - 2) / (4 * (k_ws - 1))
        p_ws = 0.0 if C_obs >= C0 else 1.0 - (C_obs / C0) ** (1.0 / 3.0)
    else:
        p_ws = 0.0
    d_kr = int(round(kbar))
    if (n * d_kr) % 2 == 1:
        d_kr -= 1
    r_grg = math.sqrt(kbar / (math.pi * (n - 1)))
    return dict(n=n, E=E, ER=p_er, BA=m_ba, WS_k=k_ws, WS_p=p_ws, KR=d_kr, GRG=r_grg)


def _baseline_generators(G, cf):
    """name -> (gen_fn(seed)->nx.Graph, n_params)."""
    n = cf["n"]
    return {
        "ER": (lambda s: nx.erdos_renyi_graph(n, float(np.clip(cf["ER"], 1e-6, 1)), seed=s), 1),
        "BA": (lambda s: nx.barabasi_albert_graph(n, int(np.clip(cf["BA"], 1, n - 1)), seed=s), 1),
        "WS": (lambda s: nx.watts_strogatz_graph(n, cf["WS_k"], cf["WS_p"], seed=s), 2),
        "KR": (lambda s: nx.random_regular_graph(int(np.clip(cf["KR"], 0, n - 1)), n, seed=s), 1),
        "GRG": (lambda s: nx.random_geometric_graph(n, float(np.clip(cf["GRG"], 1e-3, 1.5)), seed=s), 1),
        "SBM": (lambda s: generate_sbm_from_real(G, seed=s)[0], None),  # n_params resolved below
    }


def _grow_score(n, d, sigma, alpha, e_real, real_den, scorer, seed=SEED):
    """Grow a TLG (seeded at the real graph's density) and score it against the real
    graph in two phases. Phase 1 (edge gate): reach E >= (1-EDGE_TOL)*E_real. Phase 2
    (GIC patience): monitor the spectral distance every step, keep the lowest-GIC
    iteration, early-stop after PATIENCE non-improving steps. Guard: stop if
    E > EDGE_FACTOR*E_real (a degenerate depth blows past the gate and never scores).
    Seeding at the real density works for both sparse and dense targets (a fixed sparse
    seed overshoots the gate in one step on dense graphs)."""
    best = {"dist": float("inf"), "adj": None, "step": -1}
    trace, no_improve = [], 0
    e_gate = (1.0 - EDGE_TOL) * e_real
    e_cap = EDGE_FACTOR * e_real
    p0 = P0 if P0 is not None else e_real / (n * (n - 1) / 2.0)
    phase = {"two": False, "start": -1}

    def cb(step, a):
        nonlocal no_improve
        e = a.sum() / 2.0
        if e > e_cap:                       # past the useful monitoring window -> stop
            return True
        if not phase["two"]:
            if e < e_gate:                  # phase 1: still growing toward E_real
                return False
            phase["two"] = True
            phase["start"] = step
        den, _ = scorer.compute_spectral_density(nx.from_numpy_array(a))
        dist = float(entropy(real_den + 1e-10, den + 1e-10))
        trace.append((step, dist, int(e)))
        if dist < best["dist"] - 1e-12:
            best.update(dist=dist, adj=a.copy(), step=step)
            no_improve = 0
        else:
            no_improve += 1
        return no_improve >= PATIENCE

    grow_graph(n, d=d, sigma=sigma, alpha=alpha, n_steps=MAXSTEPS, seed=seed,
               p0=p0, store_snapshots=False, record_design=False, step_callback=cb)
    return dict(dist=best["dist"], adj=best["adj"], step=best["step"],
                phase2_start=phase["start"], trace=trace)


def _growth_stable(n, d, sigma, alpha, e_real, seed=SEED, probe_steps=15):
    """Cheap degeneracy probe: grow from the real-density seed and flag the depth as
    degenerate if its chain approaches a dense graph (density > 0.5 — the ERGM blow-up
    toward the complete graph). No GIC is computed (cheap), and a degenerate depth stops
    the probe early. This distinguishes a genuinely degenerate depth from a stable one
    that merely settles somewhat above E_real."""
    p0 = P0 if P0 is not None else e_real / (n * (n - 1) / 2.0)
    bad = {"flag": False}

    def cb(step, a):
        if a.sum() / (n * (n - 1)) > 0.5:
            bad["flag"] = True
            return True
        return False

    grow_graph(n, d=d, sigma=sigma, alpha=alpha, n_steps=probe_steps, seed=seed,
               p0=p0, store_snapshots=False, record_design=False, step_callback=cb)
    return not bad["flag"]


def _gic_objective(sigma, alpha, d, n, e_real, real_den, scorer):
    """Mean GIC over GIC_FIT_SEEDS growth seeds (large penalty if degenerate/unscored)."""
    dists = []
    for k in range(GIC_FIT_SEEDS):
        gr = _grow_score(n, d, sigma, alpha, e_real, real_den, scorer, seed=SEED + 101 * k)
        if gr["adj"] is not None:
            dists.append(gr["dist"])
    if not dists:
        return 1e6
    return 2.0 * float(np.mean(dists)) + 2.0 * TLG_N_PARAMS


def fit_tlg(adj, real_den, scorer):
    """For each candidate d: get (sigma, alpha) — by logistic regression (FIT_MODE=mle)
    or by minimizing the GIC with a warm-started Nelder-Mead search (FIT_MODE=gic) — then
    grow + edge-gated GIC-monitor. Select d (and its iteration) by the LOWEST GIC;
    n_params=2 is constant across d, so this is the best joint (d, iteration) spectral
    fit, and a degenerate depth (blows past the edge guard, never scores) is excluded."""
    n = adj.shape[0]
    e_real = int(adj.sum() // 2)
    per_d = {}
    for d in DGRID:
        t_d = time.perf_counter()
        X, labels = build_pair_dataset(adj, d=d, mode="bounded", layer2=True)
        f = fit_growth_params(X, labels)            # MLE (also the gic warm start)
        sigma, alpha = f["sigma"], f["alpha"]
        # Degeneracy probe: exclude depths whose generator blows up toward the complete
        # graph (density > 0.5). A stable depth that merely settles a bit above E_real
        # is kept.
        if not _growth_stable(n, d, sigma, alpha, e_real):
            per_d[d] = dict(aic=f["aic"], sigma=sigma, alpha=alpha, gic=float("inf"),
                            dist=float("inf"), adj=None, step=-1, phase2_start=-1,
                            trace=[], stable=False)
            log(f"    [d={d}] MLE σ={sigma:.3f} α={alpha:.4f} aic={f['aic']:.0f} -> "
                f"DEGENERATE (density>0.5) -> excluded ({time.perf_counter()-t_d:.1f}s)")
            continue
        gr = _grow_score(n, d, sigma, alpha, e_real, real_den, scorer)  # MLE growth
        mle_kl = gr["dist"] if gr["adj"] is not None else float("inf")
        log(f"    [d={d}] MLE σ={sigma:.3f} α={alpha:.4f} aic={f['aic']:.0f} -> "
            f"growth {'KL=%.4f @step %d' % (mle_kl, gr['step']) if gr['adj'] is not None else 'no-score (gate not reached)'}")
        # GIC search: only refine depths that already SCORE at the MLE warm start
        # (a depth that can't reach the edge gate at the MLE can't be warm-started, and
        # searching it would burn evals on full no-score growths).
        if FIT_MODE == "gic" and gr["adj"] is not None:
            from scipy.optimize import minimize
            nev = {"n": 0}

            def obj(x):
                nev["n"] += 1
                v = _gic_objective(x[0], x[1], d, n, e_real, real_den, scorer)
                if nev["n"] % 10 == 0:
                    log(f"      gic-search d={d}: eval {nev['n']} σ={x[0]:.3f} "
                        f"α={x[1]:.4f} GIC={v:.4f}")
                return v

            log(f"    [d={d}] gic-search (Nelder-Mead, warm start MLE, "
                f"<= {GIC_FIT_MAXITER} iters x {GIC_FIT_SEEDS} seeds) ...")
            res = minimize(obj, x0=[sigma, alpha], method="Nelder-Mead",
                           options={"maxiter": GIC_FIT_MAXITER, "xatol": 1e-2, "fatol": 1e-3})
            gr2 = _grow_score(n, d, float(res.x[0]), float(res.x[1]),
                              e_real, real_den, scorer)
            if gr2["adj"] is not None and gr2["dist"] < gr["dist"]:  # keep if better
                sigma, alpha, gr = float(res.x[0]), float(res.x[1]), gr2
                log(f"    [d={d}] gic-fit improved: σ={sigma:.3f} α={alpha:.4f} "
                    f"KL={gr['dist']:.4f} (MLE was {mle_kl:.4f}) in {nev['n']} evals")
            else:
                log(f"    [d={d}] gic-fit kept MLE (KL={mle_kl:.4f}) after {nev['n']} evals")
        gic = (2.0 * gr["dist"] + 2.0 * TLG_N_PARAMS) if gr["adj"] is not None else float("inf")
        per_d[d] = dict(aic=f["aic"], sigma=sigma, alpha=alpha, gic=gic, stable=True, **gr)
        log(f"    [d={d}] done ({time.perf_counter() - t_d:.1f}s)  "
            f"GIC={gic if np.isfinite(gic) else float('nan'):.4f}")

    # Select min-GIC among stable depths that scored (reached the edge gate).
    pool = [d for d in DGRID if per_d[d]["stable"] and per_d[d]["adj"] is not None]
    d_hat = min(pool, key=lambda d: per_d[d]["gic"]) if pool else min(DGRID)
    r = per_d[d_hat]
    graph = nx.from_numpy_array(r["adj"]) if r["adj"] is not None else None
    return dict(gic=r["gic"], dist=r["dist"], n_params=TLG_N_PARAMS, d_hat=d_hat,
                sigma=r["sigma"], alpha=r["alpha"], best_step=r["step"],
                phase2_start=r["phase2_start"], trace=r["trace"], graph=graph,
                aic_by_d={d: per_d[d]["aic"] for d in DGRID},
                gic_by_d={d: per_d[d]["gic"] for d in DGRID},
                unstable_by_d={d: (not per_d[d]["stable"]) for d in DGRID})


def baseline_gic(G, name, gen_fn, n_params, real_den, scorer, seed):
    dens, rep = [], None
    for r in range(NRUNS):
        try:
            g = gen_fn(seed + r)
        except Exception:
            continue
        if rep is None:
            rep = g
        dens.append(scorer.compute_spectral_density(g)[0])
    if not dens:
        return None
    avg = np.mean(dens, axis=0)
    dist = float(entropy(real_den + 1e-10, avg + 1e-10))
    return dict(gic=2.0 * dist + 2.0 * n_params, dist=dist, n_params=n_params, graph=rep)


def process_region(region):
    path = DATA_DIR / f"{region}_graph.edges"
    if not path.exists():
        print(f"  {region}: file not found at {path} — skipping")
        return None
    t0 = time.perf_counter()
    G = _load_region(path)
    adj = nx.to_numpy_array(G)
    n = G.number_of_nodes()
    real_m = _metrics(G)
    print(f"\n=== {region}: n={n} E={real_m['edges']} density={real_m['density']:.4f} "
          f"clustering={real_m['clustering']:.3f} assort={real_m['assortativity']:.3f} ===")

    scorer = GraphInformationCriterion(G, model="LG", dist="KL")
    real_den, _ = scorer.compute_spectral_density(G)

    rows = [dict(model="Real", n_params=np.nan, gic=np.nan, gic_fit=np.nan,
                 gic_penalty=np.nan, kl=np.nan, param="—", **real_m)]

    # --- TLG ---
    tlg = fit_tlg(adj, real_den, scorer)
    tlg_m = (_metrics(tlg["graph"]) if tlg["graph"] is not None
             else dict(edges=np.nan, density=np.nan, clustering=np.nan,
                       assortativity=np.nan))
    rows.append(dict(model="TLG", n_params=tlg["n_params"], gic=tlg["gic"],
                     gic_fit=2.0 * tlg["dist"], gic_penalty=2.0 * tlg["n_params"],
                     kl=tlg["dist"],
                     param=f"d={tlg['d_hat']}, σ={tlg['sigma']:.2f}, α={tlg['alpha']:.3f}",
                     **tlg_m))
    def _fmt_gic(k, v):
        if np.isfinite(v):
            return f"{k}:{v:.2f}"
        return f"{k}:{'deg' if tlg['unstable_by_d'][k] else 'no-score'}"
    gic_str = [_fmt_gic(k, v) for k, v in tlg['gic_by_d'].items()]
    aic_str = [f"{k}:{v:.0f}" for k, v in tlg['aic_by_d'].items()]
    print(f"  TLG[{FIT_MODE}]: d_hat={tlg['d_hat']} by GIC (per-d GIC {gic_str}; AIC {aic_str}) "
          f"sigma={tlg['sigma']:.3f} alpha={tlg['alpha']:.4f} | phase2@step "
          f"{tlg['phase2_start']} best step {tlg['best_step']}  KL={tlg['dist']:.4f}  "
          f"GIC={tlg['gic']:.4f}")

    # --- baselines (closed-form) ---
    cf = closed_form_params(G)
    gens = _baseline_generators(G, cf)
    cf_params = dict(ER=f"p={cf['ER']:.4f}", BA=f"m={cf['BA']}",
                     WS=f"k={cf['WS_k']}, p={cf['WS_p']:.3f}", KR=f"d={cf['KR']}",
                     GRG=f"r={cf['GRG']:.3f}", SBM="Louvain blocks")
    for name in ("ER", "BA", "WS", "KR", "GRG", "SBM"):
        gen_fn, n_params = gens[name]
        if name == "SBM":  # resolve true k(k+1)/2 from one fit
            _, n_params = generate_sbm_from_real(G, seed=SEED)
        res = baseline_gic(G, name, gen_fn, n_params, real_den, scorer, SEED)
        if res is None:
            print(f"  {name}: generation failed — skipping")
            continue
        rows.append(dict(model=name, n_params=res["n_params"], gic=res["gic"],
                         gic_fit=2.0 * res["dist"], gic_penalty=2.0 * res["n_params"],
                         kl=res["dist"], param=cf_params[name], **_metrics(res["graph"])))
        print(f"  {name}: n_params={res['n_params']} KL={res['dist']:.4f} GIC={res['gic']:.4f}")

    df = pd.DataFrame(rows)
    models = df[df["model"] != "Real"].copy()
    models = models.sort_values("gic")
    models["rank"] = range(1, len(models) + 1)
    df = pd.concat([df[df["model"] == "Real"], models], ignore_index=True)
    df["rank"] = df["rank"].astype("Int64")
    df["region"] = region

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["region", "model", "rank", "gic", "gic_fit", "gic_penalty", "kl",
            "n_params", "edges", "density", "clustering", "assortativity", "param"]
    df[cols].to_csv(OUT_DIR / f"{region}_table.csv", index=False)
    _plot_bar(df, region, OUT_DIR / f"{region}_gic_bar.png")
    _plot_trace(tlg["trace"], tlg["best_step"], region, OUT_DIR / f"{region}_tlg_trace.png")
    print(df[cols].to_string(index=False))
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    return df[cols]


# --------------------------------------------------------------------------- plots
CB = {"TLG": "#0072B2", "ER": "#E69F00", "BA": "#009E73", "WS": "#CC79A7",
      "KR": "#D55E00", "GRG": "#56B4E9", "SBM": "#000000"}


def _plot_bar(df, region, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = df[df["model"] != "Real"].sort_values("gic")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = range(len(m))
    ax.bar(x, m["gic_fit"], color=[CB.get(mm, "#888") for mm in m["model"]],
           label="2·KL (fit)")
    ax.bar(x, m["gic_penalty"], bottom=m["gic_fit"], color="#cccccc",
           label="2·n_params (penalty)", edgecolor="white")
    ax.set_xticks(list(x)); ax.set_xticklabels(m["model"])
    ax.set_ylabel("GIC  (= 2·KL + 2·n_params)")
    ax.set_title(f"Twitch {region}: GIC by family (lower = better; stacked terms)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
    for i, (_, r) in enumerate(m.iterrows()):
        ax.text(i, r["gic"], f"{r['gic']:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def _plot_trace(trace, best_step, region, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not trace:
        return
    steps = [s for s, _, _ in trace]
    dists = [2.0 * d + 2.0 * TLG_N_PARAMS for _, d, _ in trace]  # GIC per iteration
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(steps, dists, "-o", color="#0072B2", ms=4, lw=1.6)
    bi = [i for i, (s, _, _) in enumerate(trace) if s == best_step]
    if bi:
        ax.scatter([best_step], [dists[bi[0]]], color="#D55E00", s=90, zorder=5,
                   edgecolor="white", label=f"min GIC (step {best_step})")
    ax.set_xlabel("TLG growth iteration (phase 2: edge count ≈ real)")
    ax.set_ylabel("GIC vs real graph")
    ax.set_title(f"Twitch {region}: TLG GIC per growth iteration (edge-gated, early-stopped)")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    print(f"TLG Twitch GIC  quick={QUICK} regions={REGIONS} dgrid={DGRID} "
          f"maxsteps={MAXSTEPS} patience={PATIENCE} nruns={NRUNS} seed={SEED}")
    if not DATA_DIR.exists():
        print(f"No twitch data under {DATA_DIR} (gitignored — place *_graph.edges there).")
        return
    all_tables = []
    for region in REGIONS:
        t = process_region(region)
        if t is not None:
            all_tables.append(t)
    if not all_tables:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_tables, ignore_index=True)
    combined.to_csv(OUT_DIR / "all_tables.csv", index=False)
    # rank summary across regions
    rk = combined[combined["model"] != "Real"].pivot_table(
        index="model", columns="region", values="rank", aggfunc="first")
    rk["mean_rank"] = rk.mean(axis=1)
    rk = rk.sort_values("mean_rank")
    rk.to_csv(OUT_DIR / "summary.csv")
    print("\n=== rank by region (mean rank, lower = better) ===")
    print(rk.to_string())
    print(f"\nWrote {OUT_DIR}/")


if __name__ == "__main__":
    main()
