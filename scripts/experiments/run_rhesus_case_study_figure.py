#!/usr/bin/env python3
"""Connectome case-study figure: node-link comparison of the observed connectome and each model's
best fit (default rhesus_brain_1, where LG wins), all seven models + Original labelled by KL spectral
divergence. Computed via tlg_latent_gic_common (reconstructs the cached LG fit; KL reconciled)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
_repo = _here.parents[1]
for p in (_repo / "src", _repo / "scripts" / "closedform"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import tlg_latent_gic_common as C   # noqa: E402  (loaders, KL scorer, samplers, ensemble KL)
import run_tlg_twitch_gic as tw     # noqa: E402  (closed_form_params, baseline gens, community feat)

# Which connectome to plot. Default: rhesus_brain_1, where LG is the best fit (lowest KL).
# Set LG_CASE_NET=rhesus_cerebral.cortex_1 for the thesis-Fig.-3.11 network (where GRG wins).
NET_ID = os.environ.get("LG_CASE_NET", "rhesus_brain_1")
OUT = _here / "runs" / "rhesus_case_study"
LAYOUT_SEED = 7
KL_TOL = 1e-3                       # recomputed KL must match the cache within this

DISPLAY = {"rhesus_brain_1": "Rhesus macaque brain",
           "rhesus_cerebral.cortex_1": "Rhesus macaque cerebral cortex"}

# Okabe-Ito accents applied to the two best models BY KL RANK (1 = best, 2 = runner-up).
RANK_HL = {1: "#009E73", 2: "#0072B2"}
RANK_NOTE = {1: "  ★ best fit", 2: "  (2nd)"}


def _net_display(nid):
    return DISPLAY.get(nid, nid.replace("_", " "))


def _lg_from_cache(G, scorer, real_den):
    """Reconstruct the LG best-fit graph + ensemble-KL from the cached winning config for this
    network (d=1, dist kernel, k=2). Falls back to a full fit_tlg() if the cache is missing."""
    cache_path = C._out_dir("connectome") / "cache" / f"{NET_ID}.json"
    if not cache_path.is_file():
        C.log(f"  (cache {cache_path} absent — running full fit_tlg)")
        fit = C.fit_tlg(G, scorer, real_den)
        return fit["kl"], fit["graph"], None
    cache = json.loads(cache_path.read_text())
    sel = cache["tlg_selected"]                       # {"d":1,"kernel":"dist","k":2}
    tr = next(t for t in cache["tlg_trace"]
              if t["d"] == sel["d"] and t["kernel"] == sel["kernel"] and t["k"] == sel["k"])
    n = G.number_of_nodes(); e_real = G.number_of_edges()
    rows, cols = np.triu_indices(n, k=1)
    Bc, _ = tw._community_feature(G, rows, cols, C.SEED, resolution=1.0)
    Bf, _ = tw._community_feature(G, rows, cols, C.SEED, resolution=C.FINE_RES)
    w, U = np.linalg.eigh(nx.to_numpy_array(G))
    L = C._latent_from_eig(w, U, sel["k"], rows, cols, sel["kernel"])
    x = (tr["alpha"], tr["gc"], tr["gf"], tr["lam"])
    kl, rep = C._ensemble_kl(x, n, rows, cols, Bc, Bf, L, e_real, scorer, real_den,
                             C.EVAL_SEEDS, sel["d"], want_graph=True)
    return kl, rep, tr["eval_kl"]


def compute():
    """Load the connectome and compute, for every model, its best-fit graph + KL (vs cache)."""
    G = C._graphml(C.DATA / "connectomes" / f"{NET_ID}.graphml")
    n, m = G.number_of_nodes(), G.number_of_edges()
    C.log(f"{_net_display(NET_ID)} ({NET_ID}): n={n} m={m}")
    scorer = C.GraphInformationCriterion(G, model="LG", dist="KL")
    real_den, _ = scorer.compute_spectral_density(G)

    cache = {}
    cpath = C._out_dir("connectome") / "cache" / f"{NET_ID}.json"
    if cpath.is_file():
        cache = {f["model"]: f["kl"] for f in json.loads(cpath.read_text())["families"]}

    results = {}   # model -> dict(kl, graph)
    kl_lg, rep_lg, cached_lg = _lg_from_cache(G, scorer, real_den)
    results["LG"] = dict(kl=kl_lg, graph=rep_lg)
    if cached_lg is not None and abs(kl_lg - cached_lg) > KL_TOL:
        raise SystemExit(f"LG KL {kl_lg:.4f} disagrees with cache {cached_lg:.4f} — investigate.")

    cf = tw.closed_form_params(G)
    gens = tw._baseline_generators(G, cf)
    for fam in ("ER", "BA", "WS", "KR", "GRG", "SBM"):
        gen_fn, npar = gens[fam]
        if fam == "SBM":
            _, npar = C.generate_sbm_from_real(G, seed=C.SEED)
        res = C.baseline_gic(G, gen_fn, npar, real_den, scorer)
        if res is None:
            raise SystemExit(f"{fam}: generation failed")
        results[fam] = dict(kl=res["kl"], graph=res["graph"])
        ck = cache.get("TLG" if fam == "LG" else fam)
        if ck is not None and abs(res["kl"] - ck) > KL_TOL:
            raise SystemExit(f"{fam} KL {res['kl']:.4f} disagrees with cache {ck:.4f} — investigate.")

    for fam in results:
        cm = cache.get("TLG" if fam == "LG" else fam)
        tag = "" if cm is None else f"  (cache {cm:.4f}; OK)"
        C.log(f"  {fam:4s} KL={results[fam]['kl']:.4f}  "
              f"m={results[fam]['graph'].number_of_edges()}{tag}")
    return G, results


def _draw(ax, graph, color="#9ecae1", edgecolor="#7f7f7f"):
    pos = nx.spring_layout(graph, seed=LAYOUT_SEED)
    nx.draw_networkx_edges(ax=ax, G=graph, pos=pos, edge_color=edgecolor, width=0.35, alpha=0.5)
    nx.draw_networkx_nodes(ax=ax, G=graph, pos=pos, node_color=color, node_size=42,
                           linewidths=0.4, edgecolors="white")
    ax.set_axis_off()


def plot(G, results):
    n, m = G.number_of_nodes(), G.number_of_edges()
    ranked = sorted(results, key=lambda f: results[f]["kl"])     # GRG, LG, SBM, ER, KR, BA, WS
    rank = {fam: i + 1 for i, fam in enumerate(ranked)}
    panels = [("Original", G, None)] + [(fam, results[fam]["graph"], results[fam]["kl"])
                                        for fam in ranked]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5))
    for ax, (name, graph, kl) in zip(axes.flat, panels):
        hl = None if kl is None else RANK_HL.get(rank[name])
        _draw(ax, graph, color=(hl if hl else "#9ecae1"))
        if kl is None:
            ax.set_title(f"Original connectome\n$n$ (nodes) $= {n}$,  $m$ (edges) $= {m}$",
                         fontsize=13, fontweight="bold")
        else:
            note = RANK_NOTE.get(rank[name], "")
            ax.set_title(f"{name}{note}\nKL $= {kl:.3f}$   |   "
                         f"$m$ (edges) $= {graph.number_of_edges()}$",
                         fontsize=13, fontweight="bold", color=(hl if hl else "black"))
            if hl:                                  # frame the two best panels
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(hl); sp.set_linewidth(2.2)
                ax.patch.set_visible(False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(f"{_net_display(NET_ID)}: observed connectome vs. best-fit graph from each model"
                 "\n(node-link views, ranked by KL spectral divergence; lower is better)",
                 fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"rhesus_case_study.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return ranked, rank


def main():
    G, results = compute()
    ranked, rank = plot(G, results)

    import csv
    with open(OUT / "rhesus_case_study_kl.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "kl", "edges", "kl_rank", "is_best"])
        for fam in ranked:
            wr.writerow([fam, f"{results[fam]['kl']:.6f}",
                         results[fam]["graph"].number_of_edges(), rank[fam],
                         "yes" if rank[fam] == 1 else ""])
    (OUT / "README.txt").write_text(
        f"Connectome case-study figure — {_net_display(NET_ID)} ({NET_ID}).\n\n"
        "Figure: rhesus_case_study.png / .pdf — node-link views of the observed connectome and the\n"
        "best-fit graph from each of the seven models, ranked by KL spectral divergence.\n\n"
        "GIC vs KL: models are ranked by KL (spectral-density divergence to the observed graph). KL\n"
        "is the goodness-of-fit term of the GIC: GIC = 2*KL + 2*|theta|, where |theta| is the\n"
        "model's parameter count. The figure reports the un-penalized KL so it matches the ranking.\n"
        "The Original panel is the reference and carries no divergence.\n\n"
        f"Best fit: {ranked[0]} (KL {results[ranked[0]]['kl']:.3f}); runner-up: {ranked[1]} "
        f"(KL {results[ranked[1]]['kl']:.3f}).\n"
        "Ranking (lower KL = better): " +
        ", ".join(f"{f} {results[f]['kl']:.3f}" for f in ranked) + ".\n\n"
        "Set LG_CASE_NET to another connectome id to regenerate for it; "
        "LG_CASE_NET=rhesus_cerebral.cortex_1 reproduces the thesis-Fig.-3.11 network (GRG best).\n")

    C.log("\nKL ranking (lower = better):  " +
          "  ".join(f"{f}#{rank[f]} {results[f]['kl']:.3f}" for f in ranked))
    C.log(f"Wrote {OUT}/ (rhesus_case_study.png/.pdf, rhesus_case_study_kl.csv, README.txt)")


if __name__ == "__main__":
    main()
