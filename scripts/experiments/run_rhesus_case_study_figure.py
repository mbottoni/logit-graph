#!/usr/bin/env python3
"""Rhesus cerebral-cortex case-study figure (thesis Fig. 3.11), corrected.

Regenerates the node-link "visual comparison of the observed connectome and the best-fit graph
from each model" for the rhesus macaque cerebral cortex connectome, fixing the defects of the
old figure (figs2/animal/rhesus_graphs.png):

  * OLD: only 5 panels (Original/LG/ER/BA/WS) -> the actual winner GRG, plus KR and SBM, were
    missing.  NEW: all seven candidate models (LG, ER, BA, WS, KR, GRG, SBM) + the Original.
  * OLD: panels were labelled by GIC (LG 0.450), which does not match the case-study ranking
    (by KL) and disagrees with the KL table (LG 0.666).  NEW: every model panel is labelled by
    the **KL spectral divergence** (KL between the model's generated-graph adjacency spectral
    density and the observed one) -- the same metric the narrative ranks by.  GIC = 2*KL +
    2*|theta| is the *penalized* quantity; the un-penalized KL is what the ranking uses, so the
    old "GIC 0.450" and the table's "KL 0.666" are two different numbers for LG.
  * OLD: "Original GIC: nan" (a self-divergence).  NEW: the Original is the reference and shows
    only n / m, no divergence.
  * GRG is the best fit (lowest KL) and LG second; both are highlighted.
  * "LG" everywhere (not "TLG").

Everything is computed with the repo's own machinery (scripts/closedform/tlg_latent_gic_common.py
as C, run_tlg_twitch_gic as tw): the same loader, the same GraphInformationCriterion KL scorer,
the same baseline samplers, and the same ensemble-mean-density KL estimator used by the GIC sweep.
The LG fit is reconstructed from the cached parameters for this exact network
(runs/tlg_latent_connectome_gic/cache/rhesus_cerebral.cortex_1.json: d=1, dist kernel, k=2) so we
do not re-run the (slow) hyperparameter search; it falls back to a full fit_tlg if the cache is
absent.  The recomputed KL values are asserted to reconcile with that cache.

Output under scripts/experiments/runs/rhesus_case_study/ (gitignored):
  rhesus_case_study.png / .pdf   the 8-panel figure
  rhesus_case_study_kl.csv       per-model KL, edge count, KL rank
  README.txt                     the GIC<->KL note for whoever copies the figure into the thesis

Run:  .venv/bin/python scripts/experiments/run_rhesus_case_study_figure.py
"""
from __future__ import annotations

import json
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

NET_ID = "rhesus_cerebral.cortex_1"
OUT = _here / "runs" / "rhesus_case_study"
LAYOUT_SEED = 7
KL_TOL = 1e-3                       # recomputed KL must match the cache within this

# Okabe-Ito accents for the two best models (consistent with the GIC bar figure's palette).
HILITE = {"GRG": "#009E73", "LG": "#0072B2"}


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
    C.log(f"rhesus cerebral cortex: n={n} m={m}")
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
        hl = HILITE.get(name)
        _draw(ax, graph, color=(hl if hl else "#9ecae1"))
        if kl is None:
            ax.set_title(f"Original connectome\n$n = {n}$,  $m = {m}$",
                         fontsize=13, fontweight="bold")
        else:
            note = {1: "  ★ best fit", 2: "  (2nd)"}.get(rank[name], "")
            tcol = hl if hl else "black"
            ax.set_title(f"{name}{note}\nKL $= {kl:.3f}$   |   $m = {graph.number_of_edges()}$",
                         fontsize=13, fontweight="bold", color=tcol)
            if hl:                                  # frame the two best panels
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(hl); sp.set_linewidth(2.2)
                ax.patch.set_visible(False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    fig.suptitle("Rhesus macaque cerebral cortex: observed connectome vs. best-fit graph from "
                 "each model\n(node-link views, ranked by KL spectral divergence; lower is better)",
                 fontsize=16, y=0.99)
    fig.text(0.5, 0.015,
             "KL is the spectral-density divergence between each model's generated graph and the "
             "observed connectome — the goodness-of-fit term of GIC $= 2\\,\\mathrm{KL} + "
             "2\\,|\\theta|$ (the old figure's “GIC 0.450” for LG was the penalized quantity; "
             "its KL is 0.666). GRG fits best, LG second.",
             ha="center", fontsize=11)
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
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
        "Rhesus cerebral-cortex case study (thesis Fig. 3.11), corrected.\n\n"
        "Figure: rhesus_case_study.png / .pdf — node-link views of the observed connectome and the\n"
        "best-fit graph from each of the seven models, ranked by KL spectral divergence.\n\n"
        "GIC vs KL: the case study ranks models by KL (spectral-density divergence to the observed\n"
        "graph). KL is the goodness-of-fit term of the GIC: GIC = 2*KL + 2*|theta|, where |theta| is\n"
        "the model's parameter count. The figure reports the un-penalized KL so it matches the\n"
        "ranking; the old figure's 'GIC 0.450' for LG was the penalized quantity, while the KL is\n"
        "0.666. The Original panel is the reference and carries no divergence (the old 'GIC: nan').\n\n"
        "Ranking (lower KL = better): " +
        ", ".join(f"{f} {results[f]['kl']:.3f}" for f in ranked) + ".\n")

    C.log("\nKL ranking (lower = better):  " +
          "  ".join(f"{f}#{rank[f]} {results[f]['kl']:.3f}" for f in ranked))
    C.log(f"Wrote {OUT}/ (rhesus_case_study.png/.pdf, rhesus_case_study_kl.csv, README.txt)")


if __name__ == "__main__":
    main()
