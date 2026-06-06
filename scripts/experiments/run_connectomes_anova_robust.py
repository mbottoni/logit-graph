#!/usr/bin/env python3
"""Connectomes ANOVA on sigma-hat with a dyadic-cluster-robust SE.

Honest replacement for the ``extract_sigma_estimation`` "Table 2" of
``notebooks/connectomes_datasets/16-1-analysis.ipynb``, which bootstrapped 80%
of the edges *with replacement* from each fixed graph and ran ``f_oneway``. For a
single observed connectome there is no genuine replication, so that within-graph
variance has no sampling interpretation (its p-value is an artifact of the
bootstrap design, dial-able by subsample size).

Instead, per connectome we fit the offset-logit sigma-hat on the full graph at
its AIC-optimal d in {0,1} and attach the dyadic-cluster-robust (sandwich) SE
(Aronow-Samii-Assenova 2015, via logit_graph.robust_se). We then test equality
of sigma across connectomes with an omnibus Wald chi^2(k-1) and all C(k,2)
pairwise Wald z-tests (Bonferroni + BH-FDR). The SE has a real sampling
interpretation and is not dial-able by subsample size.

Caveat: connectomes are fit at per-graph optimal d, so sigma is not strictly
like-for-like across d=0 vs d=1 graphs (reported in the output note).

Mirrors scripts/experiments/run_twitch_anova_robust.py; reuses the GraphML
largest-CC loader convention from scripts/closedform/run_connectomes_closedform.py.
Reproducible: fixed seed (LG_CAR_SEED), BLAS threads pinned to 1. Writes only
under runs/connectomes_anova_robust/.

Env knobs (all optional):
  LG_CAR_SEED (12345)      LG_CAR_QUICK (0 -> full; 1 -> smoke on small graphs)
  LG_CAR_D_MAX (1)         LG_CAR_MIN_NODES (20)   LG_CAR_MAX_NODES (2000)
  LG_CAR_MAX_NETS (all)

  make anova-connectomes-robust         full run
  make anova-connectomes-robust-quick   smoke on the small connectomes
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import networkx as nx
import pandas as pd
from scipy.stats import norm, chi2
from statsmodels.stats.multitest import multipletests

_here = Path(__file__).resolve().parent
_repo_root = _here.parents[1]
_src = _repo_root / "src"
for p in (_src, _here):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

warnings.filterwarnings("ignore")

from logit_graph.robust_se import dyadic_robust_sigma_se, select_d_aic  # noqa: E402


def _int(env, default):
    raw = os.environ.get(env)
    return int(raw) if raw is not None else default


QUICK = os.environ.get("LG_CAR_QUICK", "0") == "1"
SEED = _int("LG_CAR_SEED", 12345)
# d candidates default to {0,1}: the d>=2 offset feature is O(n^3) at full n
# (per-pair multi-hop BFS over ~n^2/2 pairs), infeasible on the larger
# connectomes (n up to ~1800). Raise LG_CAR_D_MAX to search wider.
D_MAX = _int("LG_CAR_D_MAX", 1)
MIN_NODES = _int("LG_CAR_MIN_NODES", 20)
MAX_NODES = _int("LG_CAR_MAX_NODES", 300 if QUICK else 2000)
MAX_NETS = _int("LG_CAR_MAX_NETS", 4 if QUICK else 10_000)
FEATURE_MODE = "incremental"


def _load_graphml(path):
    """Undirected, unweighted (binary) largest connected component, relabeled."""
    G = nx.read_graphml(path)
    G = nx.Graph(G)  # collapse multi-edges + direction
    G.remove_edges_from(nx.selfloop_edges(G))
    cc = max(nx.connected_components(G), key=len)
    return nx.convert_node_labels_to_integers(G.subgraph(cc).copy())


def _abbrev(names):
    """Compact, unique short codes for the connectome stems (for the tex matrix)."""
    out, seen = [], {}
    for name in names:
        base = name.replace("_graph", "").replace("_neural", "")
        base = base.split(".")[0][:8] if "." in base else base[:8]
        code = base
        k = seen.get(base, 0)
        if k:
            code = f"{base[:6]}{k+1}"
        seen[base] = k + 1
        out.append(code)
    return out


# ---------------------------------------------------------------------------
# Tests (mirror run_twitch_anova_robust.py)
# ---------------------------------------------------------------------------

def _omnibus_wald(sigmas, ses):
    """Cochran's Q / fixed-effect equality test: H0 all sigma equal. ~ chi2(k-1)."""
    w = 1.0 / np.asarray(ses) ** 2
    s = np.asarray(sigmas)
    s_bar = float(np.sum(w * s) / np.sum(w))
    Q = float(np.sum(w * (s - s_bar) ** 2))
    dof = len(sigmas) - 1
    return Q, dof, float(chi2.sf(Q, dof))


def _pairwise_wald(names, sigmas, ses):
    rows = []
    for a, b in combinations(range(len(names)), 2):
        z = (sigmas[a] - sigmas[b]) / math.sqrt(ses[a] ** 2 + ses[b] ** 2)
        p = 2.0 * norm.sf(abs(z))
        rows.append({"name_i": names[a], "name_j": names[b], "z": z, "p_raw": p})
    df = pd.DataFrame(rows)
    df["p_bonf"] = multipletests(df["p_raw"], method="bonferroni")[1]
    df["p_fdr"] = multipletests(df["p_raw"], method="fdr_bh")[1]
    return df


def _write_tex(pdf, names, codes, out_path):
    """Lower-triangular Bonferroni p-value matrix; bold if significant at 0.05."""
    code_of = dict(zip(names, codes))
    pmat = {}
    for _, row in pdf.iterrows():
        pmat[(row["name_i"], row["name_j"])] = row["p_bonf"]
        pmat[(row["name_j"], row["name_i"])] = row["p_bonf"]
    disp = [code_of[n] for n in names]
    lines = [r"% legend: " + ", ".join(f"{c}={n}" for c, n in zip(codes, names)),
             r"\begin{tabular}{l" + "c" * (len(names) - 1) + "}", r"\toprule",
             " & " + " & ".join(disp[1:]) + r" \\", r"\midrule"]
    for ri in range(1, len(names)):
        cells = []
        for ci in range(len(names) - 1):
            if ci < ri:
                pv = pmat[(names[ri], names[ci])]
                cells.append(f"$\\mathbf{{{pv:.1e}}}$" if pv < 0.05 else f"${pv:.1e}$")
            else:
                cells.append("")
        lines.append(f"{disp[ri]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir = _repo_root / "data" / "connectomes"
    files = sorted(data_dir.glob("*.graphml"))
    if not files:
        print(f"No .graphml files under {data_dir} — connectome data is gitignored; "
              f"place the .graphml files there first.")
        return
    out_dir = _here / "runs" / "connectomes_anova_robust"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"connectomes dyadic-robust ANOVA  seed={SEED}  quick={QUICK}  "
          f"window=[{MIN_NODES},{MAX_NODES}]  max_nets={MAX_NETS}  d_candidates=0..{D_MAX}")

    loaded = []
    for f in files:
        try:
            G = _load_graphml(f)
        except Exception as e:
            print(f"  skip {f.stem}: {e}")
            continue
        loaded.append((f.stem, G))
    loaded.sort(key=lambda t: t[1].number_of_nodes())

    rows = []
    picked = 0
    for name, G in loaded:
        if picked >= MAX_NETS:
            break
        n = G.number_of_nodes()
        if not (MIN_NODES <= n <= MAX_NODES):
            continue
        picked += 1
        t0 = time.perf_counter()
        adj = nx.to_numpy_array(G, weight=None)  # binarize (connectomes are weighted)
        m = int(np.triu(adj).sum())
        d_hat, sigma_hat, offsets, labels, aic_by_d = select_d_aic(
            adj, list(range(D_MAX + 1)), feature_mode=FEATURE_MODE)
        se_r, se_n = dyadic_robust_sigma_se(offsets, labels, sigma_hat, n)
        rows.append(dict(name=name, n=n, edges=m, d_hat=d_hat, sigma_hat=sigma_hat,
                         se_robust=se_r, se_naive=se_n, aic_by_d=aic_by_d))
        ratio = se_r / se_n if se_n > 0 else float("nan")
        print(f"  {name[:28]:28s}  n={n:5d}  E={m:7d}  d_hat={d_hat}  "
              f"sigma_hat={sigma_hat:+.4f}  SE_robust={se_r:.4f}  SE_naive={se_n:.4f}  "
              f"(robust/naive={ratio:.1f}x)  [{time.perf_counter() - t0:.0f}s]")

    if len(rows) < 2:
        print("\nNeed >=2 connectomes in window.")
        return
    df = pd.DataFrame(rows)
    names = list(df["name"])
    codes = _abbrev(names)
    sigmas = df["sigma_hat"].to_numpy()
    ses = df["se_robust"].to_numpy()

    Q, dof, p_omni = _omnibus_wald(sigmas, ses)
    pdf = _pairwise_wald(names, sigmas, ses)

    df.drop(columns=["aic_by_d"]).to_csv(out_dir / "summary.csv", index=False)
    pdf.to_csv(out_dir / "pairwise.csv", index=False)
    _write_tex(pdf, names, codes, out_dir / "connectomes_pairwise_robust.tex")
    (out_dir / "results.json").write_text(json.dumps({
        "omnibus": {"Q": Q, "dof": dof, "p": p_omni},
        "connectomes": df.drop(columns=["aic_by_d"]).to_dict(orient="records"),
        "n_pairs": int(len(pdf)),
        "sig_raw": int((pdf["p_raw"] < 0.05).sum()),
        "sig_bonferroni": int((pdf["p_bonf"] < 0.05).sum()),
        "sig_fdr": int((pdf["p_fdr"] < 0.05).sum()),
    }, indent=2, default=float))

    print("\n" + "=" * 70)
    print(f"Omnibus Wald (H0: all sigma equal): Q={Q:.1f}  dof={dof}  p={p_omni:.3e}")
    print(f"Pairwise Wald: {len(pdf)} pairs  "
          f"(significant @0.05: raw={int((pdf['p_raw'] < 0.05).sum())}  "
          f"bonferroni={int((pdf['p_bonf'] < 0.05).sum())}  "
          f"fdr={int((pdf['p_fdr'] < 0.05).sum())})")
    d_set = sorted(set(df["d_hat"]))
    if len(d_set) == 1:
        cross_d = (f"All connectomes selected d={d_set[0]} (full-graph AIC), so the "
                   f"comparison is like-for-like at a single d.")
    else:
        cross_d = (f"Connectomes span d in {d_set}; sigma is not strictly like-for-like "
                   f"across different d, so read cross-d pairs with care.")
    print("\nNOTE: SEs are dyadic-cluster-robust (real sampling interpretation, not "
          f"dial-able by subsample size). {cross_d}")
    print(f"Wrote {out_dir}/ (summary.csv, pairwise.csv, "
          f"connectomes_pairwise_robust.tex, results.json)")


if __name__ == "__main__":
    main()
