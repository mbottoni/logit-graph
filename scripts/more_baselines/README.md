# Additional modern baseline models

Fits and scores a set of **newer** random-graph baselines, evaluated with the **same** spectral
estimator and structural-discrepancy metrics as the main latent-TLG sweep, so their numbers are
directly comparable. This experiment fits **only the new baselines** — it does not re-run
LG/TLG/ER/BA/WS/KR/GRG/SBM.

## Models

| Key | Model | Parameter fitting |
|-----|-------|-------------------|
| `ChungLu` | Expected-degree (Chung–Lu) | none — uses the observed degree sequence |
| `Config` | Configuration model | none — matches the degree sequence exactly (stub-matching) |
| `RDPG` | (Generalized) random dot-product graph | adjacency spectral embedding (same ASE as the TLG latent feature); dimension by an eigenvalue elbow |
| `DCSBM` | Degree-corrected SBM | Louvain blocks + node degree corrections (no graph-tool needed) |
| `HolmeKim` | BA growth + triad formation | `m` from density; triad prob `p` matched to observed clustering |
| `Hyperbolic` | Threshold hyperbolic (H2) graph | radial exponent from the degree power law; radius `R` fitted to the edge count |

## Metrics (per network, ensemble-averaged over the sweep's `EVAL_SEEDS`)

- `kl` — adjacency-spectral KL divergence to the observed graph (same as the main comparison).
- `ks_deg`, `d_clustering`, `d_assortativity` — degree-distribution KS distance and clustering /
  assortativity discrepancies (same helper as the main comparison PR).
- `mean_edges`, `edge_ratio` — sanity check that the fit reproduces the observed edge count.

## Usage

```bash
python scripts/more_baselines/run_more_baselines.py connectome
python scripts/more_baselines/run_more_baselines.py connectome twitch
python scripts/more_baselines/run_more_baselines.py all
```

Datasets: `twitch, facebook, twitter, gplus, connectome, human, arxiv` (same loaders/size ranges as
the main sweep). Runs are **cached per network** (resumable; re-running only computes missing
networks) and **reproducible** (fixed seeds). Results are written to
`runs/more_baselines_<dataset>/`:

- `cache/<id>.json` — per-network raw results (cache-versioned).
- `per_graph.csv` — one row per (network, model) with all metrics.
- `summary.csv` — per-model medians (`median_kl`, `median_ks_deg`, …), fit `success_rate`, and
  median `edge_ratio`.

## Notes

- The O(n²) probability-matrix models (`RDPG`, `DCSBM`) are skipped above `MB_MAX_DENSE_N`
  (default 3000) — the main datasets are capped well below this by the loaders.
- `RDPG` and `DCSBM` are more heavily parameterized than the LG model, so under the
  complexity-free KL ranking they get every advantage (the same caveat noted for the SBM).
- Generated `runs/` artifacts are gitignored.
