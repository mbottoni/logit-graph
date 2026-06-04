# Closed-form baselines vs grid (Twitter ego networks) — findings

Experiment: `run_twitter_closedform.py` (Twitter twin of the gplus run). Score
ER/BA/WS/KR/GRG against SNAP Twitter ego networks by spectral GIC
(2·KL + 2·n_params), comparing **closed-form moment estimators** vs the current
**fixed-interval grid**, alongside LG fit by **burn-in + ensemble mean** of the
spectral density.

Config: 30 ego nets sampled (seeded) from the 857 in the window 50 ≤ n ≤ 300
(of 973 total), `n_runs=5`, `grid_points=5`, LG d∈{0,1,2}, seed 12345.
Undirected, self-loop-free largest connected component.

Sampled spread: n 54–240, density 0.051–0.392 (median 0.150); 17% above the ER
grid cap, none below 0.01 — i.e. **mostly inside the grid box**.

## Aggregate (mean GIC across 30 ego nets, lower = better)

| family | grid | closed-form | Δ(grid−cf) | cf wins |
|---|---|---|---|---|
| ER | **3.591** | 4.729 | −1.14 | 13% |
| BA | **3.281** | 4.438 | −1.16 | 13% |
| WS | **4.678** | 8.691 | −4.01 | 0% |

Closed-form only: KR 4.641, GRG **3.816**, and **LG 3.286**.

Mean rank (LG + closed-form baselines): **LG 1.50**, **GRG 2.50**, BA 3.33,
KR 3.70, ER 3.97, WS 6.00.

## Conclusion — Twitter behaves like gplus

Twitter ego networks are mid-density and clustered (like gplus), so their
densities sit **inside the grid's [0.01, 0.25] box**. The result is the gplus
pattern, not the connectome one:

- **Closed-form is worse for every family** (ER −1.14, BA −1.16, WS −4.01;
  cf wins only 13% / 13% / 0%). With the optimum inside the box, the grid
  reaches it and the density-matched closed form overshoots — catastrophically
  for WS (density-matched k overshoots into a near-lattice spectrum).
- **LG is the best model** (mean GIC 3.286, rank 1.50), essentially tied with the
  best grid baseline (BA-grid 3.281) and ahead of ER-grid (3.591) / WS-grid
  (4.678). LG selects d=1 for 26/30 nets.
- **GRG (closed-form) is the strongest baseline** (rank 2.50) — consistent with
  every other dataset, and still missing from the pipeline's
  `["ER","WS","BA","SBM"]` set.

## Cross-dataset picture (now four datasets)

| dataset | density range | closed-form vs grid | best model |
|---|---|---|---|
| gplus (ego) | 0.03–0.41 | grid wins | LG (GRG close) |
| **twitter (ego)** | **0.05–0.39** | **grid wins** | **LG (BA-grid tied)** |
| animal connectomes | 0.003–0.71 | mixed / cf competitive | LG (GRG 2nd) |
| human connectomes | 0.09–0.54 | cf competitive (BA wins) | GRG, then LG |

The verdict is consistent: **closed-form vs grid is decided by whether the data's
density lands inside the fixed grid interval.** Mid-density ego networks (gplus,
twitter) → grid wins; connectomes with extreme/spread densities → closed-form
becomes competitive-to-better. Across all four, **LG (scored at convergence) is
the best or near-best standard model**, and **GRG is a consistently strong
baseline the pipeline omits.** The robust fix remains data-adaptive parameter
intervals (what the closed form approximates) plus adding GRG as a baseline.
