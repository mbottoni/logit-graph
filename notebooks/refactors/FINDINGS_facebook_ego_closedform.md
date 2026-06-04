# Closed-form baselines vs grid (Facebook ego networks) — findings

Experiment: `run_facebook_ego_closedform.py`. Score ER/BA/WS/KR/GRG against the
**10 SNAP Facebook ego networks** by spectral GIC (2·KL + 2·n_params), comparing
**closed-form moment estimators** vs the current **fixed-interval grid**,
alongside LG fit by **burn-in + ensemble mean** of the spectral density.

All 10 ego nets (n 40–1034 after largest-CC) are small enough for the LG chain,
so every one is scored. `n_runs=5`, `grid_points=5`, LG d∈{0,1,2}, seed 12345.
Densities span 0.034–0.282 — the small ego nets are mid-dense, the large ones
(n≥500) are sparse.

## Aggregate (mean GIC across 10 ego nets, lower = better)

| family | grid | closed-form | Δ(grid−cf) | cf wins |
|---|---|---|---|---|
| ER | **3.317** | 4.336 | −1.02 | 40% |
| BA | **3.573** | 4.352 | −0.78 | 30% |
| WS | **4.706** | 8.569 | −3.86 | 10% |

Closed-form only: KR 4.471, GRG **3.377**, and **LG 3.483**.

Mean rank (LG + closed-form baselines): **GRG 2.0**, **LG 2.5**, ER 3.2, BA 3.4,
KR 3.9, WS 6.0.

## Conclusion — a mix, split by ego-net size/density

Facebook ego sits between gplus (grid clearly wins) and the connectomes
(closed-form competitive), because the collection itself spans two regimes:

- **Small, mid-dense ego nets** (n<300, density 0.05–0.28: ids 3980, 698, 686,
  414, 0, 348) → **grid wins**, the gplus pattern.
- **Large, sparse ego nets** (n 532–1034, density 0.034–0.11: ids 3437, 1684,
  107, 1912) → **closed-form is competitive-to-better** (e.g. id 107: ER cf 2.621
  vs grid 2.698; BA cf 2.621 vs grid 2.837). Their density falls near/below the
  grid's lower useful range, so the coarse grid resolves the optimum poorly while
  density-matching lands on it.

That split is why the aggregate shows grid ahead on the mean yet closed-form
winning a sizeable fraction (ER 40%, BA 30%).

- **GRG (closed-form) is the best baseline overall** (rank 2.0, GIC 3.377) and the
  single best model on most of the larger ego nets — consistent with every other
  dataset, and still absent from the pipeline's `["ER","WS","BA","SBM"]` set.
- **LG is 2nd** (3.483) and the best *standard* model; it selects d=0 for the
  large sparse nets and d=1–2 for the small dense ones.
- **WS closed-form collapses** (density-matched k overshoots), worst family.

## Cross-dataset picture (now five datasets)

| dataset | density | closed-form vs grid | best model |
|---|---|---|---|
| gplus (ego) | 0.03–0.41 | grid wins | LG |
| twitter (ego) | 0.05–0.39 | grid wins | LG |
| **facebook (ego)** | **0.03–0.28** | **mixed (size-split)** | **GRG, then LG** |
| animal connectomes | 0.003–0.71 | mixed / cf competitive | LG (GRG 2nd) |
| human connectomes | 0.09–0.54 | cf competitive (BA wins) | GRG, then LG |

The verdict stays consistent: closed-form vs grid tracks whether the data's
density lands inside the fixed grid interval; **LG (scored at convergence) is
best-or-near-best everywhere**; and **GRG is a consistently strong baseline the
pipeline omits**.
