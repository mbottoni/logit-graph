# Closed-form baseline estimators vs grid search (gplus) — findings

Quick experiment (`run_gplus_closedform.py`) on 8 gplus ego networks
(50 ≤ n ≤ 120), n_runs=3, grid_points=3, LG via the real pipeline (best of
d∈{0,1}). Baselines scored two ways and ranked by spectral GIC (2·KL + 2·n_params):

- **grid** — current method: fixed interval, 3 grid points, **selected by min GIC**
  (best case for the grid).
- **cf** — closed-form moment estimate (ER p=density [exact MLE]; BA m=round(E/n);
  WS k=2·round(E/n), p from clustering; KR d=round(2E/n); GRG r=√(k̄/(π(n−1)))).

## Aggregate (mean GIC across 8 nets, lower = better)

| family | grid | closed-form | Δ(grid−cf) | cf wins |
|---|---|---|---|---|
| ER | 4.363 | 4.330 | +0.03 | 38% |
| BA | 3.449 | **4.223** | −0.78 | 12% |
| WS | 4.829 | **10.823** | −5.99 | 0% |

Mean rank (LG + closed-form baselines): BA 2.63, LG 2.88, ER 3.00, KR 3.25,
GRG 3.25, WS 6.00.

## Conclusion — closed-form moment matching does NOT improve baseline GIC

The hypothesis from the prior analysis (that the fixed grid intervals are too
tight and that density-matched closed forms would lower baseline GIC) is
**refuted**:

- **ER**: a wash (the KL distance on the normalized-Laplacian histogram is
  nearly flat in p around the optimum).
- **BA**: closed-form is *worse* (density-matched m overshoots).
- **WS**: closed-form is *dramatically* worse (density-matched k≈2E/n, often
  18–44, produces a near-lattice spectrum nothing like the clustered real graph).

**Why:** GIC compares Laplacian *spectra*, not edge counts. For these clustered
ego networks (C≈0.3–0.7) the spectral-GIC-optimal baseline parameter sits at a
**lower density than the real graph**. Matching the first moment (density/degree)
overshoots into worse spectral territory. The grid winner is consistently a
*sparser* graph than the data (ER p≈0.13 regardless of real density; BA m≈4.5–8;
WS k≈8–10).

This also overturns the earlier "interval cap" concern. Network [7] has density
0.407 (above the ER cap 0.25), yet its GIC-optimal ER is still p≈0.13 — the cap
never binds harmfully because the optimum is at low density anyway.

## What this means for the comparison

The current grid already gives baselines their *best* GIC, so it is the **fair,
strong-baseline** choice. Swapping in closed-form estimators would *handicap* the
baselines (especially WS) and inflate LG's apparent advantage — the opposite of
what we'd want for a credible comparison.

The closed forms remain the correct *generative* estimators (MLE for ER; moment
match for the rest) — they are just not the GIC-minimizing parameters, because
spectral fit ≠ density fit.

## Legitimate improvements that DO hold (not closed-form)

1. **Selection rule** — the live pipeline (`model_selection.py:378`) picks the
   grid winner by min L2 density distance but reports a KL-based GIC. Selecting by
   GIC (as this experiment did) gives each baseline its true family minimum.
2. **Grid resolution / averaging** — a slightly finer grid centered on the
   low-density region plus larger n_runs would tighten baseline GIC further.

Closed-form estimation is a dead end for a spectral objective; the lever is the
search objective and resolution, not the point estimator.
