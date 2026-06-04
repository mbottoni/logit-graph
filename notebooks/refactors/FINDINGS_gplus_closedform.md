# Closed-form baseline estimators vs grid search (gplus) — findings

Experiment: `run_gplus_closedform.py`. Score ER/BA/WS/KR/GRG against gplus ego
networks by spectral GIC (2·KL + 2·n_params, KL on the 50-bin normalized-
Laplacian density), comparing **closed-form moment estimators** vs the current
**fixed-interval grid**, alongside LG fit via the real pipeline.

## Full run (matches Makefile `gic-gplus` preset)

- Window **50 ≤ n ≤ 300** → **17** ego networks (all of them).
- **n_runs=5, grid_points=5**, LG_ITER=2000, LG d-exploration **d ∈ {0,1,2}**.
- grid = fixed interval, 5 points, **selected by min GIC** (best case for grid).
- cf = closed-form: ER p=density (MLE); BA m=round(E/n); WS k=2·round(E/n),
  p from clustering; KR d=round(2E/n); GRG r=√(k̄/(π(n−1))).

### Aggregate (mean GIC across 17 nets, lower = better)

| family | grid | closed-form | Δ(grid−cf) | cf wins |
|---|---|---|---|---|
| ER | **3.328** | 4.508 | −1.18 | 0% |
| BA | **3.352** | 4.351 | −1.00 | 12% |
| WS | **4.979** | 9.081 | −4.10 | 0% |

Closed-form only (no grid): KR 4.499, GRG **3.840**, and **LG 3.999**.

Mean rank (LG + closed-form baselines): **GRG 2.12**, **LG 2.41**, BA 2.94,
KR 3.71, ER 3.82, WS 6.00.

## Conclusion 1 — closed-form does NOT improve baseline GIC (confirmed at scale)

At full scale with a finer grid and more averaging the verdict is sharper than
the pilot: closed-form is **worse for every searchable family** — ER −1.18,
BA −1.00, WS −4.10 — and **never wins for ER or WS** (0%).

**Why:** GIC compares Laplacian *spectra*, not edge counts. For these clustered
ego networks (C ≈ 0.3–0.73) the spectral-GIC-optimal baseline sits at a **much
lower density than the data**. The grid winner is consistently far sparser than
density-matching (ER p≈0.01–0.13 vs real 0.03–0.41; BA m≈3–8 vs cf 4–30; WS
k≈6–10 vs cf 8–60). Matching the first moment overshoots into worse spectral
territory — catastrophically for WS, whose density-matched k (up to 60) yields a
near-lattice spectrum nothing like the real graph.

The earlier "interval-cap" worry is also refuted: net [11] has density 0.407
(above the ER cap 0.25), yet its GIC-optimal ER is p≈0.19 — the cap never binds
harmfully because the optimum is at low density anyway.

→ Closed-form estimation is the right *generative* estimator (MLE for ER, moment
match otherwise) but the **wrong** estimator for a spectral objective. Using it
would *handicap* the baselines and inflate LG's apparent edge.

## Conclusion 2 — the finer grid + more runs DID lower grid baseline GIC

vs the pilot (gp=3, n_runs=3): ER grid 4.36 → **3.33**, BA 3.45 → **3.35**,
WS 4.83 → **4.98** (≈flat). Selecting by GIC over a finer low-density grid with
more averaging is the **real** lever for tightening baseline GIC — not the point
estimator.

## Two surprises worth a follow-up

1. **GRG (closed-form) is the strongest baseline here** — best mean rank (2.12)
   and lowest mean GIC (3.840), beating LG (3.999). The random geometric graph,
   with its single closed-form radius, captures the high-clustering/spatial
   structure of gplus ego nets well. GRG is **not** in the pipeline's baseline
   set (`["ER","WS","BA","SBM"]`); adding it would be a stronger comparison.

2. **Well-fit ER and BA are competitive with LG.** On mean GIC, grid-ER (3.33)
   and grid-BA (3.35) are *below* LG (4.00) on this window. The rank table favors
   LG only because it ranks LG against the *handicapped* closed-form ER/BA/WS.
   Caveat: LG here is capped at 2000 Gibbs iters (possibly undertrained at
   n≈300), so this is an observation to investigate, not a firm claim.

## Update — fair LG scoring (burn-in + ensemble mean), n_runs=5, grid=5

The first run scored LG via the pipeline's "best graph over a 2000-iter
trajectory". `diag_lg_convergence.py` showed that GIC is a **noisy stationary
walk**, that 2000 iters is mid-burn-in for large n and *pre*-burn-in for small
n (n=54 still at GIC≈13 at iter 2000), and that best-of-trajectory cherry-picks
the min of noise. We replaced it with the **same convention as the baselines**:
burn in (max(4000, 25n) steps) then average the spectral density over 5
post-burn-in snapshots → one GIC.

Effect — **LG improves** (the 2000 cap had been *under*training it):

| model | mean GIC |
|---|---|
| LG — best-of-2000 (old) | 3.999 |
| **LG — fair burn-in + ensemble** | **3.549** |
| ER grid | 3.328 |
| BA grid | 3.352 |
| GRG cf | 3.840 |
| WS grid | 4.979 |

LG now has the best mean rank vs the closed-form baselines (1.47) and beats GRG.
But against **properly grid-fit** ER/BA it is still a hair behind on mean GIC
(LG 3.55 vs ER 3.33, BA 3.35) — competitive, roughly tied, not a clear win.
Per-net it is mixed: LG wins the denser/mid nets, ER/BA win several sparse ones.

Caveat on fairness: baselines grid-search their parameter to *minimize* GIC,
while LG's σ is the MLE (not GIC-optimized) and only d∈{0,1,2} is searched — so
the baselines retain a small parameter-search advantage.

## Net takeaway

Closed-form is a dead end for improving baseline GIC. The levers that work are
(a) select the grid winner by GIC (the live pipeline uses L2 — `model_selection.py:378`),
(b) a finer low-density grid + larger n_runs, and (c) consider adding GRG as a
baseline. Worth re-checking LG convergence at n≈300 before drawing comparison
conclusions.
