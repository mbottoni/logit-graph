# Closed-form baseline estimators vs grid search (animal connectomes) — findings

Experiment: `run_connectomes_closedform.py` (connectome twin of the gplus run).
Score ER/BA/WS/KR/GRG against the **18 animal connectomes** by spectral GIC
(2·KL + 2·n_params, KL on the normalized-Laplacian density; dense eigvalsh for
n≤500, deterministic KPM for larger n), comparing **closed-form moment
estimators** vs the current **fixed-interval grid**, alongside LG fit by
**burn-in + ensemble mean** of the spectral density (the fair scoring developed
for gplus).

Config: all 18 connectomes (n 29–1770 after largest-CC), `n_runs=5`,
`grid_points=5`, LG d∈{0,1,2}, seed 12345. Graphs loaded from GraphML as
undirected, **unweighted (binary)** largest connected component.

## Aggregate (mean GIC across 18 connectomes, lower = better)

| family | grid | closed-form | Δ(grid−cf) | cf wins |
|---|---|---|---|---|
| ER | 4.009 | **3.807** | +0.20 | 39% |
| BA | **3.829** | 4.031 | −0.20 | 33% |
| WS | **5.642** | 6.796 | −1.15 | 28% |

Closed-form only: KR 4.766, GRG **4.179**, and **LG 3.595**.

Mean rank (LG + closed-form baselines): **LG 2.22**, **GRG 2.50**, ER 2.89,
KR 3.61, BA 3.94, WS 5.83.

## Conclusion — the verdict flips vs gplus, and it's about the grid interval

On gplus (ego nets, densities 0.03–0.41) closed-form was clearly *worse* for
every family. On connectomes it is **competitive — ER is even slightly better
on average** (Δ +0.20), and BA/WS lose by far smaller margins than on gplus
(BA −0.20 vs −1.00; WS −1.15 vs −4.10).

**Why:** the connectome densities span **0.003 → 0.713** — far wider than gplus
— and frequently fall **outside the grid's fixed ER interval [0.01, 0.25]**.
When the true density is outside the interval the grid is capped away from the
optimum and the density-matched closed form wins:

- **Too dense (> 0.25):** `mouse_brain_1` density 0.713 → ER grid capped at
  p=0.25 gives 3.536, closed-form p=0.713 gives **2.658**. Same for the dense
  rhesus cortices.
- **Too sparse (< 0.01):** `kasthuri_graph_v4` density 0.003 → grid's *lower*
  bound 0.01 forces an over-dense ER (6.327) vs closed-form **4.004**;
  `drosophila_medulla_1` 0.006 → grid 2.897 vs cf **2.337**.
- **Inside the interval (mid density):** the grid still wins, exactly as on gplus.

This is the **interval-cap effect** hypothesized in the original gplus analysis.
It did *not* bite on gplus (optimum was at low density, inside the box) but it
**does** bite here because connectome densities are extreme and spread out. So
the closed-form-vs-grid verdict is **dataset-dependent**, governed by whether the
data's density lands inside the fixed grid interval.

## LG and GRG

- **LG is the strongest model here** — mean GIC **3.595**, best mean rank (2.22),
  and it beats *all* the grid-fit baselines (ER 4.009, BA 3.829, WS 5.642), unlike
  gplus where grid ER/BA edged it out. LG selects **d=0** for the dense
  connectomes (mouse_brain, rat brains, mouse_retina, drosophila) and d=1 for the
  sparser C. elegans / rhesus nets.
- **GRG (closed-form) is again the strongest baseline** (rank 2.50), and is the
  single best model on the dense spatial brain connectomes (rat brains GRG ≈
  2.1–2.4). It is still absent from the pipeline's baseline set
  (`["ER","WS","BA","SBM"]`) — worth adding.
- **WS closed-form still loses** (density-matched k overshoots — e.g. k=170 for
  mouse_retina), though less catastrophically than on gplus.

## Net takeaway

Closed-form moment matching is **not** a universal improvement, but it is
**competitive on connectomes** precisely where the fixed grid interval fails to
bracket the observed density. The robust, dataset-agnostic fix is what the closed
form approximates: **data-adaptive intervals** (centre the grid on the empirical
density / degree, then refine locally), so the parameter search can reach the
optimum at any density. LG, scored at convergence, is the best model on this
dataset; GRG is a strong, currently-missing baseline.
