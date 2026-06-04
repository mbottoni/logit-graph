# Closed-form baselines vs grid (OASIS-3 human connectomes) — findings

Experiment: `run_human_connectomes_closedform.py` (human-connectome twin of the
gplus / animal-connectome runs). Score ER/BA/WS/KR/GRG against OASIS-3 brain
networks by spectral GIC (2·KL + 2·n_params), comparing **closed-form moment
estimators** vs the current **fixed-interval grid**, alongside LG fit by
**burn-in + ensemble mean** of the spectral density.

Config: 8 subjects sampled (seeded) from each of 4 parcellation scales —
`oasis3 scale1` (n≈110, density≈0.50), `scale2` (n≈156, 0.43), `scale3` (n≈258,
0.31), `repeated_10_scale_250` (n≈378, 0.085) — **32 subjects**, `n_runs=5`,
`grid_points=5`, LG d∈{0,1,2}, seed 12345. Graphs loaded from GraphML as
undirected, **unweighted (binary)** largest connected component (brain edges are
weighted fiber densities).

## Aggregate (mean GIC across 32 subjects, lower = better)

| family | grid | closed-form | Δ(grid−cf) | cf wins |
|---|---|---|---|---|
| ER | 3.798 | 3.802 | −0.00 | 56% |
| BA | 4.103 | **3.700** | +0.40 | **75%** |
| WS | **5.306** | 8.165 | −2.86 | 0% |

Closed-form only: KR 3.808, **GRG 2.773**, **LG 3.575**.

Mean rank (LG + closed-form baselines): **GRG 1.00**, **LG 2.78**, BA 3.19,
ER 3.72, KR 4.31, WS 6.00.

## Headline — GRG is the best model on every single subject

**GRG (closed-form) ranks 1st on all 32 subjects** (mean rank exactly 1.00,
mean GIC 2.773 — far below everything else). Brain connectomes have strong
geometric / spatial organisation, and the random geometric graph — a single
closed-form radius — captures it better than any other family *and* better than
LG. GRG was 2nd on the animal connectomes; on the human OASIS-3 nets it is the
outright winner. It remains **absent from the pipeline's baseline set**
(`["ER","WS","BA","SBM"]`) — the strongest argument yet for adding it.

## Closed-form is competitive-to-better here — the interval cap bites hard

The OASIS scales are **dense** (density 0.31–0.54), far above the grid's ER cap
(0.25) and BA cap (m=8):

- **BA: closed-form clearly wins** (cf 3.700 vs grid 4.103, cf wins 75%). The
  grid caps at m=8 while the real m is ≈25–45, so grid-BA is far too sparse;
  closed-form m matches and wins decisively.
- **ER: a wash** (cf wins 56%) — the dense subjects' p>0.25 is unreachable by the
  grid, but the normalized-Laplacian KL is fairly flat in p, so the cap costs
  less than for BA.
- **WS: closed-form still collapses** (density-matched k≈50–90 → near-lattice
  spectrum nothing like a brain), and grid-WS is the worst family overall.
- The **sparse** scale (`repeated_10_scale_250`, density 0.085) behaves like
  gplus — grid edges out cf — confirming the verdict tracks whether the density
  lands inside the grid box.

## LG

LG is the **2nd-best model** (mean GIC 3.575) and **beats every grid-fit baseline
at every scale**:

| scale | n | density | LG | ER-grid | BA-grid | WS-grid |
|---|---|---|---|---|---|---|
| scale1 | 110 | 0.498 | **3.534** | 3.898 | 3.948 | 5.561 |
| scale2 | 156 | 0.433 | **3.666** | 3.752 | 4.322 | 5.684 |
| scale3 | 258 | 0.304 | **3.476** | 3.552 | 4.486 | 5.370 |
| rep_250 | 378 | 0.085 | **3.625** | 3.992 | 3.653 | 4.610 |

LG selects **d=0** for the dense OASIS scales and d=1 for the sparse repeated
scale. So among the standard pipeline models LG is the best — only the
(currently-unused) GRG beats it.

## Net takeaway

Consistent with the animal connectomes and sharper: on dense, spatially-organised
brain networks the fixed grid interval is the binding constraint (closed-form /
density-matching is competitive-to-better, BA especially), **GRG is the dominant
baseline**, and **LG is the best of the standard models**. The robust fixes are
the same: data-adaptive parameter intervals (what closed-form approximates) and
adding GRG to the baseline set.
