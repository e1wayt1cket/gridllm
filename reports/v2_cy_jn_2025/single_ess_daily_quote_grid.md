# Single-ESS daily quote grid (P0-2)

Day: `2022-12-30|2025-12-31`. Primary target: ESS14; second target:
ESS27. For each target, the other three ESS are truthful for all 24 blocks. The target holds one charge quote and one
discharge quote for all 24 blocks. Each combination is a complete 24-block
rolling SOCP with the same load, price, `line50_q2.0`, cycle cost 100, initial
SOC 0.5 and terminal SOC 0.5. Data: `single_ess_quote_surface_ESS14.csv/json`;
the combined files are `single_ess_quote_surface.csv/json`.

This is a **conditional** optimum inside the fixed-quote family with three
truthful competitors. It is not the four-agent dynamic-bidding global optimum.

## Surface (ESS14 payoff, CNY/day)

Rows are charge quotes, columns discharge quotes.

| charge \ discharge | 0 | 33.3 | 66.7 | 100 | 133.3 | 166.7 | 200 | 233.3 | 266.7 | 300 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 67.7 | 76.8 | 66.8 | 66.9 | −98.0 | −132.7 | −132.7 | −132.7 | −132.7 | −132.8 |
| 16.7 | 67.7 | 76.7 | 66.7 | 66.7 | −98.1 | −132.8 | −132.8 | −132.8 | −132.8 | −132.8 |
| 33.3 | 76.8 | 77.0 | 66.1 | 66.1 | −41.6 | −76.3 | −76.3 | −76.3 | −76.3 | −76.3 |
| 50.0 | 76.9 | **77.0** | 66.1 | 66.1 | −41.6 | −76.3 | −76.3 | −76.3 | −76.3 | −76.3 |
| 66.7 | 73.7 | 73.7 | 64.2 | 63.7 | 12.3 | −2.5 | −2.5 | −2.5 | −2.5 | −2.5 |
| 83.3 | 73.7 | 73.7 | 64.1 | 63.7 | 12.3 | −2.5 | −2.5 | −2.5 | −2.5 | −2.5 |
| 100 | 73.7 | 73.7 | 64.2 | 63.7 | 12.3 | −2.5 | −2.5 | −2.5 | −2.5 | −2.5 |
| 116.7 | 72.4 | 72.4 | 63.9 | 63.4 | 12.3 | −0.0 | −0.0 | −0.0 | −0.0 | −0.0 |
| 133.3 | 72.4 | 72.4 | 63.9 | 63.4 | 12.3 | −0.0 | −0.0 | −0.0 | −0.0 | −0.0 |
| 150 | 72.4 | 72.4 | 63.9 | 63.4 | 12.3 | −0.0 | −0.0 | −0.0 | −0.0 | −0.0 |

B1 truthful ESS14 payoff is **67.73**; the honest grid point (0,0) reproduces
it. The global grid point is **(50, 33.3) at 77.05**, a +9.32 CNY gain over
B1. A 5×5 local refinement around it finds **(37.5, 46.7) at 77.08**, only
+0.03 CNY above the grid point; the surface is a broad plateau, not a sharp
peak.

## Features

| feature | value |
|---|---|
| best 10×10 quote | charge 50.0, discharge 33.3 |
| best payoff / ΔG vs B1 | 77.05 / +9.32 CNY |
| refined best | charge 37.5, discharge 46.7; 77.08 (+0.03) |
| honest (0,0) | 67.73 |
| cells within 1 CNY of best | 6 / 100 |
| local maxima (8-neighbour) | 9 |
| best 3×3 neighbour range | 12.9 CNY |
| best boundary? | no (interior) |
| high discharge ≥166.7 | payoff ≤ 0; 19–24 zero blocks; discharge ≤ 0.59 MWh |

The surface is **flat-topped and multi-modal at the 1-CNY scale**, but it has a
clear low-discharge basin. High discharge quotes (≥166.7) are a loss region:
the target is displaced and the block profit is non-positive. The low-charge /
low-discharge region (charge 0–50, discharge 0–47) is the useful region.

## Current Actor point

The D+A best actor's mean day quotes for ESS14 are charge **20.8**, discharge
**110.2**. The nearest grid point is (16.7, 100), payoff 66.74; the distance to
the grid best is **10.31 CNY**. Its own-policy payoff (all four agents using
their actors) is **−95.41 CNY**, which is a different policy family from the
three-truthful grid and must not be read as the grid value.

The old R10 actor mean is charge 27.3, discharge 165.8; nearest grid
(33.3, 166.7) payoff −76.31, distance 153.36 to the grid best.

The actor is therefore **above the grid-optimal discharge quote** (110 vs 33),
but its charge quote is close. The grid says the target's best response when
the other three are truthful is a low discharge quote; the actor's discharge
is in the high-loss region.

## Answer

- **Optimal single-ESS quote:** charge ≈ 37.5–50, discharge ≈ 33–47 (the
  10×10 best is 50/33.3; the refinement is 37.5/46.7).
- **Surface shape:** a broad low-discharge basin with several local plateaus,
  not a single sharp peak; 6/100 cells within 1 CNY of the best. High
  discharge is a clear loss region.
- **Actor gap:** the selected actor's discharge mean is ~110, far above the
  ~33–47 optimum; the charge mean is close. The nearest-grid gap is 10.3 CNY
  (conditional family), while the actor's own-policy payoff is much lower
  because all four actors are quoting, not three truthful.

## ESS27 (second grid)

The same 10×10 enumeration was run for ESS27. B1 truthful ESS27 payoff is
**106.62**; the grid best is **(0, 33.3) at 112.31**, a +5.69 CNY gain over B1.
The best is on the charge boundary (charge 0). A 5×5 refinement around it finds
(0, 70) at 110.97, **below** the 10×10 best; the 10×10 point remains the best.
The surface has 2/100 cells within 1 CNY of the best and 9 local maxima; the
low-discharge basin is broad, and high discharge ≥166.7 is again negative.

| quantity | ESS27 |
|---|---:|
| B1 truthful payoff | 106.62 |
| grid best (charge, discharge) | (0.0, 33.3) |
| best payoff / ΔG vs B1 | 112.31 / +5.69 |
| refined best | (0.0, 70.0), 110.97 |
| D+A best actor mean quotes | (37.9, 77.4) |
| nearest-grid payoff | 110.56 |
| nearest-grid gap to best | 1.75 |
| actor own-policy payoff | 110.33 |
| old R10 mean / nearest-grid payoff | (0.0, 190.2) / −132.58 |

ESS27 is the opposite of ESS14: the selected actor is within 1.75 CNY of the
conditional grid best and its own-policy payoff is 110.33. The actor can choose
a near-optimal quote for ESS27 while it is far from the ESS14 optimum. The
problem is unit-specific, not a uniform gate/actor failure.

## Boundary

No training and no checkpoint write. The target's fixed quote is the only
enumerated variable; the other three are truthful. The grid value is a
conditional optimum, not the four-agent equilibrium.
