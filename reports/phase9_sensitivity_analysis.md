# Phase 9 — Load, Capacity, Siting and Price-Scenario Sensitivity

> **Correction (2026-10-12) — two averaging axes were conflated here.** This
> report treats the four price scenarios as one kind of object ("the mean
> curves"). They share one property and differ in another. All four are
> **spatial** means (`spatial_statistic: spatial_mean`), so no bus or zone is
> represented by any of them; they differ on the **temporal** axis, where
> `261009` is a **single day** and `7days` / `30days` / `halfyear` are means
> over 7 / 30 / 180 days (`temporal_statistic`, also emitted per scenario in
> `outputs/phase9_sensitivity.json`). `261009`'s single-day reading is a
> **project ruling**, not an external confirmation. None of the conclusions
> move: the capacity result (the bound is bit-identical from 0.2 MW / 1.6 MWh
> to 10 MW / 100 MWh), the siting result and the 0-of-33-buses-clearing screen
> all follow from the curves' own min/max against the 200.6957 CNY/MWh
> break-even floor, which does not depend on how the averaging is described.
> Figures below are kept exactly as recorded. Current reading:
> `reports/final_research_summary.md` §2.1 and §6.

Four sensitivities, one frozen configuration, no MATD3.

This phase runs in **two steps** by design: a feasibility screen decides how
much of the experiment is worth running, and only what survives it gets a full
96-period clearing. The screen is not a substitute for physical verification —
every run below still checks voltage, lines, SOC and feasibility — it decides
only whether a *revenue* comparison could possibly say anything.

## The result, in one paragraph

**The arbitrage window is shut at every bus of the network, on every price
curve, for every battery size.** The screen's argument is an inequality rather
than an observation: the most any battery can capture is the day's extremes,
`eta_dis · max − min / eta_ch`, which ignores both the power and the energy
limit and is therefore an optimistic bound on *every* capacity and *every*
location at once. On all four curves that bound falls short of the required margin -- which is
derived from the clearing's objective, not assumed -- at all 33 buses. Capacity therefore cannot open the window —
the bound does not contain the capacity at all — and no placement can either.
**Nothing was tuned to change that**, and the physical half of each sensitivity
is reported as measured.

## 1. Configuration

| | |
|---|---|
| scenario | `typical_day_industry` (IEEE 33-bus, 3.715 MW coincident peak) |
| fleet | 4 × (0.4 MW / 1.6 MWh), buses 9, 14, 17, 27 |
| `eta_ch` / `eta_dis` | 0.92 / 0.92; SOC 0.10–0.90 from 0.50 |
| `cycle_cost` | 100 CNY/MWh per leg → **a cycle must clear 200.6957 CNY/MWh** |
| horizon | 96 × 15 min, `dt = 0.25 h` |
| strategy | `fixed` (truthful) throughout |
| settlement | `counterfactual_eval.settle_day` |
| driver | `src/phase9_sensitivity.py` |
| raw output | `outputs/phase9_sensitivity.json` |

`cycle_cost`, efficiency, price limits and the greedy thresholds are exactly as
Phases 5–8 left them. **No parameter was adjusted in this phase.**

## 2. Economic feasibility screen

### The argument

For any battery, on any price series,

```
capturable margin  ≤  eta_dis · max_t price[t]  −  min_t price[t] / eta_ch
```

The right-hand side charges at the day's single cheapest instant and discharges
at its single dearest, respecting neither the power limit nor the energy limit.
It is an **optimistic bound on every real battery at once** — bigger or smaller,
here or there. If it does not clear the requirement, nothing does.

### External price and the optimistic margin

| curve | role | mean | min | max | available spread | optimistic margin | required | shortfall |
|---|---|---|---|---|---|---|---|---|
| `261009` | `same_day_or_same_day_mean` | 386.83 | 225.90 | 445.00 | 219.10 | 163.86 | 200.70 | **−36.83** |
| `7days` | `trailing_7_day_mean` | 395.55 | 257.50 | 477.87 | 220.37 | 159.75 | 200.70 | **−40.95** |
| `30days` | `trailing_30_day_mean` | 407.82 | 333.08 | 450.86 | 117.78 | 52.75 | 200.70 | **−147.95** |
| `halfyear` | `trailing_180_day_mean` | 346.33 | 284.93 | 393.86 | 108.93 | 52.64 | 200.70 | **−148.05** |

### The network's own nodal prices

The screen does not assume the internal LMP equals the external curve — it
clears the day and measures each bus.

| curve | best nodal margin | at bus | buses clearing (of 33) | best shortfall |
|---|---|---|---|---|
| `261009` | 173.74 | 17 | **0** | −26.96 |
| `7days` | 169.56 | 17 | **0** | −31.14 |
| `30days` | 52.75 | 0 | **0** | −147.95 |
| `halfyear` | 52.64 | 0 | **0** | −148.05 |

The deepest-feeder buses (17 among them) carry a nodal premium over the
wholesale curve, and it is real — bus 17 reaches 468.07 CNY/MWh against a curve
maximum of 445.00. **It is still not enough.** On `30days` and `halfyear` the
best bus is the slack, where the nodal price *is* the wholesale price, because
no bus anywhere gathers a premium worth having.

### Does any battery size help?

| battery | best optimistic margin | at bus | clears | best LP margin over all buses |
|---|---|---|---|---|
| 0.2 MW / 1.6 MWh | 173.74 | 17 | no | 0.0000 |
| 0.4 MW / 1.6 MWh (baseline) | 173.74 | 17 | no | 0.0000 |
| 0.4 MW / 3.2 MWh | 173.74 | 17 | no | 0.0000 |
| 0.8 MW / 1.6 MWh | 173.74 | 17 | no | 0.0000 |
| 2.0 MW / 8.0 MWh | 173.74 | 17 | no | 0.0000 |
| 4.0 MW / 32.0 MWh | 173.74 | 17 | no | 0.0000 |
| **10.0 MW / 100.0 MWh** | 173.74 | 17 | no | 0.0000 |

The bound is **identical to six decimal places across a 50× range of size**,
because it does not contain the capacity. `test_the_capacity_bound_is_the_same_for_every_size`
asserts exactly that, so the argument cannot silently stop holding.

**Scope decision.** With the window shut at every bus and for every size, a
capacity × siting × curve grid would repeat the same idle clearing many times
over. Per the instruction to stop when configurations only reproduce the same
idle result, the grid was **not** run. What follows is the physical half, which
is capable of varying, plus the confirmation runs.

## 3. Load-shape sensitivity — `typical_day` vs `typical_day_industry`

### A confound, stated first

**These two scenarios differ in two things at once.** `typical_day` uses the
native load shape priced on `synthetic_v1`; `typical_day_industry` uses the
industry shapes priced on `dianyicha`. So a difference in any *price-dependent*
column cannot be attributed to the load shape alone. Only the columns below the
line are cleanly attributable; the rest are confounded and are reported as such.

### Cleanly attributable — properties of the shape alone

| | `typical_day` | `typical_day_industry` |
|---|---|---|
| coincident peak | 3.7150 MW | **3.7150 MW** (identical, as declared) |
| daily energy | 47.04 MWh | **54.79 MWh** (+16.5 %) |
| load factor | 0.5276 | **0.6145** |
| peak period | slot 74 (18:30) | slot 65 (16:15) |

The peak is unchanged by construction — the scenario contract pins the
coincident peak to the network's own demand. The energy is **higher** on the
industry day, because the industry curves are flatter than the synthesised
`native` profile (whose night base is 0.60), so the same peak is held for more
of the day. This was predicted backwards in the Phase-4 plan and corrected in
that report; it is restated here because it is the one clean load-shape result.

### Confounded — depends on both load shape and price source

| | `typical_day` | `typical_day_industry` |
|---|---|---|
| voltage range | 0.9646 – 1.0000 pu | 0.9697 – 1.0076 pu |
| band violations | 0 | 0 |
| max line utilisation | **0.1960** | **0.1673** |
| curtailment | **14.265 MWh** | **2.180 MWh** |
| network-best nodal margin | **201.03** (bus 13) | 173.74 (bus 17) |
| fleet settled profit | −1.51e-04 CNY | −8.19e-05 CNY |

### A knife-edge worth recording

`typical_day`'s network-best nodal margin is **201.03 CNY/MWh against a
200.6957 requirement** — nominally above break-even. Checking the configured
fleet itself:

| bus | nodal range | margin | shortfall | LP margin |
|---|---|---|---|---|
| 9 | 149.95313 – 395.19594 | 200.587736 | −0.107916 | 0.0000 |
| 14 | 150.71820 – 396.21726 | 200.695749 | **+0.000097** | 0.0000 |
| 17 | 151.41801 – 397.03627 | 200.688574 | −0.007078 | 0.0000 |
| 27 | 149.91974 – 392.80198 | 198.421588 | −2.274064 | 0.0000 |

**One** of the four configured buses is above break-even by the *optimistic*
measure, and by **0.0001 CNY/MWh** — a knife edge in the literal sense. The
fleet does move — **0.0405 MWh**, not the 1e-6 noise of the industry day. Yet
the LP, which must respect 0.4 MW and 1.6 MWh over the actual time separation
between trough and peak, returns **exactly zero** at every one of them, and the
settled profit is still ≈ 0.

> **Corrected in Phase 10.** This table was first written against the lossless
> `2 * cycle_cost = 200` and reported three of four buses above break-even
> (+0.59, +0.70, +0.69). The requirement is the derived
> `cycle_cost * (1/eta_ch + eta_dis) = 200.6957`, under which only bus 14
> nominally clears. Its shortfall is **+0.000097**, not the +0.004 this report
> carried for one revision: that figure was the difference of two *rounded*
> margins (`200.70 − 200.6957`) at 2-dp precision, 43× the real value. The rows
> are now read from `outputs/phase9_sensitivity.json`, where `load_shape`
> carries the per-bus table, and `test_the_native_day_sits_on_breakeven_at_the_configured_buses`
> re-derives each margin from that bus's own prices. The reading of the finding
> is unchanged and in fact sharpened — the window is shut at every one of these
> buses by the LP's own measure — but the recorded numbers had to move with the
> formula. See `reports/final_research_summary.md` §5.

This is the clearest possible illustration of why the optimistic bound is only
a screen: **a margin of +0.0001 CNY/MWh is not capturable by a battery that
cannot be in two places at once.** It also shows how close the native day sits to the
edge — which is why the fleet behaved as idle there in every earlier phase
without anything being wrong.

### What the industry-data claim is and is not

**[FORBIDDEN, restated from Phase 8]** The industry workbook supplies
**normalised shapes only**. This is not an experiment about Jiangsu's real load
magnitude, no absolute-MW conversion was performed, and `absolute_mw()` still
refuses.

## 4. Capacity sensitivity

Five configurations, full 96-period clears, `261009`:

| case | P each | E each | fleet profit | charge | discharge | V violations | max line util. | SOC 96/96 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.400 | 1.60 | −8.19e-05 | 7.5e-07 | 6.3e-07 | 0 | 0.1673 | yes |
| power ÷2 | 0.200 | 1.60 | −6.83e-06 | 5.9e-08 | 5.0e-08 | 0 | 0.1673 | yes |
| power ×2 | 0.800 | 1.60 | −1.46e-04 | 1.4e-06 | 1.2e-06 | 0 | 0.1673 | yes |
| energy ÷2 | 0.400 | 0.80 | −1.78e-05 | 1.5e-07 | 1.3e-07 | 0 | 0.1673 | yes |
| energy ×2 | 0.400 | 3.20 | −2.68e-04 | 2.4e-06 | 2.0e-06 | 0 | 0.1673 | yes |
| joint ×2 | 1.600 | 3.20 | −4.69e-05 | 4.4e-07 | 3.7e-07 | 0 | 0.1673 | yes |

**Recorded finding: under the current cost and price scenarios, changing the
capacity produced no distinguishable economic revenue.** Every figure is
floating-point residue around zero; the throughput scales with capacity only
because the residue scales with capacity. Line utilisation is *identical* across
all six because the fleet does not move, so the network state is unchanged.

What capacity does affect, and what is worth reporting: **none of the
configurations violated a constraint, and every SOC trajectory verified
96/96.** Capacity changes the fleet's physical envelope, not its revenue here.

## 5. Siting sensitivity

Five placements, count and ratings fixed, `261009`:

| placement | buses | fleet profit | best margin at own buses | network-best margin | V violations | max line util. |
|---|---|---|---|---|---|---|
| configured | 9, 14, 17, 27 | −8.19e-05 | 173.74 | 173.74 | 0 | 0.1673 |
| previous | 6, 20, 24, 31 | −1.55e-04 | 169.19 | 173.86 | 0 | 0.1649 |
| previous | 10, 15, 19, 26 | −2.13e-04 | 173.29 | 173.78 | 0 | 0.1656 |
| deep feeder | 16, 17, 18, 19 | −1.91e-04 | 173.78 | 173.78 | 0 | 0.1685 |
| spur ends | 21, 24, 30, 31 | −1.64e-05 | 169.94 | 173.91 | 0 | 0.1663 |

The placements deliberately include the two the project previously used and two
sets chosen to sit on the deepest-feeder buses, where the nodal premium is
largest — the most favourable locations the network has, not an arbitrary
sample.

**Every placement leaves all its own buses short of the requirement**, and the
network-wide best never exceeds 173.91 against 200. All placements are
physically feasible with no voltage violation and no line near its limit.

**No economic ranking is offered.** With nothing trading, ranking these
placements by a ~1e-4 CNY residue would be ranking numerical noise as though it
were siting performance. The reported siting result is the physical one: *the
placement changes line utilisation (0.1649–0.1685) and the nodal premium each
unit sits on (169.19–173.78), and none of it reaches break-even.*

**Absolute-MW industry load remains refused**, and no capacity or siting
experiment introduced an unverified industry load scale.

## 6. Price-scenario sensitivity

| curve | role | mean | min | max | spread | optimistic margin | clears | best nodal margin | buses clearing | curtailment |
|---|---|---|---|---|---|---|---|---|---|---|
| `261009` | `same_day_or_same_day_mean` | 386.83 | 225.90 | 445.00 | 219.10 | 163.86 | **no** | 173.74 | 0 | 2.180 |
| `7days` | `trailing_7_day_mean` | 395.55 | 257.50 | 477.87 | 220.37 | 159.75 | **no** | 169.56 | 0 | 1.611 |
| `30days` | `trailing_30_day_mean` | 407.82 | 333.08 | 450.86 | 117.78 | 52.75 | **no** | 52.75 | 0 | 0.155 |
| `halfyear` | `trailing_180_day_mean` | 346.33 | 284.93 | 393.86 | 108.93 | 52.64 | **no** | 52.64 | 0 | 0.916 |

Provenance for all four: workbook SHA-256
`5fc8f9a8461b680fd38a6d2fefb3d2ff99f239c8bd13a6c77df27de3bb9612f3`,
`is_measured = false` on every curve, header labels preserved verbatim
(`261009` is 日前均值). **These are four price scenarios, not four dates.**

The spread does not track the mean: `30days` has the highest mean (407.82) and
one of the two narrowest spreads (117.78), while `261009` has the lowest mean
and the widest. Curtailment follows the spread rather than the level — a
narrower spread makes importing relatively more attractive than dispatch.

**[FORBIDDEN]** No cross-day variance, positive-day rate, or generalisation
statistic is computed from these four columns.

## 7. Combinations not run, and why

| not run | reason |
|---|---|
| capacity × siting × 4-curve grid | the screen shows no bus clears on any curve and no size clears at all, so every cell would reproduce an idle clearing |
| full sensitivity on `7days`/`30days`/`halfyear` for capacity and siting | the same — `261009` is the curve with the **widest** spread and the **highest** nodal margin, so it is the most favourable case; the other three can only be worse |
| any `mpc` / `greedy_threshold` / MATD3 arm | out of scope: this phase is about configuration sensitivity, and Phase 7 already established all strategies idle on this band |
| absolute-MW industry load scenarios | refused by design; the conversion is unverified |

The screen is what justifies the small experiment, and it is itself tested: if
the bound ever cleared on some bus, `test_no_bus_in_the_network_clears_on_any_curve`
fails and this section has to be rewritten.

## 8. Tests

**New:** `tests/test_phase9_sensitivity.py`, 17 tests.

```
python -m pytest tests/test_phase9_sensitivity.py -v   -> 17 passed (84s)
python -m pytest tests/   -> 775 passed, 51 deselected, 0 failed (7m50s)
```

775 = 758 at the end of Phase 8, plus the 17 new.

Coverage maps to the required list:

| requirement | test |
|---|---|
| cache isolation across configuration changes | `test_repeating_a_configuration_reproduces_it` (an intervening different config is run to poison a naive cache), `test_the_config_is_restored_after_a_reconfigured_run` |
| storage capacity and bus position actually update | `test_a_capacity_change_reaches_the_built_fleet`, `test_a_siting_change_reaches_the_built_fleet` |
| explicit price-curve selection | `test_each_price_curve_is_selected_explicitly_and_differs` (each must match its sheet and differ from the others) |
| reproducibility | `test_repeating_a_configuration_reproduces_it`, `test_the_config_is_restored_after_a_reconfigured_run` |
| the screen's argument | `test_no_curve_clears_its_optimistic_margin`, `test_no_bus_in_the_network_clears_on_any_curve`, `test_the_capacity_bound_is_the_same_for_every_size` |
| the load-shape contract | `test_the_load_shape_comparison_holds_the_scale_and_changes_the_shape`, `test_both_load_shape_days_are_physically_feasible` |
| phases 1–8 interfaces intact | `test_phase_1_to_8_interfaces_are_intact` |

The screens assert **inequalities**, not recorded numbers: if a future price
curve or network change reopened the window, the tests fail and name the report
that needs revisiting, rather than passing against a stale expectation.

## 9. Boundaries

Inherited from `reports/phase8_data_limitations.md` and unchanged:

- **[FORBIDDEN]** Concluding that the Jiangsu real day-ahead market has no
  storage arbitrage. Four **mean** curves under one cycle cost, one battery and
  one network model were measured.
- **[FORBIDDEN]** Cross-day statistics from four curves that are not four dates.
- **[FORBIDDEN]** Any statement about MATD3 — none was trained, loaded or
  evaluated.
- **[FORBIDDEN]** Adjusting `cycle_cost`, efficiency, thresholds or network
  constraints to turn this zero into a positive. The zero is the measurement.
- The results are **modelled storage day-ahead energy net revenue within the
  model's boundary**, not a settlement bill; `other_cost` is still zero and the
  out-of-model charges are still unmodelled.

New in this phase: the capacity and siting conclusions are statements about
**this** band and **this** cost. They rest on the margin bound, which is a
property of the prices, the efficiency and the cycle cost — change any of those
and the screen must be re-run before any conclusion is carried over.

## 10. Workspace

HEAD `a1151fd2ab52e4a49a610173a1bba04c56bda563`, branch `master`, unchanged.
Backed up to
`C:/Users/26036/AppData/Local/Temp/pre_phase9_worktree.diff` before any edit.

**No tracked file changed in this phase.** The whole-tree diff is identical to
the Phase-9 baseline, verified hunk by hunk across all 27 modified files, with
each would-be "lost" line checked against the user's own original diff to
distinguish a superseded line of mine from a lost line of theirs. **No line of
the user's was lost.** Additions this phase: `src/phase9_sensitivity.py` and
`tests/test_phase9_sensitivity.py`.

No `reset`, `checkout --` or `clean` was run; `gurobi.env` was not read, copied
or printed.

## Status

Phase 9 complete. Stopping here for review. No further phase was started.
