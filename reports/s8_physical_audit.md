# S8 — the physical audit of the subset day

One cleared day on `typical_day_industry_subset`: 96 periods, 33 buses. Produced by `src/s8_physical_audit.py`; every number is a solver output and none has been adjusted.

> **Not an externally certified result.** The reading unit's basis is the recorded project ruling `user_confirmed_project_ruling`; `is_externally_verified` is **False** and this audit inherits that. The ledger it reads is a **formal** ledger — the ruling is an admissible basis and `validate_formal_ledger` passes it — but a formal basis is not a verified one, so nothing here may be presented as externally certified.

> **Conditional on this repository's own line ampacities.** Those are project-assigned rather than the published `case33bw` ratings (§3). Every limit statement below holds under that assumption and nothing here is a validation against authoritative standard line ratings.

> **Cross-date diagnostic.** Load day **2022-11-30**, price day **2026-10-09** — different days, paired by decision. No figure here is what either day actually met in its own market.

## 1. The verdict

**The network carried the load.** Served **55.8 MWh** of **55.8 MWh** (**100.00%**); shed **0.000 MWh**.

**Both halves of that sentence are conditional.** The voltage band is declared, and the line ampacities are **project-assigned** rather than the published `case33bw` ratings (§3): 0 bus-periods leave the band and 0 line-periods reach their limit **under this repository's own ampacity rule**. Nothing here is a validation against authoritative standard line ratings, and the result must not be described as one.

| | |
|---|---|
| scenario | `typical_day_industry_subset` |
| periods × buses | 96 × 33 |
| served / asked | 55.7941 / 55.7941 MWh |
| shed | 0.0000 MWh |
| served share | 100.0000% |

## 2. Voltage, all 96 periods and 33 buses

| | |
|---|---|
| band | 0.9300 .. 1.0500 pu (declared, not tuned) |
| min | **0.9480** pu at bus 32, period 88 |
| max | 1.0166 pu at bus 30, period 17 |
| bus-periods below the band | **0** |
| bus-periods above the band | 0 |
| sitting on the lower bound | **False** |

Per bus:

| bus | min pu | max pu | mean pu | periods at the lower bound |
|---|---|---|---|---|
| 0 | 1.0000 | 1.0000 | 1.0000 | 0 |
| 1 | 0.9980 | 0.9996 | 0.9989 | 0 |
| 2 | 0.9913 | 0.9997 | 0.9950 | 0 |
| 3 | 0.9896 | 0.9989 | 0.9937 | 0 |
| 4 | 0.9884 | 0.9982 | 0.9929 | 0 |
| 5 | 0.9858 | 0.9967 | 0.9909 | 0 |
| 6 | 0.9850 | 0.9964 | 0.9905 | 0 |
| 7 | 0.9763 | 0.9924 | 0.9878 | 0 |
| 8 | 0.9746 | 0.9919 | 0.9869 | 0 |
| 9 | 0.9747 | 0.9920 | 0.9869 | 0 |
| 10 | 0.9762 | 0.9922 | 0.9880 | 0 |
| 11 | 0.9763 | 0.9922 | 0.9881 | 0 |
| 12 | 0.9738 | 0.9911 | 0.9867 | 0 |
| 13 | 0.9731 | 0.9907 | 0.9863 | 0 |
| 14 | 0.9725 | 0.9911 | 0.9858 | 0 |
| 15 | 0.9717 | 0.9907 | 0.9853 | 0 |
| 16 | 0.9711 | 0.9905 | 0.9850 | 0 |
| 17 | 0.9707 | 0.9903 | 0.9847 | 0 |
| 18 | 0.9967 | 0.9990 | 0.9982 | 0 |
| 19 | 0.9859 | 0.9956 | 0.9929 | 0 |
| 20 | 0.9834 | 0.9950 | 0.9917 | 0 |
| 21 | 0.9822 | 0.9950 | 0.9914 | 0 |
| 22 | 0.9867 | 1.0009 | 0.9933 | 0 |
| 23 | 0.9777 | 1.0037 | 0.9904 | 0 |
| 24 | 0.9676 | 1.0057 | 0.9866 | 0 |
| 25 | 0.9856 | 0.9966 | 0.9907 | 0 |
| 26 | 0.9856 | 0.9966 | 0.9906 | 0 |
| 27 | 0.9651 | 1.0082 | 0.9863 | 0 |
| 28 | 0.9638 | 1.0081 | 0.9861 | 0 |
| 29 | 0.9602 | 1.0107 | 0.9857 | 0 |
| 30 | 0.9528 | 1.0166 | 0.9861 | 0 |
| 31 | 0.9481 | 1.0165 | 0.9849 | 0 |
| 32 | 0.9480 | 1.0164 | 0.9847 | 0 |

## 3. Lines

| | |
|---|---|
| lines in the radial model | 32 |
| worst utilization | **0.1889** on line 1, period 52 |
| line-periods at or over the limit | **0** |
| lines at or over the limit at some period | **0** |

### Where the capacity comes from

- field: `net.line.max_i_ka, the thermal current limit`
- used by: dispatch_socp._build_line_params converts it to a per-unit current limit; the constraint is I_sq <= limit^2
- provenance: **PROJECT-ASSIGNED, not the published case33bw rating. grid.build_base_network computes clip(base_ampacity_ka / sqrt(r_ohm_per_km * length_km), min_line_ka, max_line_ka) * line_capacity_multiplier * per-line override**
- declared inputs: `base_ampacity_ka` 0.25, clip to [0.08, 0.8] kA, `line_capacity_multiplier` 1.5, overrides none
- resulting ampacity range: 0.2652 .. 1.2000 kA
- lines marked out of service: [32, 33, 34, 35, 36]

**The ampacity is project-assigned, not the published `case33bw` rating.** The line capacities are a modelling choice made in this repository, so two limits follow and both hold for every utilization figure below and for the verdict in §1:

1. Every statement here about a line being within its limit is **conditional on this ampacity rule**. A different rule would move every utilization number, and nothing here would show it.
2. **This is not a validation against authoritative standard line ratings**, and it must not be described as one. The `case33bw` ampacities were replaced, not adopted.

| line | max utilization | at period | ampacity kA | over limit |
|---|---|---|---|---|
| 1 | 0.1889 | 52 | 0.5341 | no |
| 23 | 0.1784 | 88 | 0.3962 | no |
| 22 | 0.1784 | 52 | 0.3957 | no |
| 18 | 0.1659 | 76 | 0.3058 | no |
| 21 | 0.1298 | 52 | 0.5583 | no |
| 0 | 0.1242 | 75 | 1.2000 | no |
| 29 | 0.1132 | 17 | 0.3799 | no |
| 30 | 0.1119 | 88 | 0.6730 | no |
| 36 | 0.0960 | 52 | 0.5303 | no |
| 32 | 0.0937 | 76 | 0.2652 | no |
| 28 | 0.0889 | 88 | 0.5264 | no |
| 19 | 0.0794 | 76 | 0.5860 | no |
| 17 | 0.0594 | 76 | 0.9260 | no |
| 34 | 0.0592 | 76 | 0.2652 | no |
| 2 | 0.0538 | 73 | 0.6199 | no |
| 4 | 0.0510 | 34 | 0.4144 | no |
| 27 | 0.0446 | 92 | 0.4182 | no |
| 7 | 0.0440 | 76 | 0.3695 | no |
| 20 | 0.0406 | 76 | 0.4454 | no |
| 3 | 0.0391 | 34 | 0.6075 | no |
| 33 | 0.0382 | 76 | 0.2652 | no |
| 11 | 0.0320 | 76 | 0.3095 | no |
| 14 | 0.0165 | 76 | 0.4341 | no |
| 8 | 0.0141 | 76 | 0.3670 | no |
| 12 | 0.0130 | 76 | 0.5096 | no |
| 15 | 0.0129 | 76 | 0.3303 | no |
| 5 | 0.0126 | 76 | 0.8667 | no |
| 24 | 0.0120 | 48 | 0.8323 | no |
| 16 | 0.0097 | 76 | 0.4383 | no |
| 25 | 0.0070 | 47 | 0.7034 | no |
| 31 | 0.0052 | 76 | 0.6422 | no |
| 10 | 0.0040 | 76 | 0.6129 | no |

## 4. PCC — power at the point of common coupling

### There is no import limit

**Read this before the table.** The one PCC constraint this model carries is on **reverse flow**. `dispatch_core.add_grid_exchange` builds a single net-exchange variable and bounds it **below**:

```
p_grid = addVars(T, lb=-network.reverse_power_limit_mw, ub=+INFINITY)
```

So the declared 2.5 MW caps **export** and bounds import not at all, and every engine that clears — `socp`, `lindistflow`, `dc` — builds it through that one helper. **Import is priced, not capped.** An import maximum compared against this figure compares two different quantities; the quantity the bound acts on is the most negative net exchange, which is reported below.

| | |
|---|---|
| import max | 3.2269 MW at period 75 |
| import mean | 1.7610 MW |
| import energy | 42.2635 MWh |
| **import limit** | **none — unbounded by construction** |
| export max | 0.0000 MW |
| export energy | 0.0000 MWh |
| periods exporting | 0 |
| **most negative net exchange** | 0.5324 MW at period 17 |
| reverse-flow limit (the bound actually applied) | −2.5000 MW |
| slack on that bound | 3.0324 MW |
| reverse limit breached | **False** |

Both are the same variable, in MW, at the same period index: the import series is `max(net, 0)` and the export series `max(-net, 0)`, taken from the solver's own `p_grid[t].X` — the variable the bound sits on. So the reading and the constraint are on the same quantity, and on this day the exchange never went negative at all.

## 5. Supply, load, and the energy balance

| term | MWh |
|---|---|
| load asked | 55.7941 |
| served | 55.7941 |
| shed | 0.0000 |
| load peak (asked) | 3.6534 MW at period 75 |
| load peak (served) | 3.6534 MW at period 75 |
| grid import | 42.2635 |
| grid export | 0.0000 |
| renewable available | 16.5889 |
| renewable consumed | 14.2538 |
| renewable curtailed | 2.3351 |
| storage charged | 0.0000 |
| storage discharged | 0.0000 |
| **network losses** | **0.7232** |

The identity, per period, rebuilt from the clearing's own terms:

```
grid_import - grid_export + pv_used + wind_used + p_dis
    - served - p_ch - losses  =  0
```

| residual | MW |
|---|---|
| max | 1.145e-09 |
| mean | 1.369e-10 |

`losses` is the model's own `r * I^2` per line, recovered from the solver's line currents through `dispatch_socp._build_line_params`. Load closure `served + unserved - asked` is within 1.278e-10 MW.

## 6. Solver state, nodal-price fallbacks, anomalous periods

| | |
|---|---|
| status | `OPTIMAL` (optimal: True) |
| runtime | — s |
| fell back to a single-period engine | False |
| balance duals available | True |
| nodal-price fallbacks (cells) | 0 |
| fallback periods | none |
| price source | `nodal` |
| **anomalous periods** | none |

An anomalous period is one whose price is not a nodal balance dual -- either it fell back to the wholesale curve, or its price is not finite. `lmp` may not be read as a nodal price field over any period listed above.

## 7. Reactive load: what it is and what it is not

| | |
|---|---|
| rule | q_load = served * tan(phi), per agent and per period (dispatch_socp's reactive-power balance) |
| power factor | 0.8503 |
| tan(phi) | 0.618956 |
| Q at the peak period | 2.2613 MVAr (period 75) |
| Q energy over the day | 34.5341 MVArh |
| benchmark Q | 2.3 MVAr |
| inverter reactive support | True |

**This is an assumption, not a measurement.** simulation assumption on ACTIVE load, not a measurement -- no meter in this workbook reports reactive power. It is charged on the **served** active load, so a period that sheds is also a period that draws less reactive power. The factor is `0.8503`, declared to reproduce the benchmark's 2.3 MVAr at the benchmark's own 3.715 MW; this day peaks below that, so the reactive load is correspondingly below 2.3 MVAr. It was not re-tuned.

## 8. What the clearing objective and the RE rate are

| | |
|---|---|
| `clearing_objective` | 19,451.74 CNY |
| what it is | the clearing model's own optimization objective, in CNY. It is NOT social welfare and NOT any participant's payoff: it values each storage unit at that unit's own declared bid and offer rather than at the settlement price, so it cannot be read as anyone's profit |
| `re_consumption_rate` | 85.9235 % |
| how it is computed | 100 * total_re_used_mwh / total_re_avail_mwh, where available sums the fleet's forecast PV and wind over the horizon and used is what the clearing consumed |
| carbon | 24.5128 tCO2 |
| carbon definition | grid_carbon_tco2 on the import part of the net exchange times market_design.emission_factor_grid |
| carbon intensity | 0.439344 tCO2/MWh |

**The clearing objective is not welfare and not a payoff.** It is the optimization expression the solver minimised, in CNY, and it values each storage unit at that unit's own declared bid and offer rather than at the settlement price. Any economic statement about this day belongs to `participant_payoff`, not to this number.

**The RE rate is a ratio of energies, in percent.** Its numerator is the renewable energy the clearing consumed and its denominator is the energy the fleet's PV and wind could have produced over the horizon, both in MWh. It says nothing about price, cost or revenue.

## 9. What was not tuned

read from configuration and reported as found; no load, power factor, line capacity, voltage bound or storage parameter was moved to produce this result:

| parameter | value read |
|---|---|
| voltage band | 0.93 .. 1.05 pu |
| `load_power_factor` | 0.8503 |
| `line_capacity_multiplier` | 1.5 |
| `reverse_power_limit_mw` | 2.5 |
| `reactive_support` | True |

## 10. What this audit does not establish

- It is **not** the formal physical network case. The reading unit is unverified; the ledger is an `audit` ledger.
- It is **not** a historical replay: the load day and the price day differ by decision.
- The line ampacities are **project-assigned**, so every utilization figure is conditional on that assignment.
- It says nothing about a learned policy, about profit, or about what any participant would have earned.

## Reproduce

```bash
PYTHONPATH=src python src/s8_physical_audit.py --scenario typical_day_industry_subset
```
