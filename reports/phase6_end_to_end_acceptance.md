# Phase 6 — Single-Scenario End-to-End Acceptance

One representative price scenario run through the whole chain: metered industry
load shapes, the IEEE 33-bus network, the independent storage fleet, a 96-period
SOCP clearing, and the shared settlement.

**The acceptance found and fixed one real defect**, which is the most important
thing in this report — see §3. The mechanics were otherwise sound: 96 periods
cleared, no voltage or thermal violation, no fallback, both settlement paths
agreeing to the last bit, and the run reproducing exactly.

**The fleet earns nothing on this day.** Every unit is idle. That is a measured
result with a measured cause (§7), not a failure of the run, and nothing was
tuned to change it.

## 1. Inputs and provenance

| | |
|---|---|
| industry shapes | `data/processed/industry_typical_profiles.csv` |
| shapes SHA-256 | `0a19913e7d068390…` |
| price workbook | `docs/日前价格261009.xlsx` |
| workbook SHA-256 | `5fc8f9a8461b680fd38a6d2fefb3d2ff99f239c8bd13a6c77df27de3bb9612f3` |
| curve used | `261009`, index **0**, role `same_day_or_same_day_mean` |
| `is_measured` | **false** |
| interval convention | `label_is_interval_end`, `label_index == period_index` |
| units | CNY/MWh, `dt = 0.25 h` |
| scenario | `typical_day_industry` |
| acceptance driver | `src/accept_typical_day_industry.py` |
| raw report | `outputs/phase6_acceptance.json` |

**`261009` derives from real historical market data, but its aggregation is
not settled.** Its sheet name carries the date 26/10/09 and its column header
says `日前均值（CNY/MWh）`. Nothing in the file resolves whether it is that day's
96 actual period prices or a statistical day-mean, so it is **not** treated as
a confirmed daily series. In this report it is the *"261009 day-ahead price
scenario"* — a representative 96-period price scenario, never a verified daily
clearing. See `reports/final_research_summary.md` §2.

**All four curves were read**, and their metadata checked: `261009`, `7days`,
`30days`, `halfyear`, each 96 points, finite, CNY/MWh, `is_measured: false`.
Only `261009` prices the run; the other three were read to confirm the reader
and the curve-selection interface, as instructed.

**Curve selection is explicit and does not touch the global.** The scenario
declares `wholesale_curve_index: 0`; `_scenario_fingerprint`-style resolution
happens in `_build_scenario` and the caller's `day_id` still wins. The report
records `global_day_id_used: false`.

## 2. Scenario, mapping and load

| | |
|---|---|
| load agents | 32 (buses 1–32; bus 0 is the slack) |
| storage | `ESS9`, `ESS14`, `ESS17`, `ESS27` — 0.4 MW / 1.6 MWh each |
| PV / wind | 0.6 MW / 1.0 MW, unchanged |
| coincident peak | **3.715000 MW** |
| network nominal demand | 3.715000 MW |
| absolute error | < 1e-6 MW |
| industries used | 13, all above the reliability threshold |
| excluded by name | 分布式光伏 (a generator), 5G基站, 电解铝 |

The mapping is flagged `simulation_mapping: true`. **It is not a geographic
assignment**: this feeder carries no user-to-node connection table, and the
report says so. 分布式光伏 is excluded by name and enforced, so the workbook's
generation cannot re-enter as a device; PV and wind are still the 0.6/1.0 MW the
scenario declares.

The day is **IEEE 33-bus demand redistributed by industry shapes**, not measured
Jiangsu load. No absolute-MW conversion was attempted.

## 3. A defect the acceptance found, and its fix

The first run **failed**, not with a wrong number but with a raised error:

```
ValueError: wholesale_market.day_id 11 is not a declared dianyicha curve;
this source serves 4 (['261009', '7days', '30days', 'halfyear']), selected by
index. These are price scenarios, not calendar days.
```

**Cause.** `_build_scenario` priced the day from the scenario's declared curve —
but only for *its own* call. `BiddingEnv.reset()` re-derives the price for every
episode and reads the global `wholesale_market.day_id` when it is not told
otherwise. That global holds `11`, the synthetic library's reference day, which
on a `dianyicha` scenario is out of range.

The Phase-4 implementation passed the index as a **call argument** specifically
to avoid the global; that fixed one of the ten price-creation sites and left the
other nine reading the global. `get_scenario`'s own docstring says why they read
the global: *"the price has ten creation sites and threading a new argument
through all of them would be a refactor in service of one number."* The
Phase-4 choice was inconsistent with the design it was sitting in, and the
inconsistency is what surfaced here.

**Consequence had it not raised.** On `dianyicha` it could only raise, because
the index is out of range. On a **synthetic** scenario it would have priced the
wrong day **in silence** — the same latent bug, one config change away from
being invisible. That is worth stating plainly: the loud failure is the benign
case.

**Fix.** `_build_scenario` now writes the resolved index into the config cache,
exactly as it already does for `wholesale_market.source`:

```python
curve = day_id if day_id is not None else cfg.get("wholesale_curve_index")
if curve is None:
    curve = get_default("wholesale_market.day_id", 0)
set_default("wholesale_market.day_id", int(curve))
```

A caller's explicit `day_id` still wins, so `train_rl`'s day rotation is
unaffected. To close the stickiness that writing a global introduces, **every
price-bearing scenario now declares its own index** — `typical_day` gained
`wholesale_curve_index: 11` (the synthetic library's median-range reference day,
previously only a file default). The guard is
`test_a_build_does_not_price_the_next_scenario_from_a_stale_index`, which builds
the industry day and then the native day and asserts the native day is back on
the synthetic library at day 11.

The provider's refusal — raising rather than wrapping or defaulting — is what
made this findable. A provider that had quietly returned curve 0 would have
produced a plausible day built on a price nobody selected.

## 4. The 96 periods

| | |
|---|---|
| periods cleared | 96 / 96 |
| LMP fallbacks | **0** — every price is a nodal dual |
| settlement reported clean | **true** |
| clearing objective | 19,769.70 CNY |
| carbon, on net import | 23.43 tCO2 |

The 96-period horizon is contiguous and unshifted: the price reader asserts the
exact interval-END clock (`15 * (i + 1)` minutes at slot `i`), so slot `i` of
the price is slot `i` of the simulation with no offset.

### Solver provenance

The result now carries a `solver` block. (It did not before this supplement —
see §13 — so the first version of this report could not state which solver ran.)

| field | value |
|---|---|
| `engine` | `socp` |
| `primary.name` | `gurobi` |
| `primary.status_code` | `2` |
| `primary.status_name` | `OPTIMAL` |
| `primary.termination` | `optimal` |
| `used_fallback` | **false** |
| `fallback` | `null` |
| `final.name` | `gurobi` — identical to the primary, no fallback occurred |
| `duals_available` | **true** |
| `fell_back` key present | no — set only on the fallback path |

**`duals_available` is reported apart from `optimal`**, and the separation is
the point: a model can return OPTIMAL and still hand back no power-balance
dual, which is why `BarHomogeneous=1` is set and why `lmp_fallbacks` exists. A
caller that needs prices must ask about duals, not about the status; conflating
them would let a price-less day pass as a successful one. Here both are true and
`lmp_fallbacks` is 0, so the two answers agree — but they are two answers.

`final` names whichever solver actually produced the returned numbers. After a
fallback it names the **fallback**, never the primary whose answer was
discarded; `test_a_fallback_result_names_the_fallback_as_final` pins that by
forcing the batch to decline and asserting the primary's own status survives
alongside without being presented as the outcome.

**One behaviour worth recording:** the SOCP batch has no working fallback. If
`solve_socp_opf_batch` returns `None`, `clear_market` routes to the per-period
loop, and `solve_opf_gurobi` refuses `opf_mode='socp'` outright — so a degraded
SOCP day **raises** rather than silently degrading to a different engine. That
is the right failure mode (loud), but it means the `used_fallback: true` SOCP
result is unreachable; the fallback metadata was therefore verified on
`lindistflow`, which does support it.

## 5. Physics

| quantity | value |
|---|---|
| bus voltage range | 0.9697 … 1.0076 pu (band 0.93–1.05) |
| periods below / above the band | **0 / 0** |
| worst-voltage bus | 32 |
| max line utilisation | 0.1673 (worst line 1) |
| lines / periods over limit | **0 / 0** |
| grid import | 40.40 MWh |
| grid export | 0.00 MWh |
| simultaneous import and export | **0 periods** |
| curtailment | 2.1796 MWh |
| RE consumption rate | 86.86 % |

**No constraint is slack-relaxed and none was relaxed to make this pass.** The
day is comfortably inside the band — unlike the synthetic native day, whose
maximum line utilisation is 0.543 and whose voltage sits *on* the 0.93 floor.
The reason is the load shape: the industry curves are flatter than the
synthesised `native` profile, so the fleet's load factor is higher and its peak
stress lower. This is a property of the shapes, not a relaxation.

**The day-ahead market serves its own basis in full.**

| | |
|---|---|
| basis | `load_forecast` |
| forecast energy | 54.24 MWh |
| realized energy | 54.79 MWh |
| forecast error | +1.01 % |
| served | 54.24 MWh — equal to the forecast |
| unserved | 8.64e-08 MWh (solver noise) |

This distinction matters and is reported explicitly: the DA clearing dispatches
against `load_forecast`, so **unserved must be measured against the forecast,
not against the realized load**. Measuring it against `load_real` would
manufacture a 0.55 MWh "shortfall" that no constraint produced — that gap is
forecast error, which the real-time settlement exists to handle and which this
day-ahead acceptance does not settle.

## 6. Storage

Every unit idle for all 96 periods.

| unit | SOC start | SOC end | charge MWh | discharge MWh | transitions verified | max residual | worst slot |
|---|---|---|---|---|---|---|---|
| ESS9 | 0.500 | 0.500 | ~1.9e-07 | ~1.6e-07 | **96 / 96** | 8.72e-12 | 61 |
| ESS14 | 0.500 | 0.500 | ~1.9e-07 | ~1.6e-07 | **96 / 96** | 1.01e-11 | 41 |
| ESS17 | 0.500 | 0.500 | ~2.0e-07 | ~1.7e-07 | **96 / 96** | 7.83e-12 | 60 |
| ESS27 | 0.500 | 0.500 | ~1.6e-07 | ~1.3e-07 | **96 / 96** | 2.47e-11 | 62 |

The charge/discharge figures are solver noise around zero — no unit moved.

**All 96 transitions are now verified**, against the model's own state
variables. The recursion checked is
`boundary[t+1] - boundary[t] == (eta_ch·ch[t] - dis[t]/eta_dis)·dt/e_max` for
every `t` in `0..95`, with `dt = 0.25 h`, tolerance `1e-6`, worst residual
**2.47e-11** at slot 62 of ESS27.

### What changed, and why the previous report could only check 94

The model already carried `T + 1 = 97` state variables: `soc[0]` pinned to the
initial state, `soc[t+1]` the state after period `t`, and — on a whole-day clear
— `soc[T] == soc[0]`. Nothing in the optimisation had to change.

The **readout** was the problem. The per-period `soc` array is length `T`, and
its last entry was overwritten with the terminal state:

```python
s['soc'][t] = soc[nm][t].X        # t in 0..T-1
schedules[nm]['soc'][T - 1] = soc[nm][T].X   # overwrites index T-1
```

That convention makes `soc[-1]` mean "the state at the end of the day", which is
what an existing reader expects — and it discards `soc[T-1]`, leaving the final
transition unverifiable. So the fix **adds** rather than changes: the readout
keeps its meaning exactly, and a new `soc_boundary` carries all `T + 1` states
with none overwritten.

| field | meaning |
|---|---|
| `soc` (unchanged) | length `T`; index `t` is the state at the start of period `t`, **except index `T-1`, which holds the terminal state** |
| `soc_boundary` | length `T + 1`; `[0]` initial, `[t+1]` the state after period `t` |
| `soc_terminal_target` | what the horizon was pinned to, or `None` on a window |
| `soc_terminal_actual` | what the solve reached; never the target substituted for it |

`LinDistFlow` needed no equivalent repair — it already kept the terminal state
in a separate `soc_final` key rather than overwriting. It now exposes
`soc_boundary` and the two terminal fields as well, so a reader need not know
which engine cleared the day; it also retains `soc_final` for existing readers.

`soc_terminal_actual` is asserted to equal `soc_boundary[T]`, so the two cannot
drift. On this day the target and the actual are both 0.500 because the horizon
is pinned and the fleet is idle; the fields are separate so that a windowed
clear can report `None` for the target rather than a value, which
`test_a_rolling_window_reports_no_terminal_target` checks.

## 7. Why the fleet is idle

The clearing charges a cycle at its derived floor,
`cycle_cost * (1/eta_ch + eta_dis) = 200.6957` CNY/MWh of throughput. (This
report originally wrote `2 × cycle_cost = 200`; the derivation is in
`reports/final_research_summary.md` §5.) A
unit can only profit if its own bus delivers more than that across the day:

| unit | bus | LMP min | LMP max | round-trip margin | required | shortfall |
|---|---|---|---|---|---|---|
| ESS9 | 9 | 234.30 | 463.86 | 172.08 | 200.70 | **−28.62** |
| ESS14 | 14 | 235.39 | 466.10 | 172.96 | 200.70 | **−27.74** |
| ESS17 | 17 | 236.33 | 468.07 | 173.74 | 200.70 | **−26.96** |
| ESS27 | 27 | 236.64 | 463.53 | 169.22 | 200.70 | **−31.47** |

Every unit falls short by 27–31 CNY/MWh. The cause is structural and was
anticipated: the real Jiangsu mean curve spans 225.90–445.00 CNY/MWh, against a
`cycle_cost` of 100 per leg inherited from a synthetic market configured with a
120–1000 band. **The window is closed, so the fleet correctly does not move.**

`cycle_cost`, the prices, the floor and cap are all reported exactly as
configured. **Nothing was adjusted to open the window**, and the test
`test_the_result_is_not_tuned_to_produce_a_profit` pins the cycle cost.

## 8. Settlement

Both settlement paths were run and compared.

| | path A | path B |
|---|---|---|
| entry | `counterfactual_eval.settle_day` | `eval_agents.run_episode` + `settle_capture` |
| unit payoffs | all ≈ −2.0e-05 | identical |
| max difference | **0.00e+00** | |
| clean | true | true / capture usable |

The two entry points — the direct settled path and the rolling episode the
evaluation CLI uses — agree to the last bit. This is the Phase-5 property
checked on a real day rather than a fixture.

Breakdown per unit (CNY); all four are numerically identical in structure:

| term | value |
|---|---|
| revenue | +5.0e-05 |
| purchase cost | −3.5e-05 |
| generation cost | 0 |
| degradation cost | −3.6e-05 |
| **`other_cost`** | **0.0 — never assigned** |
| penalty | 0 |
| **total** | **−2.1e-05** |

Fleet reconciliation: the sum of the settled per-unit profits (−8.192148e-05)
equals the sum of independently computed `PayoffBreakdown.total` values to the
bit. Every unit reconciles term by term, and every unit's breakdown matches its
settled figure.

**`other_cost` is reported as zero and is not silently filled.** No out-of-model
charge was estimated to make the ledger look complete. As
`reports/jiangsu_market_mechanism.md` records, the breakdown covers revenue,
purchase, generation, degradation and penalty, and **does not** cover 输配电价,
线损, 系统运行费, 政府性基金及附加, deviation charges, capacity or cost
compensation, or imbalance funds. The figure here is a **modelled storage
day-ahead energy profit**, not a settlement bill.

The rolling reward was recorded separately (2.5e-05 … 4.1e-05). It is reported
under its own name and **is not** the KPI: on this idle day it is a different
number from the settled payoff, which is exactly why Phase 5 separated them.

## 9. Repeatability

| check | result |
|---|---|
| runs | 2 |
| prices identical | **true** (exact array equality) |
| peak identical | **true** |
| profits identical | **true** |
| all clean | **true** |

The repeat resets `reload_defaults()`, the price provider cache and the baseline
cache between runs, so a stale global index, a cached provider or a cached
baseline would all have shown up as a difference.

## 10. Tests

**New:** `tests/test_end_to_end_acceptance.py`, **38 tests** — 26 over one
shared acceptance run, plus 12 added by the supplement (§13) covering solver
provenance and the SOC trajectory.

```
python -m pytest tests/test_end_to_end_acceptance.py -v          -> 38 passed (45s)
python -m pytest <clearing / storage / payoff / settlement, 10 files>
                                                                 -> 108 passed, 15 deselected
python -m pytest tests/                -> 737 passed, 51 deselected, 0 failed (6m42s)
```

737 = 725 at the end of the first Phase-6 pass, plus the 12 new.

Coverage maps to the required list: the scenario builds and reads the industry
curves; the declared `dianyicha` curve is the one that prices the day; all 96
periods run with an unshifted time axis; the load peak is conserved; SOC and
power limits hold and the recursion closes; the day's money comes from the
shared settlement; a rebuild does not price the next scenario from a stale
index; and the native and legacy scenarios are unchanged.

The supplement's 12 add: the result names the solver that produced it; dual
availability is reported apart from optimality; every status code the classifier
knows has a readable name; infeasible and limit-reached are not conflated; a
fallback result names the **fallback** as `final` while keeping the primary's
own status; the boundary array has `T + 1` entries; **all 96 transitions verify
on a moving day**; the fleet on that day actually moves (a guard on that guard);
the readout's old convention is unchanged; the terminal target and actual are
separate fields; a windowed clear reports no target; and `LinDistFlow` carries
the same schema.

**The regression that matters.** `test_every_transition_verifies_on_a_moving_day`
runs on `typical_day`, whose fleet cycles (12.16 MWh charged), and asserts two
things: the boundary array closes all 96 transitions, **and** the per-period
readout does *not* — its last entry is the terminal state, so stepping it fails
where the boundary holds. On the industry day every state is 0.500 and that
difference is invisible; the test would pass on a constant array and prove
nothing. The fixture is chosen so it cannot. If the overwrite were reintroduced
into the boundary fill, the final transition would stop closing and the test
fails.

The acceptance deliberately **does not assert the day is profitable**. A test
demanding that would be asking for exactly the tuning this project forbids.
Instead `test_the_idle_fleet_is_explained_by_the_arbitrage_window` asserts that
the idleness and the closed window are consistent — if the fleet were ever idle
while the window was open, that test fails and the diagnosis is wrong.

## 11. Warnings, open items, and what cannot be claimed

1. **The fleet earns nothing here, and the reason is the price scenario.**
   Nothing may be concluded about storage profitability in Jiangsu from this
   day. It says the *261009 mean curve* does not cover a 100 CNY/MWh-per-leg
   cycle cost at these buses. Whether a real day would is untested — no real
   daily series exists.
2. **The 0.55 MWh forecast error is unsettled.** This is a day-ahead-only
   acceptance; the real-time leg is out of scope, so the realized load exceeding
   the forecast is reported and not resolved.
3. ~~Solver status is not in the result schema.~~ **Resolved by the supplement
   (§13)** — the result now carries engine, primary status code and name,
   termination class, fallback status and final-solver identity, with dual
   availability reported as a separate question.
4. ~~Two of 96 SOC transitions cannot be verified.~~ **Resolved by the
   supplement (§13)** — all 96 verify against a `T + 1` boundary array; the
   per-period readout is unchanged for existing readers.
5. **`other_cost` remains zero and the cost scope is incomplete**, exactly as
   the Phase-3 market document and the Phase-5 audit record. No claim of a
   complete settlement bill is made anywhere in this report.
6. **`261009` is a mean, not a measured day.** Every number above is a
   price-scenario result. It is not evidence about day-to-day behaviour,
   volatility, or generalisation across days.
7. **One scenario, one curve, one strategy.** Only `fixed` was run. No ranking
   of `mpc`, the greedy threshold strategy or MATD3 is offered, and no MATD3
   checkpoint was loaded.

## 12. Workspace

HEAD `a1151fd2ab52e4a49a610173a1bba04c56bda563`, branch `master`, unchanged.
Backed up to
`C:/Users/26036/AppData/Local/Temp/phase6_backup.diff` before any edit. No
`reset`, `checkout --` or `clean` was run. `gurobi.env` was not read, copied or
printed.

Line-by-line comparison of every pre-existing hunk against the backup: all
preserved. **One apparent loss was investigated and cleared** — seven "added"
lines in `src/scenarios.py` no longer match the backup, and all seven are **my
own Phase-4 text** (the `day_id=` call argument and its comment) which the
Phase-6 fix superseded. The user's original diff for that file contained **zero
added lines**, and none of the seven appears in it. Nothing of the user's was
altered.

The supplement backed up to
`C:/Users/26036/AppData/Local/Temp/phase6b_backup.diff` before its own edits and
repeated the comparison over 24 files, distinguishing an addition of mine that
was later superseded from an addition of the user's that was lost. Result: **all
user changes preserved**. `src/dispatch_core.py` carries the user's own
uncommitted deletion of `StorageConstraints.apply_ramp`; it is intact, and the
one extra removal in that file's diff is a line I replaced inside
`empty_schedules`. `gurobi.env` was not read, copied or printed, and no
destructive git command was run.

## 13. Supplement: solver provenance and the SOC trajectory

Two gaps the first pass recorded as limitations are closed. Neither changed a
price, an SOC bound, a capacity, a cycle cost or a bid threshold, and the
acceptance day is the same one.

### What was added

| file | change |
|---|---|
| `src/dispatch_core.py` | `GUROBI_STATUS_NAMES`, `categorize_termination`, `gurobi_solver_record`, `fallback_solver_record`, `solver_metadata`, and the shared last-solve record; `empty_schedules` gains `soc_boundary` and the two terminal fields |
| `src/dispatch_socp.py` | records its outcome on success and on failure; fills `soc_boundary` without overwriting anything; attaches the `solver` block |
| `src/dispatch_ldf.py` | same schema and same solver block; keeps `soc_final` |
| `src/market.py` | the per-period fallback now names what failed and what ran, and fills the boundary on its own path |
| `src/accept_typical_day_industry.py` | verifies all 96 transitions against `soc_boundary`; reports the solver block |
| `tests/test_end_to_end_acceptance.py` | 12 new tests |
| `reports/phase6_end_to_end_acceptance.md` | this section, and §4/§6/§10/§11 corrected |

**Nothing was changed to make the day pass.** The fleet is still idle, the
arbitrage window is still closed by 26–31 CNY/MWh, and the settled fleet profit
is still ~−8.2e-05 CNY. The only numbers that moved are the ones the new output
makes visible.

### How the schema stays compatible

Every addition is a **new key**; no existing key changed meaning or shape.

- `soc` keeps its length and its convention, including the last entry holding
  the terminal state.
- `fell_back` still appears, and only on the fallback path.
- `lmp_fallbacks` is unchanged and still the price-side statement.
- `LinDistFlow`'s `soc_final` is retained.

### The solver-status failure mode that was found while testing

The SOCP batch has no working fallback: `solve_opf_gurobi` refuses
`opf_mode='socp'`, so a `None` from the batch raises rather than degrading. The
fallback metadata was therefore exercised on `lindistflow`, where it does work,
and the test forces the batch to decline there. The SOCP `used_fallback: true`
branch is unreachable **by design** and is not claimed to be tested.

The last-solve record is module-level, which this project has been careful about.
It is explicitly a *last-call* cache: overwritten on every solve, never read to
decide anything, and it exists only so a failure can be reported with its own
status code rather than as "the batch returned nothing".

## Status

Phase 6 complete. Stopping here for review; Phase 7 is not started.
