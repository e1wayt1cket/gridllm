# Phase 7 — Arbitrage Reference Bound and Three-Strategy Comparison

Four price curves, three bidding strategies, one frozen configuration. No MATD3
policy was trained or loaded, and no capacity, siting or threshold was moved.

## The result, in one paragraph

**Every unit is idle under every strategy on every curve, and the
perfect-foresight reference is exactly zero on all four.** The fleet cannot
profitably cycle because the round-trip margin the real Jiangsu mean curves
offer (52.6–163.9 CNY/MWh) is below what the clearing charges for a cycle
(200.6957 CNY/MWh). This is not a failure of any strategy: **the perfect forecaster, who
knows the whole day in advance and faces no network, no competition and no
bidding, also earns nothing.** A closed window is the finding, and it is stated
as an inequality a test can re-check rather than as a number in a table.

## 1. Configuration

| | |
|---|---|
| scenario | `typical_day_industry` (IEEE 33-bus at its own 3.715 MW demand) |
| fleet | 4 × (0.4 MW / 1.6 MWh), buses 9, 14, 17, 27 |
| `eta_ch` / `eta_dis` | 0.92 / 0.92 |
| SOC band | 0.10 – 0.90, start 0.50, terminal pinned to start |
| `cycle_cost` | 100 CNY/MWh **per leg** → 200.6957 required per cycle |
| horizon | 96 × 15 min, `dt = 0.25 h` |
| price source | `dianyicha`, all four sheets |
| settlement | `counterfactual_eval.settle_day` |
| driver | `src/phase7_strategy_comparison.py` |
| raw output | `outputs/phase7_strategy_comparison.json` |

**All four curves derive from real historical market data.** Three are means,
each over the window its role names; `261009`'s aggregation is unresolved (see
`reports/final_research_summary.md` §2.1). They are four **price scenarios**,
not four dates. Nothing here is a cross-day statistic and no cross-day claim is
made.

## 2. Strategy audit: what each one actually uses

| strategy | what it reads | look-ahead | how it declares |
|---|---|---|---|
| `fixed` | nothing — every participant at its own spec's truthful point | none | storage at `(0, 0)` |
| `mpc` | an LMP forecast over the next `H = 8` periods, from `market_history["price"]` if a caller supplies one else the day-ahead curve | **yes, 8 periods** | a corner of the quote box, chosen by the MPC's charge/discharge decision |
| `greedy_threshold` | **the current period's price only** | **none** | a corner of the quote box, chosen by one threshold comparison |

All three return **reservation prices in each agent's own action space**; none
writes `p_ch`, `p_dis` or `soc`, and the clearing decides every traded quantity.
`greedy_threshold` did not exist before this phase and is added in
`src/strategies/greedy_bidding.py`.

**The quote convention, verified rather than assumed** (from
`mpc_storage_bidding`, and matched by the new strategy): a unit asks *little*
for the leg it wants and *much* for the leg it does not.

| intent | `charge_price` | `discharge_price` |
|---|---|---|
| wants to charge | low (0) | high (300) |
| wants to discharge | high (150) | low (0) |
| neutral | truthful (0) | truthful (0) |

`test_the_corners_match_the_documented_convention` pins this.

**What a mean curve makes of this.** The MPC reads the day-ahead curve as if it
were a known price shape. On a *mean* curve that shape is an average over a
window, smoother than any single day. So the MPC is looking ahead through a
statistic, and the greedy is thresholding a statistic. Neither is reading a day.
The MPC's forecast is also the **external** curve, while the unit is settled at
its **nodal** LMP — the two differ by the network's congestion and loss terms,
so the MPC is optimising against a forecast of the settlement price, not the
settlement price.

### The thresholds

| | |
|---|---|
| `greedy.threshold_source` | `fixed` |
| `greedy.low_threshold` | 300.0 |
| `greedy.high_threshold` | 400.0 |
| `greedy.threshold_unit` | CNY/MWh |

Pre-registered from the real price band so that the low threshold sits in the
midday trough and the high one above the night plateau. **Not fitted to
outcomes, and not re-read from the curve under evaluation** — a test asserts the
thresholds do not move when a curve is read.

## 3. The arbitrage reference bound

`arbitrage_lp.arbitrage_lp`, audited before use. Its objective is

```
maximise  Σ_t price[t] · (dis[t] − ch[t]) · dt  −  cycle_cost · (ch[t] + dis[t]) · dt
```

subject to the SOC transition, the power limits, the SOC band, and
`soc[T] == soc[0]`. Both legs are costed, which is why the requirement is
the derived floor `cycle_cost * (1/eta_ch + eta_dis)`. Units are MW / h / MWh /
CNY/MWh throughout, and `soc` is a
fraction of `e_max` carrying the same `dt / e_max` factor the clearing writes.

| | |
|---|---|
| battery | matches the fleet: 0.4 MW, 1.6 MWh, 0.92/0.92, SOC 0.10–0.90 from 0.50 |
| terminal SOC | pinned to the initial state, as the settlement pins it |
| horizon | 96 × 15 min, `dt = 0.25 h` |
| solver | HiGHS via `scipy.linprog`, presolve off |

### What it is not

**It is the storage-side perfect-foresight arbitrage reference, not an upper
bound on what a strategy in this market can earn.** It contains:

- **no IEEE 33-bus network** — no voltage band, no line limits, no losses;
- **no clearing and no bidding** — the unit trades its whole schedule at the
  price it is given, and no quote is ever refused;
- **no competitors** — the other three units, and every load and prosumer, are
  absent, so the price is not moved by the unit's own dispatch.

In this market a unit is settled at its **nodal** LMP through a radial network
it shares with three other batteries. That is a different object, and the
reference is reported beside the strategies rather than as a target they are
shown to reach. Nothing below claims any strategy attains it.

### The bound

| curve | band min | band max | round-trip margin | required | bound / unit | bound / fleet |
|---|---|---|---|---|---|---|
| `261009` | 225.90 | 445.00 | 163.86 | 200.70 | **0.000000** | **0.000000** |
| `7days` | 257.50 | 477.87 | 159.75 | 200.70 | **0.000000** | **0.000000** |
| `30days` | 333.08 | 450.86 | 52.75 | 200.70 | **0.000000** | **0.000000** |
| `halfyear` | 284.93 | 393.86 | 52.64 | 200.70 | **0.000000** | **0.000000** |

`round-trip margin = eta_dis · max − min / eta_ch`; `required` is the derived
floor `cycle_cost * (1/eta_ch + eta_dis)` = 200.6957, not the lossless
`2 * cycle_cost`. Every curve fails, by between 36.8 and 148.1 CNY/MWh.

**A zero bound demands a control**, because an LP that silently stopped working
would produce it too:

| control input | bound / unit |
|---|---|
| synthetic band 120 → 1000 CNY/MWh | **344.29 CNY** (0.696 MWh charged, 0.401 cycles) |
| flat 500 with a 200 trough | **30.85 CNY** (0.800 MWh charged) |

The same LP returns a large positive number when a window exists.
`test_the_lp_is_not_silently_returning_nothing` pins both controls beside the
real curves' zero, so the zero cannot be mistaken for a broken solver.

**The nodal bound is also zero.** Pricing the LP on each unit's own cleared
nodal LMP — the price it is actually settled at — gives 0.0000 for the fleet on
all four curves. The network's congestion and loss terms, which make the nodal
band wider than the wholesale one, still do not create a spread this battery can
clear `cycle_cost` against.

## 4. The paired comparison

Same day, same fleet, same price curve within each block; only the strategy
changes.

| curve | strategy | fleet settled profit (CNY) | declared intent chg/dis/neu | traded chg/dis (MWh) | solver | duals |
|---|---|---|---|---|---|---|
| `261009` | `fixed` | −8.19e-05 | 0 / 0 / 96 | 1.9e-07 / 1.6e-07 | OPTIMAL | yes |
| `261009` | `mpc` | −1.05e-04 | 7 / 1 / 88 | 2.4e-07 / 2.1e-07 | OPTIMAL | yes |
| `261009` | `greedy_threshold` | −2.54e-04 | 21 / 63 / 12 | 7.4e-07 / 6.3e-07 | OPTIMAL | yes |
| `7days` | `fixed` | −2.09e-04 | 0 / 0 / 96 | 3.9e-07 / 3.3e-07 | OPTIMAL | yes |
| `7days` | `mpc` | −2.04e-04 | 7 / 2 / 87 | 3.8e-07 / 3.2e-07 | OPTIMAL | yes |
| `7days` | `greedy_threshold` | −1.60e-04 | 16 / 64 / 16 | 3.5e-07 / 3.0e-07 | OPTIMAL | yes |
| `30days` | `fixed` | −1.15e-04 | 0 / 0 / 96 | 1.3e-07 / 1.1e-07 | OPTIMAL | yes |
| `30days` | `mpc` | −2.92e-04 | 3 / 1 / 92 | 3.3e-07 / 2.8e-07 | OPTIMAL | yes |
| `30days` | `greedy_threshold` | −7.90e-05 | 0 / 63 / 33 | 9.5e-08 / 8.1e-08 | OPTIMAL | yes |
| `halfyear` | `fixed` | −1.83e-04 | 0 / 0 / 96 | 2.2e-07 / 1.8e-07 | OPTIMAL | yes |
| `halfyear` | `mpc` | −1.83e-04 | 0 / 0 / 96 | 2.2e-07 / 1.8e-07 | OPTIMAL | yes |
| `halfyear` | `greedy_threshold` | −1.66e-04 | 18 / 0 / 78 | 2.0e-07 / 1.7e-07 | OPTIMAL | yes |

**Those profit figures are not economic quantities.** They are worth reading
next to the throughput column: 1e-7 MWh is **1.6e-8 of one real cycle** (a full
fleet cycle is 6.4 MWh). Every "profit" is the degradation cost of solver noise,
i.e. `−cycle_cost × ~1e-7`, and the spread between strategies (~1e-4 CNY) is
float residue. **No strategy earns anything, and no ranking may be read from
these columns.**

`settled_clean` is `true` for all twelve runs, every unit's breakdown
reconciles with its settled figure, and `other_cost` is 0 as always.

### The intent columns are where the strategies differ

This is the one place the three are actually distinguishable, and the tally is
read off the quotes each declared, not inferred from the price series:

- **`fixed`** declares the truthful corner in all 96 periods, on every curve, by
  construction. It never asks for anything.
- **`mpc`** solves its 8-period look-ahead LP and finds almost nothing: 7, 7 and
  3 periods of charge intent on the first three curves, and **nothing at all on
  `halfyear`**. Its occasional single discharge-intent period is the LP's
  numerical residue, not a decision.
- **`greedy_threshold`** declares a great deal — 84, 80 and 63 periods of
  directional intent on the first three curves — and **the clearing refuses
  every one of them**. Its throughput is the same ~1e-7 MWh as `fixed`'s.

**So the answer to "does the threshold rule trade often or stay neutral?" is:
it asks often, and it is never filled.** The rule is not idle; the market is
closed to it.

### The fixed thresholds meet each curve differently

A structural observation, and a caution about carrying one threshold pair
across scenarios:

| curve | band | periods below 300 | periods above 400 |
|---|---|---|---|
| `261009` | 225.9 – 445.0 | 21 | 63 |
| `7days` | 257.5 – 477.9 | 16 | 64 |
| `30days` | 333.1 – 450.9 | **0** — the floor is above the low threshold | 63 |
| `halfyear` | 284.9 – 393.9 | 18 | **0** — the ceiling is below the high threshold |

On `30days` the rule can only ever want to discharge; on `halfyear` it can only
ever want to charge. The same declared thresholds mean qualitatively different
things across the four curves — which is a property of the curves, and is
reported rather than corrected by re-tuning the thresholds per curve (that would
be fitting the baseline to the scenario it is meant to be a baseline for).

## 5. Constraint diagnostics

Identical across strategies within each curve, because nothing is traded:

| curve | voltage range | band violations | max line util. | over-limit | unserved | curtailment | SOC 96/96 |
|---|---|---|---|---|---|---|---|
| `261009` | 0.9697 – 1.0076 | 0 / 0 | 0.1673 | 0 | 8.6e-08 | 2.180 | yes |
| `7days` | 0.9702 – 1.0076 | 0 / 0 | 0.1621 | 0 | 2.2e-07 | 1.611 | yes |
| `30days` | 0.9711 – 1.0076 | 0 / 0 | 0.1580 | 0 | 7.8e-08 | 0.155 | yes |
| `halfyear` | 0.9711 – 1.0076 | 0 / 0 | 0.1580 | 0 | 1.7e-07 | 0.916 | yes |

Band `0.93 – 1.05` pu. No violation, no line near its limit, no unserved energy
above solver noise, every SOC transition verified on the `T+1` boundary array.
Curtailment varies by curve (0.155 – 2.180 MWh) because the clearing trades
curtailment against import at the price it faces; that is a price effect, not a
strategy effect, and it is the same for all three strategies on a given curve.

## 6. Answers to the questions asked

**Is there exploitable arbitrage at this price band, net of efficiency and cycle
cost?** **No.** The perfect forecaster earns zero on all four curves, and the
nodal-priced bound is zero too. The margin falls short by 36–147 CNY/MWh.

**Does truthful bidding produce any trade?** **No.** The fleet is idle under
`fixed` on every curve, and its settled profit is float noise.

**Where would an MPC gain come from?** Nowhere on these curves. Its look-ahead
LP finds almost nothing to do, and its near-zero intent counts *agree* with the
independent perfect-foresight LP — two different methods, the same conclusion.
That agreement is the strongest single piece of evidence here that the window is
closed rather than that a strategy is failing.

**Does the threshold rule trade often or stay neutral?** It declares heavily and
trades nothing (§4).

**What explains the gap between realised profit and the reference?** On these
curves there is no gap to explain: both are zero. The reference's extra
assumptions (no network, no clearing, no competition) would matter only if the
window were open.

**Why can a strategy declare and still earn zero?** Because the clearing charges
its derived floor on throughput a spread of 52–164 cannot cover. Declaring an
aggressive corner makes a unit *willing*; it does not make the cycle *profitable*,
and the clearing will not take a trade that loses money at its own prices.

**Observation versus causation, kept apart.** It is *observed* that all three
strategies settle at ~0 and that the greedy declares most. It is *inferred*, from
the margin arithmetic and the LP control, that the cause is the price band being
too narrow for the declared cycle cost. No ranking of strategies is offered:
with every unit idle, the twelve runs are economically the same run, and
"`fixed` beat `mpc` on `261009`" would be a statement about the third decimal of
a floating-point residue.

## 7. Tests

**New:** `tests/test_phase7_strategy_comparison.py`, 21 tests.

```
python -m pytest tests/test_phase7_strategy_comparison.py -v   -> 21 passed (8.8s)
python -m pytest tests/   -> 758 passed, 51 deselected, 0 failed (6m55s)
```

758 = 737 at the end of Phase 6, plus the 21 new.

Coverage maps to the required list:

| requirement | test |
|---|---|
| thresholds fixed and reproducible | `test_the_thresholds_come_from_configuration_and_are_reproducible`, `..._do_not_move_with_the_curve_being_evaluated` |
| the greedy reads no future price | `test_a_periods_quote_does_not_depend_on_later_prices` (perturbs the future), `test_the_rule_is_a_pure_function_of_the_current_price` |
| threshold actions go through the quote and clearing interface | `test_the_strategy_returns_quotes_in_each_agents_own_action_space`, `test_the_strategy_declares_no_power_variables` |
| all three share one settlement path | `test_every_strategy_is_settled_by_the_same_path` |
| curves are not overwritten by a stale global or cache | `test_the_price_scenario_is_not_taken_from_a_stale_cache` |
| an unclean settlement is not written as zero | `test_an_unclean_settlement_is_never_reported_as_zero` |
| phases 1–6 interfaces intact | `test_phase_1_to_6_interfaces_are_intact`, `test_the_registered_strategies_are_exactly_the_expected_set` |
| the zero is not a broken LP | `test_the_lp_is_not_silently_returning_nothing` (two controls), `test_the_real_curves_close_the_arbitrage_window` |

The look-ahead test perturbs every price after period 10 and asserts the first
11 quotes are **bit-identical**, then asserts the perturbation did land — so it
cannot pass by doing nothing. The window test asserts the *inequality*
(`margin < 2 · cycle_cost`), so if a future curve ever reopened the window the
test fails and names the report that needs revisiting.

## 8. What may and may not be claimed

**May be claimed:**

- On four Jiangsu day-ahead **mean** curves, a 0.4 MW / 1.6 MWh, 92%-round-trip
  battery cannot profitably arbitrage against a 100 CNY/MWh-per-leg cycle cost.
- All three baselines therefore settle at economically zero, and the truthful
  baseline trades nothing.
- The threshold rule declares directional intent in most periods and is filled
  in none, which is a statement about the clearing's prices rather than about
  the rule's willingness.
- The IEEE 33-bus day is served within its voltage band and line limits under
  every strategy and curve, with no meaningful unserved energy.

**May not be claimed:**

- Anything about MATD3. No policy was trained, loaded or evaluated.
- Anything about **cross-day** behaviour. Four mean curves are four price
  scenarios; there is no measured daily series in this project, so there is no
  variance to estimate, no positive-rate to report, and no generalisation to
  argue.
- Anything about **Jiangsu storage economics in practice**. A different
  `cycle_cost`, a different battery, a real day's volatility rather than a
  mean's smoothness, or market rules other than this model's would change the
  answer; none of them was varied here.
- That any strategy is **generally better** than another. With every unit idle
  the runs are economically identical.
- That the reference bound is the strategies' attainable ceiling. It ignores the
  network, the clearing and the other participants, and is reported as a
  reference, not a target.

## 9. Workspace

HEAD `a1151fd2ab52e4a49a610173a1bba04c56bda563`, branch `master`, unchanged.
Backed up to
`C:/Users/26036/AppData/Local/Temp/pre_phase7_worktree.diff` before any edit. No
`reset`, `checkout --` or `clean` was run; `gurobi.env` was not read, copied or
printed.

Every one of the 27 modified files was compared hunk by hunk against the backup,
with each "lost" line tested against the user's own original diff to tell a
superseded line of mine from a lost line of theirs. **No line of the user's was
lost, and none was reported as a candidate.** The files edited this phase are
`config/defaults.yaml` (a new `greedy:` block) and
`src/strategies/__init__.py` (one import); `src/strategies/greedy_bidding.py`,
`src/phase7_strategy_comparison.py` and the test module are new.

## Status

Phase 7 complete. Stopping here for review; Phase 8 is not started.
