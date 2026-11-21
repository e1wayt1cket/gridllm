# Local day-ahead RL stack: design

Date: 2026-11-17. Covers workstreams B (market mechanism), C (RL stack
migration) and D (protocol v7). The data layer is
`docs/superpowers/specs/2026-11-17-candidate-day-library-v3-design.md`; this
document consumes its output and does not restate it.

**Build order.** This document cannot be frozen before the v3 library is
frozen. The day source, the split, the holiday block and the feasible-day
count all come from it.

## Intent

The research main line clears on `market.clear_market` / `dispatch_socp`: a
single relaxed SOCP with the cycle cost inside the objective, no
charge/discharge exclusivity, duals read as prices, and storage quoting a
price with a fixed 0.4 MW quantity. The local day-ahead engine
(`local_da_clearing` / `da_market_*`) clears an exact MIQCP with exclusivity
and prices a fixed-direction convex program. Two mechanisms, one training
loop, and only one of them is the market the paper claims to study.

This design moves training, evaluation and the protocol onto the local
engine, and grows the storage action from a price pair to a price-and-quantity
pair so the agent can withhold capacity.

## Goals

1. Every training block clears on `local_da_clearing`, not `clear_market`.
2. A storage action expresses both a reservation price and an offered
   quantity per direction.
3. The reward is the storage participant's ledger payoff, not the clearing
   objective.
4. A protocol version that names the new engine, action, observation and
   prices, and can be audited end to end.

## Non-goals

- No change to the network, the voltage band or the library.
- No real-time, intraday or ancillary market.
- No change to the hidden layers or the training algorithm.
- No external reactive device.

## Anti-goals

- Training a policy whose reward comes from an objective the participant is
  not paid.
- Presenting an unprobed dual as a qualified settlement price.
- Letting a failed clear silently zero a reward and call it a bad episode.
- Selecting training days by whether they clear without recording that the
  selection happened.
- Selecting validation or test days by anything measured after the split.

## Constraints

- One episode is one day, 96 periods; a block commits 4 periods and looks 16
  ahead, so a block clears a 20-period window.
- Terminal SOC is free. Its value enters the day-end payoff explicitly,
  below.
- Nothing about the actual day-ahead price may enter a decision before that
  period is reached.
- The v3 library supplies the day: `bus_active_load_mw.npz`,
  `bus_reactive_load_mvar.npz` and `split.json`'s load|price pairs.
- The market's declared reactive asset (buses 17/31/32, 2.5 MVAr, zero active
  output) is carried by `local_da_days.window_day`, so a training window clears
  the same feeder the screening unit cleared.

## Approaches

1. **Keep `clear_market` for training, use the local engine only for
   evaluation.** Cheapest, changes no code path. Rejected: the agent would
   optimise a market the paper does not claim to study, and the evaluation
   engine's prices would never reach the policy.
2. **Port `rl_env` behind a thin adapter (chosen).** The block loop, the
   differential baseline, the gate, the shaping and the audit plumbing stay;
   `clear_market(...)` becomes a `DayAheadInput` build plus
   `ClearingModel`/`choose_directions`/`ClearingModel(prices=True)`.
3. **Write a new environment.** Rejected: it discards the reward, gate and
   audit plumbing that already passed phase gates, for no gain over 2.

## Decided

| Decision | Value | Basis |
| --- | --- | --- |
| Action | 4 quote dimensions per unit: charge price, charge quantity, discharge price, discharge quantity | Quantity bidding is the mechanism's content; price-only cannot express withholding |
| Gate | Unchanged, 2 per unit, 8 total | A gate opens a leg; the leg carries a price and a quantity together |
| Action spec | `storage_quote_v5`, 16 quote dimensions | Any change to a frozen action name is a protocol change |
| Charge-leg semantics | Unchanged: `charge_price` is a credit per MWh charged, not a willingness-to-pay | Requires a mechanism change to fix; this design keeps the mechanism and states what the dimension is |
| Observation | `v7_12d`, still 12 dimensions, 5 unique + 7 shared | The bus LMP is the mechanism's output; a fleet mean erases it |
| Price used in training | `local_da_clearing` direction-conditional duals | The ledger pays those; a second price stack would be a different market |
| Formal price qualification | `da_market_pricing.qualify_prices(backend="cutting")` on every validation and test day plus the three smoke days | Those days are fully cleared at evaluation anyway, so the audit is free |
| Probe sampling | One block per day, fixed index 11 | Block 23 has no lookahead; see C |
| Terminal inventory | `(E_96 - E_0) * external_price[95]`, once, in the day-end block | Matches `settle_day`'s inventory term |
| Baseline | One re-clear per block with every storage unit truthful | Fleet-truthful, not per-agent; see D |
| Day source | v3 library pair | The library exists to be trained on |
| Feasibility filter | Train only, recorded as `feasibility_selected` | A day that cannot clear has no payoff to learn from; the exclusion is declared, not hidden. The screen it reads predates the declared reactive asset, so any day it dropped is re-checked against the current default before it stays dropped |

## A. Action contract

Three different counts exist and must not be conflated:

| Quantity | Today | v5 |
| --- | --- | --- |
| Quote action dimensions (`ActionSpec`) | 8 (4 units x 2) | **16** (4 units x 4) |
| Gate head | 8 (4 units x 2) | 8 |
| Policy logits per unit (`da_quote_map.POLICY_DIM`) | 4 (2 gate + 2 quote) | **6** (2 gate + 4 quote) |
| Policy vector | 16 | **24** |

The actor's hidden layers are unchanged; its quote head widens from 8 to 16.
The gate head is unchanged and is a separate output.

Per unit the policy logits are, in order, `(g_charge, g_discharge,
charge_price, charge_quantity, discharge_price, discharge_quantity)`. With `g`
the existing linear-core / quadratic-tail map:

```
charge_price     = g_charge     * 150 * g(z_cp)      if g_charge     else 0
charge_quantity  = 0.4 * (1 - g_charge     * (1 - g(z_cq)))
discharge_price  = g_discharge  * 300 * g(z_dp)      if g_discharge  else 0
discharge_quantity = 0.4 * (1 - g_discharge * (1 - g(z_dq)))
```

Closed gate gives `price = 0`, `quantity = 0.4`. A single closed gate must
not be read as "price-taker"; see the next section. Open gate gives a positive
price and a quantity in `(0, 0.4)`, so withholding is expressible. The declared
quantity is a hard upper bound in the clearing (`local_da_clearing` sets
`ch.UB` / `dis.UB` from it).

Bounds and the declared truthful point:

| Dimension | Bounds | Truthful |
| --- | --- | --- |
| charge price | 0 .. 150 | 0 |
| charge quantity | 0 .. 0.4 | 0.4 |
| discharge price | 0 .. 300 | 0 |
| discharge quantity | 0 .. 0.4 | 0.4 |

### What the two price dimensions mean in this clearing

`local_da_clearing.py` minimizes

```
sum_t dt * (discharge_price * discharge - charge_price * charge)
```

plus the external energy cost. Reading the two legs off that expression:

- **Discharge price is an offer.** Discharging one MWh reduces the objective
  by `discharge_price` and the PCC draw by one MWh, so it is worth doing when
  `external_price > discharge_price`. A higher quote means less dispatch.
- **Charge price is a credit.** Charging one MWh adds `external_price` and
  subtracts `charge_price`, so it is worth doing when
  `charge_price > external_price`. With the bound at 150 and the external
  price in the hundreds, the charge leg is **inactive except when the external
  price falls below 150**.

Therefore the declared `(0, 0)` is *not* a price-taking declaration. It reads:
"sell at any non-negative price, charge only when the system is otherwise
forced to". `run_local_da` builds exactly this bid today, and the baseline
every comparison is measured from is this same declaration, so the design is
internally consistent — it is just not what "truthful" usually means.

The four charge dimensions are kept, not removed. Their role is a switch that
becomes live in curtailment periods, and the design states that role rather
than pretending to a demand-bid semantic. Fixing the sign is a mechanism
change and is out of scope here.

The deviation penalty's box normalization is unchanged: `p / high` for a
price, `(0.4 - q) / 0.4` for a quantity, i.e. the fraction withheld. The space
grows from 8 to 16 quote dimensions, so `CALIBRATED_DEV_PENALTY` must be
re-derived by `calibrate_dev_penalty.py`; the frozen 0.10 was calibrated on 8.

## B. Observation

`v7_12d`, 12 dimensions.

Unique per agent: `load_feat`, `re_feat`, `soc`, `lmp_bus_norm`,
`avg_lmp_bus_hist_norm`.

Shared: `block_pos`, `slr_norm`, `avg_other_dev`, `all_dev_std`, `ema_dev`,
`price_trend`, `pred_lmp_norm`.

`lmp_bus_norm` is the most recent committed period's LMP **at the agent's own
storage bus**, divided by `PRICE_REF_CNY_MWH`; `avg_lmp_bus_hist_norm` is the
mean of that unit's own bus LMPs over the committed periods so far, same
denominator.

The forecaster and the shared price history **stay on the cross-node mean**
(`_update_price_history`). `ema_dev`, `price_trend` and `pred_lmp_norm`
therefore remain shared and keep their v6 definitions: `ema_dev` is the
deviation of the last observed LMP from the agent forecaster's EMA,
`price_trend` the slope of the last up-to-4 observed LMPs normalized by that
EMA, `pred_lmp_norm` the forecaster's next-block forecast divided by the
reference. All three read past-only state; `audit_pilot_gates` already pins
that. Moving them onto the bus LMP would turn them unique and change the
5+7 split, which is not part of this design.

## C. Day source and windowed clearing

`rl_env` builds each window's `DayAheadInput` from the v3 library pair:

- `active_load_mw`, `reactive_load_mvar`: the pair's load date, sliced to the
  window;
- `external_price_cny_mwh`: the pair's price date, sliced to the window;
- `storage`: the four `StorageUnit`s with the carried SOC as `soc_initial`;
- `generators`: PV21/PV23/WIND30 with the window's available curves and
  `support_day` ratings; no `EXTERNAL33`;
- `bids`: the 4-dim action expanded to constant per-period arrays over the
  window;
- `terminal_soc_equal`: `False` in every block.

The window is `[t_start, min(t_start + 20, 96))`. **The last block is
truncated**: at `t_start = 92` the window is 4 periods with no lookahead at
all, and the committed payoff is that window's whole length. The block count
is 24 throughout.

Each block runs `ClearingModel(day, exact=True).solve()`,
`choose_directions`, and `ClearingModel(day, directions, exact=False)
.solve(prices=True)`.

## D. Reward and settlement

For one unit and one committed period the ledger arithmetic is

```
revenue    = dt * LMP[bus, t] * discharge[t]
purchase   = dt * LMP[bus, t] * charge[t]
degradation = 100 * dt * (charge[t] * eta_charge + discharge[t] / eta_discharge)
```

The block payoff is `sum_t (revenue - purchase - degradation)` over the
committed periods. The day-end block adds one term:

```
inventory = (E_96 - E_0) * external_price[95]
```

where `E_0` is the episode's initial stored energy. This is exactly
`local_da_accounting.settle_day`'s `inventory_value_cny`, evaluated on the
day-end window. Every other block carries no terminal term: intermediate SOC
is carried, not valued, so no stored energy is priced twice.

The reward keeps the B_C2 structure: per-block payoff minus the same block's
payoff under truthful bidding by every storage unit, plus the existing C2 SOC
shaping, scaled by the frozen scale.

The baseline is a **fleet-truthful** counterfactual, not a per-agent one. The
four units are moved together in one re-clear, so the differential reward
attributes to a unit a comparison against a fleet that stopped shading, and a
unit can be paid for the others' restraint. A per-agent counterfactual would
need one clear per unit per block. This is a known, accepted approximation and
must be stated wherever the reward is described.

Rewards use window LMPs; the research KPIs use a single day-long clear. The
same period therefore carries two different prices. The reward is a
rolling-horizon approximation of the settlement, never a reproduction of it,
and the gap is measured (Acceptance 5).

The research KPIs (`pi_ai`, `pi_baseline`, `delta_pi`) are computed at
evaluation from `settle_day` on fully qualified days, never from this reward.

## E. Probes, price qualification and failure

**Probe.** One block per training day, fixed index 11 (periods 44-63, full
lookahead). It runs `clear_day(price_probes=True)` and records
`price_qualification`, `idle_direction_stable` and the audit results.

**Per-block check.** Every training block records whether `choose_directions`
resolved every storage period to a single stable direction (the information it
already computes). A block whose direction flipped on re-solve is
`DIRECTION_UNSTABLE`; its reward is still used, and the block is counted.

**Audit days.** `qualify_prices(backend="cutting")` runs on every validation
and test day and on the three smoke days. It never runs on a training day
during training, and never on test data for model selection.

**Probe failure.** A failed probe does not remove the day from training. It is
recorded as `PROBE_FAILED` with its reason, counted, and the run reports the
failed fraction. Removing the day would be an outcome-based selection and
would need to be named and counted like `feasibility_selected`; this design
does not remove it.

**Failure taxonomy.** Five solve points can fail and each has a name:

| Point | Code | Effect |
| --- | --- | --- |
| exact clear | `CLEAR_INFEASIBLE` | abort episode |
| direction selection | `DIRECTIONS_FAILED` | abort episode |
| conditional price solve | `PRICE_SOLVE_FAILED` | abort episode |
| baseline re-clear | `BASELINE_INFEASIBLE` | abort episode |
| probe | `PROBE_FAILED` | record, continue |

An aborted episode writes nothing to the replay buffer, counts toward the
episode budget as an exposure, and is counted as an abort, not as a completed
episode. If aborts in a round exceed 5% of that round's episodes, the round is
marked `DEGRADED` in the report. No failure is converted into a zero reward.

## F. Cost

Per block: one exact MIQCP, one direction selection, one convex price solve,
and one truthful-baseline re-clear. Per episode: 24 blocks, plus one probed
block per day.

The baseline cannot be cached: it is solved from the same carried SOC and the
same window, and both depend on the policy's own trajectory.

Before the budget is fixed, a three-day measurement records, on the target
machine: one block's exact solve, one block's price solve, one block's
baseline re-clear, and one full-day probe. The budget in
`training_contract.py` is derived from that measurement.

**Pre-declared fallback.** If the measured per-episode time exceeds the
budget, reduce the lookahead from 16 to 8 periods (window 20 to 12), re-measure,
and never go below 8. Block count stays at 24, because the block count sets
the episode's settlement structure. If the budget is still exceeded at
lookahead 8, the protocol does not launch and the measurement is the record.

## Invariants

- **Declared-point identity.** With every gate closed, a unit's action equals
  its spec's truthful vector, its box-normalized deviation is 0, and its block
  payoff equals its baseline block payoff exactly.
- **Exclusivity.** Every cleared block satisfies `charge * discharge == 0`;
  `physical_audit` passes on the exact and the conditional dispatch.
- **No leak.** No feature reads a price from a period after the decision. The
  existing test is extended to `lmp_bus_norm` and `avg_lmp_bus_hist_norm`;
  `ema_dev`, `price_trend` and `pred_lmp_norm` stay covered by
  `audit_pilot_gates`.
- **Bound respect.** Cleared charge and discharge never exceed the declared
  quantity.
- **Failure naming.** Every failed solve carries one of the five codes; no
  episode is scored from a non-optimal solve.
- **No double inventory.** The inventory term appears in the day-end block
  and nowhere else.
- **Split integrity.** Only train days may be dropped for feasibility; every
  validation and test day in `split.json` is evaluated.

## Acceptance

1. A three-day smoke run produces finite rewards, no `None` clears, SOC inside
   `[0.1, 0.9]`, a declared-point deviation of 0, and exactly one inventory
   term per episode.
2. The latency measurement exists, the budget is derived from it, and the
   pre-declared fallback was applied if needed.
3. `pytest tests/` passes with the extended observation test.
4. On the probed blocks, training-path duals and
   `qualify_prices(backend="cutting")` agree in sign on at least 95% of the
   (period, bus) pairs whose price exceeds 1 CNY/MWh, and the median absolute
   relative gap is at most 10%. Below either bound is `FAIL`, not a report.
5. The block-versus-day LMP gap is measured and reported per committed period,
   with its median and its 95th percentile.
6. The re-calibrated deviation weight is reported with the share of the actor
   gradient it claims; a share above 50% is `FAIL`.
7. Every training day carries exactly one probe record; the failed fraction is
   reported.
8. Train days dropped for feasibility are counted and named; validation and
   test counts equal `split.json`.

## Risks

- **Block cost is unmeasured.** The exact MIQCP carries one product per unit
  per period; 20 periods x 4 units is 80. The fallback above is the only
  planned response.
- **The truthful baseline doubles the solves.** Every block is cleared twice.
- **The quantity dimension's declared point is the upper edge.** The
  regularizer pulls every open-gate quantity toward full capacity, so learned
  withholding is a small deviation around 1.0 unless the re-calibrated penalty
  is materially smaller than 0.10.
- **The charge dimensions are inert in most periods.** With the credit at or
  below 150 and the external price in the hundreds, they only act in
  curtailment. Four of sixteen quote dimensions therefore carry little signal,
  and the policy may learn to park them.
- **Fleet-truthful baseline under per-agent actors.** Each unit is paid
  against a fleet that stopped shading, so the reward is not a marginal
  contribution and training is non-stationary by construction.
- **Feasibility selection is an outcome-based filter on train.** It is the
  only way to train, and it is an admission that the training set is not
  outcome-blind. It must be named and counted.
- **One probe per day is a sample.** A day whose other 23 blocks are
  nonsmooth is not detected. The report says the training price quality is
  sampled, never that it is verified.
- **Two price stacks remain.** `local_da_clearing` duals and
  `da_market_pricing` cutting prices are different objects; Acceptance 4 is
  the only evidence they agree.
- **Rolling-horizon reward vs day settlement.** Acceptance 5 measures the
  gap but does not close it.

## Left to the builder

- The adapter's exact placement: `rl_env._last_wholesale` and the window-agent
  construction are replaced, not extended.
- The `da_quote_map` v5 forward function, named so a v5 policy cannot be
  loaded into a v4 loader.
- The failure-record schema and where the trainer counts aborts.
- The `split.json` holiday block: it is not in train, not in the replay
  buffer, and not used for model selection.
