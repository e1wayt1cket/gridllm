# MATD3 interface audit (V2-CY-JN-2025)

No training was started. This records what the environment actually consumes, what the price injection touches, and the result of a deterministic smoke run. It is a cross-year controlled simulation, not a same-period market result.

## Observation: OBS_V6, unchanged

| # | feature | computed from | periods read at decision block b |
|---:|---|---|---|
| 1 | `load_feat` | block mean of the agent's load forecast / its own forecast peak | the current block, `4b..4b+3` |
| 2 | `re_feat` | block mean of PV+wind forecast / forecast peak | the current block |
| 3 | `soc` | state at the start of the block | none (a state) |
| 4 | `last_lmp_norm` | most recent committed cross-node mean LMP / `PRICE_REF_CNY_MWH` | only periods already cleared |
| 5 | `block_pos` | `b / 23` | none |
| 6 | `avg_lmp_hist_norm` | mean of committed LMPs / reference | only periods already cleared |
| 7 | `slr_norm` | system load/RE ratio of the current block's forecast | the current block |
| 8 | `avg_other_dev` | signed deviation of the other agents' committed quotes | the previous block only |
| 9 | `all_dev_std` | spread of the same signed deviations | the previous block only |
| 10 | `ema_dev` | observed LMP vs its EMA | only periods already cleared |
| 11 | `price_trend` | slope of the last up-to-4 observed LMPs | only periods already cleared |
| 12 | `pred_lmp_norm` | EMA forecast of the next LMP / reference | a forecast of the future, not a realized value |

No feature reads the wholesale curve, this episode's or any other. The Jiangnan price reaches the policy only through LMPs it has already observed, so the causality the V4 replacement established still holds.

## Decision blocks

24 blocks of 4 quarter-hour periods (1 h). Each block clears a rolling window (`roll_horizon = 16` periods) and commits its first 4. The wholesale curve is sliced per window with `t_start`, and the smoke run checks every block's slice equals the pair's own curve.

## Action space

| participant | action | box | truthful |
|---|---|---|---|
| ESS9/14/17/27 | `charge_price`, `discharge_price` (CNY/MWh) | 0–150, 0–300 | (0, 0) |
| loads and prosumers | `bid_mult`, `offer_adder` | 0.3–1.8, 0–50 | (1, 0) |

The smoke run controls the four storage units only; every other participant declares its truthful quote.

## Reward and settlement

Per block, reward is the storage unit's own `participant_payoff` for the committed periods minus the same window re-cleared under truthful bidding. Cycle cost is the same `config.storage.cycle_cost` the clearing applies, so C80/C90/C100 move reward and clearing together. The environment's rolling windows do not pin terminal SOC; the whole-day settlement does, and `results.csv` shows `terminal_soc_max_err = 0` for every paired day.

## Smoke result

| pair | blocks cleared | fallbacks | LMP fallback cells | obs NaN | reward NaN | out-of-box actions | SOC out of range | max price-slice deviation (CNY/MWh) |
|---|---:|---:|---:|---|---|---|---|---:|
| 2022-07-22 → 2025-07-15 | 24 | 0 | 0 | False | False | 0 | 0 | 0.00e+00 |
| 2022-10-11 → 2025-10-10 | 24 | 0 | 0 | False | False | 0 | 0 | 0.00e+00 |
| 2022-12-09 → 2025-12-09 | 24 | 0 | 0 | False | False | 0 | 0 | 0.00e+00 |

Repeating the same three pairs gave identical rewards: **True**.

## `mv_value` and JN-RE

`mv_value` is not in any observation. Its unit, its 15-minute status and the time it becomes available cannot be confirmed from the source, and a provincial new-energy total is not a bus-level injection. Enabling JN-RE would require a new observation version (`OBS_V7`), a fresh reference-price and normalisation decision, and would invalidate every existing checkpoint. `load_policy` enforces the name, so a mismatched actor is refused rather than silently reused.

## What is not established

- No policy was trained. The smoke policy is truthful and fixed.
- The reference price `PRICE_REF_CNY_MWH` is the frozen synthetic-family constant. The Jiangnan daily mean is close to it, and replacing it would be a training-scale decision this version does not make.
- Nothing here is a performance claim about MATD3.
