# Decision-information timing audit (V2-CY-JN-2025)

## Verdict

**The publication → bid → clearing order cannot be established from the source, so the same-day actual price is *not* treated as a decision-time input for any learned policy.** The pilot's observations do not read it; the clearing uses it as the market's exogenous PCC price, which is a market mechanism rather than a feature. B2A/B2B, which do quote from the same-day curve, are marked as carrying a potential look-ahead and are not a deployable benchmark.

## What the source contains

- `docs/data.csv` columns: `﻿time, price_value_day_ahead_north, price_value_day_ahead_south, mv_value`
- No publication timestamp, no bid-submission timestamp and no clearing timestamp. The file cannot prove the order, and the simulation has no wall-clock gate that would supply it.

## The order that would be required

    price published  ->  ESS bid submitted  ->  distribution clearing

It is **not proven**. Therefore the same-day actual curve is treated as unknown at bid time for every decision-time claim.

## OBS_V6

12 features; forbidden reads found: none. Each feature and when its data is available:

| feature | source | period read | available at the decision |
|---|---|---|---|
| `load_feat` | block mean of the agent's day-ahead load forecast | current block, DA forecast | yes |
| `re_feat` | block mean of the DA PV/wind forecast | current block, DA forecast | yes |
| `soc` | state at the start of the block | none, a state | yes |
| `last_lmp_norm` | most recent committed mean LMP | already cleared periods | no |
| `block_pos` | block index / 23 | none | yes |
| `avg_lmp_hist_norm` | mean of committed mean LMPs | already cleared periods | no |
| `slr_norm` | current block forecast load/RE ratio | current block, DA forecast | yes |
| `avg_other_dev` | other agents' previous committed deviations | previous block | no |
| `all_dev_std` | spread of the same deviations | previous block | no |
| `ema_dev` | observed LMP vs its EMA | already cleared periods | no |
| `price_trend` | slope of the last up-to-4 observed LMPs | already cleared periods | no |
| `pred_lmp_norm` | EMA forecast of the next LMP | a forecast, not a realized future value | yes |

The price features are functions of LMPs already observed (`last_lmp_norm`, `avg_lmp_hist_norm`, `ema_dev`, `price_trend`) or of a per-agent EMA forecast (`pred_lmp_norm`). The forecast is a forecast, not a realized value, and its history is committed periods only. No feature reads the wholesale curve, the counterfactual result, or a future LMP; `tests/test_rl_spec.py` enforces the wholesale-array exclusion at every block.

## Consequences for the pilot

1. The policy's information set is causal by construction; the pilot may proceed.
2. MATD3 must not be described as having known the day's price curve.
3. B2A/B2B are exploratory historical baselines only. Their quotes were computed from the same-day curve, so any advantage they show is potentially an informational one, not a strategy effect.
4. If a later phase needs a price feature in the observation, it must use a published forecast or a lagged curve and be registered as a new observation version.
