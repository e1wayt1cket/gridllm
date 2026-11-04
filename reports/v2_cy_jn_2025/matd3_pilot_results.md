# MATD3 pilot results (V2-CY-JN-2025)

A controlled pilot on the common split: **two seeds, twenty episodes,** and two training pools that differ only in the 16 FULL96-only days. Eleven test days cannot support a significance claim; every table below is descriptive, and the settlement is a full-day clear with the policy's own per-block quotes.

## Training stability

| run | episodes | steps | updates | last-5 reward | last-5 raw | critic loss | actor loss | clears | fallbacks | LMP fallbacks | voltage cells | line cells | max end-SOC gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NEAR seed 11 | 20 | 480 | 417 | 63.8 | 286.5 | 1.3 | -0.1 | 480 | 0 | 0 | 0 | 0 | 4.00e-01 |
| NEAR seed 22 | 20 | 480 | 417 | 14.5 | 322.8 | 1.5 | 0.3 | 480 | 0 | 0 | 0 | 0 | 4.00e-01 |
| FULL96 seed 11 | 20 | 480 | 417 | -124.9 | 206.3 | 2.1 | 0.6 | 480 | 0 | 0 | 0 | 0 | 4.00e-01 |
| FULL96 seed 22 | 20 | 480 | 417 | -50.4 | 13.9 | 2.1 | 0.3 | 480 | 0 | 0 | 0 | 0 | 4.00e-01 |

The end-SOC gap is the rolling episode's final state against 0.5. A training episode does **not** pin terminal SOC; only the full-day settlement does, and every evaluation row reports a terminal error of 0. A 0.4 gap (the full 0.10-0.90 band) is therefore expected in training and is not a constraint violation.

## Training-day exposure

| run | pool days | distinct days seen | most-seen day count |
|---|---:|---:|---:|
| NEAR seed 11 | 53 | 20 | 1 |
| NEAR seed 22 | 53 | 20 | 1 |
| FULL96 seed 11 | 69 | 20 | 1 |
| FULL96 seed 22 | 69 | 20 | 1 |

Twenty episodes over 53 or 69 days means most days are seen at most once; the pool difference is a composition difference, not extra repetition.

## Learning trend

| run | first-5 reward | last-5 reward | first-5 raw | last-5 raw | first-5 action charge | last-5 action charge |
|---|---:|---:|---:|---:|---:|---:|
| NEAR seed 11 | -209.0 | 63.8 | 9.5 | 286.5 | 63.9 | 59.5 |
| NEAR seed 22 | -151.0 | 14.5 | 100.8 | 322.8 | 75.5 | 126.6 |
| FULL96 seed 11 | -89.1 | -124.9 | 94.5 | 206.3 | 63.4 | 47.8 |
| FULL96 seed 22 | 11.1 | -50.4 | 151.1 | 13.9 | 74.4 | 90.6 |

The reward is the differential learning signal. The raw profit is what the agent settled. A policy that learns to quote closer to truthful can raise its raw profit while its differential reward stays near zero, because both arms move together; the columns are kept apart for that reason.

## Evaluation on the common test set (11 days)

| run | settled (CNY/day) | B1 truthful | B2A threshold | MATD3 − B1 | MATD3 − B2A | days above B1 | days below B2A |
|---|---:|---:|---:|---:|---:|---:|---:|
| NEAR seed 11 | 106.7 | 167.4 | 160.3 | -60.6 | -53.6 | 0 | 7 |
| NEAR seed 22 | 42.9 | 167.4 | 160.3 | -124.5 | -117.5 | 0 | 7 |
| FULL96 seed 11 | 79.4 | 167.4 | 160.3 | -87.9 | -80.9 | 0 | 7 |
| FULL96 seed 22 | 28.4 | 167.4 | 160.3 | -139.0 | -132.0 | 0 | 7 |

All four runs settle **below B1 on every test day** (0 of 11 days above). The mechanism is visible in the learning trend: the policies learn quotes well above the truthful point (charge ~48-127 CNY/MWh, discharge ~20-82), and a quote above truthful raises the clearing's cost of charging and penalises discharging, so the fleet cycles less than the truthful arm. The positive training reward is a *local* differential against other agents' contemporaneous actions; it is not the day-level B1 comparison, and the two can disagree. This is the reward-is-not-benefit distinction the audits require.

### Paired per-day fleet payoff on the common test set

Mean over the two seeds. `FULL96 - NEAR` is the treatment difference the pilot is about.

| load_date | B1 | B2A | NEAR | FULL96 | FULL96 - NEAR |
|---|---:|---:|---:|---:|---:|
| 2022-07-24 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-07-25 | 8.2 | 8.2 | 3.0 | 1.1 | -1.9 |
| 2022-10-11 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-12 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-27 | 225.4 | 225.5 | 137.0 | 90.7 | -46.3 |
| 2022-10-30 | 115.8 | 115.8 | 66.6 | 55.3 | -11.2 |
| 2022-11-06 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-12-14 | 550.5 | 501.1 | 281.0 | 256.3 | -24.7 |
| 2022-12-19 | 557.7 | 528.8 | 200.9 | 113.1 | -87.8 |
| 2022-12-24 | 365.8 | 366.6 | 128.8 | 74.1 | -54.7 |
| 2022-12-29 | 17.6 | 17.6 | 5.5 | 2.2 | -3.4 |

FULL96 − NEAR over the 11 days: mean -20.9 CNY/day, median -3.4, positive on 0 days, negative on 7. This is a descriptive paired difference, not a significance test.

### Per-unit settled payoff (mean over seeds and the 11 days)

| run | ESS9 | ESS14 | ESS17 | ESS27 | fleet |
|---|---:|---:|---:|---:|---:|
| NEAR | 8.5 | 7.3 | 32.5 | 26.5 | 74.8 |
| FULL96 | 12.6 | 0.0 | 21.4 | 19.8 | 53.9 |

### Per-unit settled payoff by day (mean over seeds)

| load_date | NEAR E9 | NEAR E14 | NEAR E17 | NEAR E27 | FULL96 E9 | FULL96 E14 | FULL96 E17 | FULL96 E27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-07-24 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-07-25 | 0.0 | 0.0 | 1.1 | 1.9 | 0.0 | 0.0 | 1.1 | 0.0 |
| 2022-10-11 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-12 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-27 | 24.8 | 25.5 | 54.3 | 32.5 | 32.6 | 0.0 | 29.0 | 29.0 |
| 2022-10-30 | 11.5 | 11.4 | 25.2 | 18.4 | 11.8 | 0.0 | 13.8 | 29.7 |
| 2022-11-06 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-12-14 | 40.5 | 40.3 | 119.7 | 80.5 | 83.0 | 0.0 | 69.1 | 104.3 |
| 2022-12-19 | 12.0 | 2.8 | 96.7 | 89.4 | 7.0 | 0.0 | 72.8 | 33.4 |
| 2022-12-24 | 4.5 | 0.0 | 58.5 | 65.7 | 4.8 | 0.0 | 47.4 | 21.9 |
| 2022-12-29 | 0.0 | 0.0 | 2.2 | 3.4 | 0.0 | 0.0 | 2.2 | 0.0 |

`eval.csv` carries the same grid per run and per unit, together with the B1 per-unit payoff, charge/discharge energy and cycles, so the per-run values are traceable rather than only the seed average.

## Holiday block (held out, 5 days)

| run | settled | B1 | B2A | MATD3 − B1 |
|---|---:|---:|---:|---:|
| NEAR seed 11 | 70.0 | 70.0 | 70.0 | 0.0 |
| NEAR seed 22 | 70.0 | 70.0 | 70.0 | 0.0 |
| FULL96 seed 11 | 69.9 | 70.0 | 70.0 | 0.0 |
| FULL96 seed 22 | 70.0 | 70.0 | 70.0 | 0.0 |

## Physical and reward accounting on the evaluation days

| run | optimal | terminal SOC max err | voltage cells | line cells | LMP fallbacks | mean charge MWh | mean discharge MWh | mean cycles | mean dev penalty | block reward mean | block raw mean | block baseline mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NEAR seed 11 | 16/16 | 0.00e+00 | 0 | 0 | 0 | 0.9504063035339643 | 0.804423895566123 | 0.5483844372187773 | 0.018066604162259664 | 116.36184078592245 | 104.43838808412552 | -11.923452701796958 |
| NEAR seed 22 | 16/16 | 0.00e+00 | 0 | 0 | 0 | 0.5718272211562574 | 0.4839945647167989 | 0.32994430808533004 | 0.061306435791506346 | 88.2608172461079 | 103.90739393537602 | 15.646576689268098 |
| FULL96 seed 11 | 16/16 | 0.00e+00 | 0 | 0 | 0 | 0.8106051605283198 | 0.6860962089861535 | 0.4677191779732729 | 0.032963354869224554 | -6.933511206688748 | 56.21603012578344 | 63.14954133247217 |
| FULL96 seed 22 | 16/16 | 0.00e+00 | 0 | 0 | 0 | 0.19048331063867888 | 0.16122506908163134 | 0.10990886866259694 | 0.06349700754791249 | 71.47815715616997 | 90.2189665785204 | 18.74080942235041 |

The block reward is the rolling-window learning signal and the settled columns above are the day-level payoff; they are different quantities and neither replaces the other.

## Answers

1. **Price availability at bid time.** Not proven; no policy observation reads the price curve. The pilot treats the same-day price as the market's exogenous input, not as a decision feature.
2. **OBS_V6 leakage.** None found; the timing audit lists each feature's source and period.
3. **State-matched baseline.** Yes, 9/9 checks on an instrumented block: same window, current SOC, same wholesale and config, truthful committed block, opponents unchanged.
4. **Double charging.** None: cycle cost is charged once in the settlement ledger and separately (not additively) in the clearing objective; the deviation penalty is in the actor loss only.
5. **Same configuration and budget.** Yes: identical OBS/action specs, network, hyperparameters, 20 episodes and seeds 11/22; the only difference is the training-day pool.
6. **Do the extra 16 days help?** Paired FULL96 − NEAR over the 11 test days: mean -20.9 CNY/day, 0 positive, 7 negative, 4 tied at zero. On this budget they did not help. Descriptive only.
7. **More stable than B1/B2A, or a few lucky days?** NEAR vs B1: mean -92.6, above on 0 of 11 days and below on 7; FULL96 vs B1: mean -113.5, above on 0 of 11 and below on 7. The shortfall appears on every test day with a non-zero spread; there is no single-day gain carrying a positive mean.
8. **Boundaries.** Frozen assets and the clearing engine were not modified; no commit or push.

## What this pilot does not show

- It does not show MATD3 has an economic advantage. Two seeds and eleven days are a runnability check.
- It does not make B2A a fair benchmark: B2A quotes from the same-day curve, which is not proven available at bid time.
- It does not validate the FULL96 pool for training; it only measures the difference on this split with this budget.
