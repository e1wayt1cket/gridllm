# Benchmark results (V2-CY-JN-2025)

All figures are cross-year controlled simulations: a 2022 metered load day priced by a 2025 Jiangnan day-ahead curve. The price unit is an inference, not an externally confirmed data dictionary. Nothing here is a same-period settlement.

## Arms and mechanisms

| arm | mechanism | decision information | storage |
|---|---|---|---|
| B0 NoESS | day-ahead SOCP clearing, units removed | the published DA price | none |
| B1 Truthful | day-ahead SOCP clearing, all quotes truthful | the published DA price | 4 units, `(0,0)` |
| B2A threshold | day-ahead SOCP clearing, rule quotes | DA price quantiles of the same day | 4 units |
| B2B threshold+SOC | day-ahead SOCP clearing, rule quotes | DA price quantiles plus the declared SOC band | 4 units |
| B3 LP bound | **not a clearing**: one unit, perfect foresight of the external price | the whole external DA curve | price taker |

B3 is a reference value, not an outcome. It ignores nodal prices and any effect the units have on the clearing. The fleet figure is four times the single-unit bound, so it is an upper bound that no simultaneous plan is required to reach.

## Tuning

- Selected B2A: `q_hi=0.65|q_lo=0.20`
- Selected B2B: `q_hi=0.80|q_lo=0.35|soc_hi=0.90|soc_lo=0.10`
- Search: declared grid on the 53 common training days; the top two per family were re-scored on the 11 common validation days and one selected. The test and holiday days never entered the search.

### Candidate search (common training days)

| family | candidate | train days | mean | median | std | days > 1 CNY | validation mean | selected |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B2A | `q_hi=0.65|q_lo=0.20` | 53 | 261.0 | 177.5 | 295.1 | 34 | 222.9 | yes |
| B2A | `q_hi=0.65|q_lo=0.35` | 53 | 274.0 | 177.4 | 310.2 | 34 | 222.9 |  |
| B2A | `q_hi=0.80|q_lo=0.20` | 53 | 242.7 | 177.3 | 283.6 | 34 | — |  |
| B2A | `q_hi=0.80|q_lo=0.35` | 53 | 255.0 | 177.2 | 298.6 | 34 | — |  |
| B2B | `q_hi=0.65|q_lo=0.20|soc_hi=0.80|soc_lo=0.20` | 53 | 135.7 | 113.0 | 215.6 | 34 | — |  |
| B2B | `q_hi=0.65|q_lo=0.20|soc_hi=0.90|soc_lo=0.10` | 53 | 154.8 | 136.9 | 220.8 | 34 | — |  |
| B2B | `q_hi=0.65|q_lo=0.35|soc_hi=0.80|soc_lo=0.20` | 53 | 135.7 | 113.0 | 215.6 | 34 | — |  |
| B2B | `q_hi=0.65|q_lo=0.35|soc_hi=0.90|soc_lo=0.10` | 53 | 154.8 | 136.9 | 220.8 | 34 | — |  |
| B2B | `q_hi=0.80|q_lo=0.20|soc_hi=0.80|soc_lo=0.20` | 53 | 135.7 | 113.0 | 215.6 | 34 | — |  |
| B2B | `q_hi=0.80|q_lo=0.20|soc_hi=0.90|soc_lo=0.10` | 53 | 154.8 | 136.9 | 220.8 | 34 | 118.7 |  |
| B2B | `q_hi=0.80|q_lo=0.35|soc_hi=0.80|soc_lo=0.20` | 53 | 135.7 | 113.0 | 215.6 | 34 | — |  |
| B2B | `q_hi=0.80|q_lo=0.35|soc_hi=0.90|soc_lo=0.10` | 53 | 154.8 | 136.9 | 220.8 | 34 | 118.7 | yes |

Validation did not discriminate the candidates: the two top candidates' validation means differed by less than 0.01% for B2A and were identical for B2B, so the declared tie-break (candidate id, descending) chose. The selection is a recorded rule, not evidence that one threshold is better.

## Summary by split

### train (53 days)

| arm | days | mean payoff (CNY/day) | median | std | days > 1 CNY | served MWh | curtail MWh | charge MWh | discharge MWh | min V | max line | optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 NoESS | 53 | 0.0 | 0.0 | 0.0 | 0 | 107.8 | 0.00 | 0.00 | 0.00 | 0.9420 | 0.7922 | 53/53 |
| B1 Truthful (= JN-Base) | 53 | 286.3 | 177.4 | 324.7 | 34 | 107.8 | 8.02 | 3.52 | 2.98 | 0.9330 | 0.8150 | 53/53 |
| B2A price threshold | 53 | 261.0 | 177.5 | 295.1 | 34 | 107.8 | 0.00 | 2.91 | 2.46 | 0.9330 | 0.8150 | 53/53 |
| B2B threshold + SOC band | 53 | 154.8 | 136.9 | 220.8 | 34 | 107.8 | 0.00 | 1.29 | 1.09 | 0.9362 | 0.7679 | 53/53 |

### validation (11 days)

| arm | days | mean payoff (CNY/day) | median | std | days > 1 CNY | served MWh | curtail MWh | charge MWh | discharge MWh | min V | max line | optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 NoESS | 11 | 0.0 | 0.0 | 0.0 | 0 | 108.0 | 0.00 | 0.00 | 0.00 | 0.9491 | 0.6991 | 11/11 |
| B1 Truthful (= JN-Base) | 11 | 232.3 | 276.5 | 234.3 | 6 | 108.0 | 8.58 | 3.04 | 2.57 | 0.9454 | 0.6738 | 11/11 |
| B2A price threshold | 11 | 222.9 | 276.6 | 221.1 | 6 | 108.0 | 0.00 | 2.84 | 2.41 | 0.9454 | 0.6738 | 11/11 |
| B2B threshold + SOC band | 11 | 118.7 | 170.4 | 110.7 | 6 | 108.0 | 0.00 | 1.29 | 1.09 | 0.9455 | 0.6738 | 11/11 |

### test (11 days)

| arm | days | mean payoff (CNY/day) | median | std | days > 1 CNY | served MWh | curtail MWh | charge MWh | discharge MWh | min V | max line | optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 NoESS | 11 | 0.0 | 0.0 | 0.0 | 0 | 107.5 | 0.00 | 0.00 | 0.00 | 0.9386 | 0.7731 | 11/11 |
| B1 Truthful (= JN-Base) | 11 | 167.4 | 17.6 | 214.1 | 7 | 107.5 | 7.71 | 3.19 | 2.70 | 0.9348 | 0.7585 | 11/11 |
| B2A price threshold | 11 | 160.3 | 17.6 | 201.6 | 7 | 107.5 | 0.00 | 2.95 | 2.50 | 0.9348 | 0.7585 | 11/11 |
| B2B threshold + SOC band | 11 | 84.3 | 17.5 | 96.0 | 7 | 107.5 | 0.00 | 1.05 | 0.89 | 0.9350 | 0.7585 | 11/11 |

### holiday_block (5 days)

| arm | days | mean payoff (CNY/day) | median | std | days > 1 CNY | served MWh | curtail MWh | charge MWh | discharge MWh | min V | max line | optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 NoESS | 5 | 0.0 | 0.0 | 0.0 | 0 | 93.6 | 0.00 | 0.00 | 0.00 | 0.9719 | 0.5396 | 5/5 |
| B1 Truthful (= JN-Base) | 5 | 70.0 | 0.0 | 140.0 | 1 | 93.6 | 0.79 | 0.09 | 0.08 | 0.9681 | 0.5181 | 5/5 |
| B2A price threshold | 5 | 70.0 | 0.0 | 139.9 | 1 | 93.6 | 0.00 | 0.09 | 0.08 | 0.9681 | 0.5181 | 5/5 |
| B2B threshold + SOC band | 5 | 70.0 | 0.0 | 139.9 | 1 | 93.6 | 0.00 | 0.09 | 0.08 | 0.9681 | 0.5181 | 5/5 |

### full96_only (16 days)

| arm | days | mean payoff (CNY/day) | median | std | days > 1 CNY | served MWh | curtail MWh | charge MWh | discharge MWh | min V | max line | optimal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 NoESS | 16 | 0.0 | 0.0 | 0.0 | 0 | 115.2 | 0.00 | 0.00 | 0.00 | 0.9483 | 0.7126 | 16/16 |
| B1 Truthful (= JN-Base) | 16 | 313.4 | 4.4 | 826.9 | 8 | 115.2 | 5.20 | 1.71 | 1.45 | 0.9399 | 0.7234 | 16/16 |

## Common test set: per-day paired results

11 days, the same days for both datasets. Storage payoff in CNY/day; B0 has no storage, so its fleet payoff is 0 by definition.

| load_date | price_date | group | B0 | B1 | B2A | B2B | B1-B0 | B2A-B1 | B2B-B1 | ESS9 B1 | ESS14 B1 | ESS17 B1 | ESS27 B1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2022-07-24 | 2025-07-12 | summer | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-07-25 | 2025-07-14 | summer | 0.0 | 8.2 | 8.2 | 8.2 | 8.2 | 0.0 | 0.0 | 1.6 | 1.5 | 1.3 | 3.7 |
| 2022-10-11 | 2025-10-10 | autumn | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-12 | 2025-10-09 | autumn | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-27 | 2025-10-27 | autumn | 0.0 | 225.4 | 225.5 | 211.3 | 225.4 | 0.1 | -14.1 | 54.7 | 53.5 | 52.3 | 65.0 |
| 2022-10-30 | 2025-10-25 | autumn | 0.0 | 115.8 | 115.8 | 114.1 | 115.8 | 0.0 | -1.7 | 26.6 | 26.3 | 26.1 | 36.9 |
| 2022-11-06 | 2025-11-01 | autumn | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-12-14 | 2025-12-15 | winter | 0.0 | 550.5 | 501.1 | 274.1 | 550.5 | -49.4 | -276.4 | 131.7 | 129.9 | 128.2 | 160.7 |
| 2022-12-19 | 2025-12-19 | winter | 0.0 | 557.7 | 528.8 | 168.8 | 557.7 | -28.9 | -388.9 | 130.7 | 126.4 | 122.3 | 178.3 |
| 2022-12-24 | 2025-12-21 | winter | 0.0 | 365.8 | 366.6 | 133.9 | 365.8 | 0.9 | -231.9 | 81.8 | 77.7 | 73.8 | 132.5 |
| 2022-12-29 | 2025-12-30 | winter | 0.0 | 17.6 | 17.6 | 17.5 | 17.6 | 0.0 | -0.1 | 3.8 | 3.6 | 3.5 | 6.7 |

### Per-unit paired differences on the common test set

B0 has no storage, so its per-unit column is 0 by definition and the B1-B0 column is the value of adding that unit.

| load_date | unit | B0 | B1 | B2A | B2B | B1-B0 | B2A-B1 | B2B-B1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2022-07-24 | ESS9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-07-24 | ESS14 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-07-24 | ESS17 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-07-24 | ESS27 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-07-25 | ESS9 | 0.0 | 1.6 | 1.7 | 1.6 | 1.6 | 0.0 | 0.0 |
| 2022-07-25 | ESS14 | 0.0 | 1.5 | 1.5 | 1.5 | 1.5 | 0.0 | 0.0 |
| 2022-07-25 | ESS17 | 0.0 | 1.3 | 1.3 | 1.3 | 1.3 | 0.0 | 0.0 |
| 2022-07-25 | ESS27 | 0.0 | 3.7 | 3.8 | 3.7 | 3.7 | 0.0 | 0.0 |
| 2022-10-11 | ESS9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-11 | ESS14 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-11 | ESS17 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-11 | ESS27 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-12 | ESS9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-12 | ESS14 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-12 | ESS17 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-12 | ESS27 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-27 | ESS9 | 0.0 | 54.7 | 54.7 | 51.7 | 54.7 | 0.0 | -3.0 |
| 2022-10-27 | ESS14 | 0.0 | 53.5 | 53.5 | 50.4 | 53.5 | 0.0 | -3.1 |
| 2022-10-27 | ESS17 | 0.0 | 52.3 | 52.4 | 49.1 | 52.3 | 0.0 | -3.2 |
| 2022-10-27 | ESS27 | 0.0 | 65.0 | 65.0 | 60.1 | 65.0 | 0.0 | -4.8 |
| 2022-10-30 | ESS9 | 0.0 | 26.6 | 26.6 | 26.2 | 26.6 | 0.0 | -0.4 |
| 2022-10-30 | ESS14 | 0.0 | 26.3 | 26.3 | 25.8 | 26.3 | 0.0 | -0.5 |
| 2022-10-30 | ESS17 | 0.0 | 26.1 | 26.1 | 25.4 | 26.1 | 0.0 | -0.6 |
| 2022-10-30 | ESS27 | 0.0 | 36.9 | 36.9 | 36.7 | 36.9 | 0.0 | -0.2 |
| 2022-11-06 | ESS9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-11-06 | ESS14 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-11-06 | ESS17 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-11-06 | ESS27 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-12-14 | ESS9 | 0.0 | 131.7 | 120.3 | 65.0 | 131.7 | -11.4 | -66.6 |
| 2022-12-14 | ESS14 | 0.0 | 129.9 | 118.8 | 64.0 | 129.9 | -11.1 | -65.9 |
| 2022-12-14 | ESS17 | 0.0 | 128.2 | 117.4 | 63.1 | 128.2 | -10.8 | -65.1 |
| 2022-12-14 | ESS27 | 0.0 | 160.7 | 144.6 | 82.0 | 160.7 | -16.1 | -78.7 |
| 2022-12-19 | ESS9 | 0.0 | 130.7 | 124.4 | 40.5 | 130.7 | -6.3 | -90.2 |
| 2022-12-19 | ESS14 | 0.0 | 126.4 | 120.6 | 39.4 | 126.4 | -5.8 | -87.0 |
| 2022-12-19 | ESS17 | 0.0 | 122.3 | 117.0 | 38.4 | 122.3 | -5.4 | -84.0 |
| 2022-12-19 | ESS27 | 0.0 | 178.3 | 166.9 | 50.5 | 178.3 | -11.5 | -127.8 |
| 2022-12-24 | ESS9 | 0.0 | 81.8 | 82.1 | 30.8 | 81.8 | 0.2 | -51.1 |
| 2022-12-24 | ESS14 | 0.0 | 77.7 | 77.9 | 29.7 | 77.7 | 0.2 | -48.0 |
| 2022-12-24 | ESS17 | 0.0 | 73.8 | 74.0 | 28.7 | 73.8 | 0.2 | -45.1 |
| 2022-12-24 | ESS27 | 0.0 | 132.5 | 132.6 | 44.7 | 132.5 | 0.2 | -87.7 |
| 2022-12-29 | ESS9 | 0.0 | 3.8 | 3.8 | 3.8 | 3.8 | 0.0 | 0.0 |
| 2022-12-29 | ESS14 | 0.0 | 3.6 | 3.6 | 3.6 | 3.6 | 0.0 | 0.0 |
| 2022-12-29 | ESS17 | 0.0 | 3.5 | 3.5 | 3.4 | 3.5 | 0.0 | 0.0 |
| 2022-12-29 | ESS27 | 0.0 | 6.7 | 6.7 | 6.7 | 6.7 | 0.0 | 0.0 |

### Paired differences on the common test set

| comparison | days | mean | median | std | days positive | days zero | days negative |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 - B0 | 11 | 167.4 | 17.6 | 214.1 | 7 | 4 | 0 |
| B2A - B1 | 11 | -7.0 | 0.0 | 15.8 | 0 | 9 | 2 |
| B2B - B1 | 11 | -83.0 | -0.1 | 136.8 | 0 | 6 | 5 |

Eleven days is too few for an inference claim. The table is the required per-day pairing, not a significance test.

## Holiday block (held out from tuning)

| load_date | price_date | B0 | B1 | B2A | B2B |
|---|---|---:|---:|---:|---:|
| 2022-10-03 | 2025-10-03 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-04 | 2025-10-04 | 0.0 | 349.9 | 349.8 | 349.9 |
| 2022-10-05 | 2025-10-05 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-06 | 2025-10-06 | 0.0 | 0.0 | 0.0 | 0.0 |
| 2022-10-07 | 2025-10-07 | 0.0 | 0.0 | 0.0 | 0.0 |

## B3 price-taker reference

| split | days | mean single-unit bound | mean fleet (4x) bound |
|---|---:|---:|---:|
| train | 53 | 73.6 | 294.5 |
| validation | 11 | 59.2 | 236.7 |
| test | 11 | 44.5 | 177.9 |
| holiday_block | 5 | 18.2 | 72.7 |
| full96_only | 16 | 80.9 | 323.5 |

The bound is computed against the **external** price series. It is not comparable with B1/B2 payoffs as if they were the same quantity: the bound knows the whole curve, faces no market impact and settles at the external price, while B1/B2 settle at nodal LMPs they helped set.

## What these tables do not support

- No MATD3 policy was trained; nothing here is an RL result.
- B1 is not an independent arm from JN-Base; it is the same clearing.
- The 21-day C80/C90/C100/NoESS probe is not the 96-day sample.
- FULL96-only long pairs are training-only and are not in the common test set.
- A rule earning more than truthful is a **participant payoff** result, not a welfare result. The clearing objective values storage at its own quotes and is explicitly not social welfare; the two are different quantities in this project.
