# Phase 7A - Economic decomposition and Cycle Cost sensitivity

`NON_FORMAL_DIAGNOSTIC` | generated 2026-11-12T15:50:25+08:00 | frozen-policy post-hoc accounting

## Scope

This report decomposes the Phase 6 paired-clearing results under the control quote map (Z_CORE = 2.5) and the admitted candidate (Z_CORE = 10.0). Nothing is re-cleared and no policy is executed. Every number is recomputed from the per-participant ledgers the Phase 6 runner wrote, using the settlement definitions in `src/participant_payoff.py`.

The actors were trained under `linear_core_v1`. Phase 6 and this report change only the inference-time quote map, on frozen weights, on a fixed set of days. No result here says how a policy trained under Z_CORE = 10.0 would perform, and no Cycle Cost sensitivity row describes a market that was cleared again under that cost.

## Accounting definition

An independent storage unit (the four ESS) carries no load and no on-site generation, so `IndependentStoragePayoff` settles three terms and nothing else:

```
revenue       = sum_t lmp[node, t] * p_sell[t] * DT_HOURS
purchase_cost = sum_t lmp[node, t] * p_buy[t]  * DT_HOURS
cycle_cost    = rate * (charge_mwh + discharge_mwh)
net           = revenue - purchase_cost - cycle_cost
```

`rate` is `config.storage.cycle_cost` = 100 CNY/MWh on the frozen artifacts, and the term is linear in it. That linearity is confirmed on the data rather than assumed: on every row with nonzero throughput the stored cycle cost equals 100 x (charge + discharge) to within 5.7e-14 CNY/MWh.

## Phase 1 - Full P&L rebuild

| check | status | evidence |
|---|---|---|
| net_identity | PASS | max_abs_residual_cny=6.821210263296962e-13, tolerance_cny=1e-06, scope=every ESS row and every fleet row |
| fleet_equals_sum_ess | PASS | max_abs_residual_cny=4.547473508864641e-13, fields=['revenue', 'purchase', 'cycle', 'net', 'energy'] |
| paired_delta_identity | PASS | max_abs_residual_cny=7.958078640513122e-13, scope=delta_net = delta_revenue - delta_purchase - delta_cycle on every pair and scope |
| phase6d_net_agreement | PASS | max_abs_residual_cny=0.0, cells_compared=6400, source=phase6d/accounting_reconciliation.csv |
| cycle_cost_formula | PASS | max_abs_deviation_from_100_cny_per_mwh=5.684341886080802e-14, detail=stored cycle cost equals 100 x (charge + discharge) on every row with nonzero throughput |

Rows written: `ess_day_pnl.csv` 1280, `fleet_day_pnl.csv` 320, `paired_mapping_delta.csv` 800.

### Component deltas at the frozen rate (CNY/day, mean over dates)

| split | checkpoint | dates | dRevenue | dPurchase | dCycle | dNet | median dNet |
|---|---|---|---|---|---|---|---|
| Probe | seed11_final | 8 | -2.66 | -18.82 | -31.51 | 47.67 | 0.000 |
| Probe | seed23_final | 8 | -23.77 | -28.81 | -41.58 | 46.63 | 0.001 |
| Train | seed11_final | 53 | -16.14 | -10.00 | -35.60 | 29.46 | 0.000 |
| Train | seed23_final | 53 | -19.35 | -10.98 | -36.51 | 28.15 | 0.000 |
| Validation | seed11_final | 11 | -30.06 | -18.82 | -58.92 | 47.67 | 0.000 |
| Validation | seed23_final | 11 | -53.08 | -20.96 | -69.66 | 37.55 | 0.000 |

Train pooled over both seeds: mean dRevenue -17.74, mean dPurchase -10.49, mean dCycle -36.06, mean dNet 28.80.


## Phase 2 - Cycle Cost sensitivity

The cleared trajectories are held fixed - the same schedules, nodal prices, charge and discharge energy, and gate decisions - and only the cycle-cost rate is changed. A row here answers *what the same clearing would have paid under another cost assumption*. It does not answer what the policy would have priced or cleared under that cost, because a different rate changes the dispatch, the nodal prices and the settlement.

### Fleet paired delta by rate (CNY/day)

| split | checkpoint | 0 CNY/MWh | 50 CNY/MWh | 100 CNY/MWh | 150 CNY/MWh | 200 CNY/MWh | dEnergy MWh |
|---|---|---|---|---|---|---|---|
| Probe | seed11_final | 16.16 [-6.44, 47.24] | 31.92 [5.16, 68.94] | 47.67 [9.68, 97.91] | 63.43 [12.04, 129.01] | 79.18 [14.40, 160.11] | -0.3151 |
| Probe | seed23_final | 5.04 [-18.80, 32.47] | 25.84 [5.51, 52.73] | 46.63 [9.37, 93.30] | 67.42 [12.60, 142.10] | 88.21 [15.83, 190.91] | -0.4158 |
| Train | seed11_final | -6.14 [-24.72, 9.86] | 11.66 [-4.39, 27.37] | 29.46 [12.00, 49.06] | 47.26 [24.64, 72.52] | 65.06 [35.92, 98.61] | -0.3560 |
| Train | seed23_final | -8.37 [-24.95, 6.29] | 9.89 [-4.93, 23.53] | 28.15 [11.82, 45.62] | 46.41 [25.19, 69.95] | 64.66 [37.21, 95.08] | -0.3651 |
| Validation | seed11_final | -11.24 [-59.68, 34.89] | 18.22 [-22.25, 64.60] | 47.67 [1.17, 115.82] | 77.13 [13.31, 176.99] | 106.59 [16.09, 239.86] | -0.5892 |
| Validation | seed23_final | -32.11 [-92.20, 12.42] | 2.72 [-39.28, 45.89] | 37.55 [-2.34, 105.33] | 72.38 [10.55, 168.89] | 107.21 [10.82, 235.83] | -0.6966 |

Each cell is mean delta with its date-clustered 95% CI. dEnergy is the mean change in charge-plus-discharge energy per day; the whole rate dependence follows from it, since `delta(r) = delta(100) + (100 - r) * dEnergy`.

### Per-ESS at the frozen rate and at zero cycle cost (CNY/day)

| split | checkpoint | scope | mean dNet at 0 | at 100 | at 200 |
|---|---|---|---|---|---|
| Probe | seed11_final | ESS9 | 2.80 | 2.80 | 2.80 |
| Probe | seed11_final | ESS14 | 5.83 | 19.69 | 33.55 |
| Probe | seed11_final | ESS17 | 3.96 | 13.09 | 22.23 |
| Probe | seed11_final | ESS27 | 3.57 | 12.09 | 20.60 |
| Probe | seed23_final | ESS9 | 0.10 | 13.96 | 27.82 |
| Probe | seed23_final | ESS14 | 0.98 | 0.98 | 0.98 |
| Probe | seed23_final | ESS17 | 4.37 | 18.23 | 32.10 |
| Probe | seed23_final | ESS27 | -0.40 | 13.46 | 27.32 |
| Train | seed11_final | ESS9 | 4.88 | 1.42 | -2.04 |
| Train | seed11_final | ESS14 | -4.60 | 11.64 | 27.89 |
| Train | seed11_final | ESS17 | -5.43 | 9.64 | 24.71 |
| Train | seed11_final | ESS27 | -0.98 | 6.76 | 14.50 |
| Train | seed23_final | ESS9 | -2.19 | 8.13 | 18.46 |
| Train | seed23_final | ESS14 | 2.30 | 1.52 | 0.75 |
| Train | seed23_final | ESS17 | -1.37 | 8.92 | 19.21 |
| Train | seed23_final | ESS27 | -7.11 | 9.57 | 26.24 |
| Validation | seed11_final | ESS9 | 7.04 | 2.01 | -3.03 |
| Validation | seed11_final | ESS14 | -7.67 | 15.08 | 37.84 |
| Validation | seed11_final | ESS17 | -10.36 | 15.27 | 40.89 |
| Validation | seed11_final | ESS27 | -0.26 | 15.32 | 30.89 |
| Validation | seed23_final | ESS9 | -7.57 | 14.65 | 36.86 |
| Validation | seed23_final | ESS14 | 0.59 | 0.06 | -0.46 |
| Validation | seed23_final | ESS17 | -8.63 | 13.85 | 36.33 |
| Validation | seed23_final | ESS27 | -16.50 | 8.99 | 34.48 |

## Phase 3 - Concentration recheck

Reported at the frozen rate for the fleet scope; the drop-Top-N rows are a sensitivity check and do not replace the pre-registered Phase 6 test.

| split | checkpoint | scenario | dates | mean dNet | median dNet |
|---|---|---|---|---|---|
| Probe | seed11_final | full | 8 | 47.67 | 0.000 |
| Probe | seed11_final | drop_top1_pos | 7 | 28.34 | 0.000 |
| Probe | seed11_final | drop_top3_pos | 5 | -0.18 | 0.000 |
| Probe | seed11_final | drop_top5_pos | 3 | -0.30 | -0.000 |
| Probe | seed23_final | full | 8 | 46.63 | 0.001 |
| Probe | seed23_final | drop_top1_pos | 7 | 30.74 | 0.000 |
| Probe | seed23_final | drop_top3_pos | 5 | -0.09 | 0.000 |
| Probe | seed23_final | drop_top5_pos | 3 | -0.15 | 0.000 |
| Train | seed11_final | full | 53 | 29.46 | 0.000 |
| Train | seed11_final | drop_top1_pos | 52 | 24.65 | 0.000 |
| Train | seed11_final | drop_top3_pos | 50 | 17.97 | 0.000 |
| Train | seed11_final | drop_top5_pos | 48 | 11.55 | 0.000 |
| Train | seed23_final | full | 53 | 28.15 | 0.000 |
| Train | seed23_final | drop_top1_pos | 52 | 24.35 | 0.000 |
| Train | seed23_final | drop_top3_pos | 50 | 18.00 | 0.000 |
| Train | seed23_final | drop_top5_pos | 48 | 11.98 | 0.000 |
| Validation | seed11_final | full | 11 | 47.67 | 0.000 |
| Validation | seed11_final | drop_top1_pos | 10 | 15.09 | 0.000 |
| Validation | seed11_final | drop_top3_pos | 8 | -0.55 | 0.000 |
| Validation | seed11_final | drop_top5_pos | 6 | -1.45 | 0.000 |
| Validation | seed23_final | full | 11 | 37.55 | 0.000 |
| Validation | seed23_final | drop_top1_pos | 10 | 5.20 | 0.000 |
| Validation | seed23_final | drop_top3_pos | 8 | -3.31 | 0.000 |
| Validation | seed23_final | drop_top5_pos | 6 | -4.53 | 0.000 |

### Concentration statistics at the frozen rate

| split | checkpoint | dates | S+ | S- | CR5+ | CR5- | max + date | max + | max - date | max - | |d|<0.01 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Probe | seed11_final | 8 | 382.26 | -0.90 | 1.000 | 1.000 | 2022-12-10|2025-12-07 | 183.00 | 2022-10-15|2025-10-12 | -0.90 | 4 |
| Probe | seed23_final | 8 | 373.47 | -0.46 | 1.000 | 1.000 | 2022-12-09|2025-12-09 | 157.84 | 2022-10-15|2025-10-12 | -0.46 | 4 |
| Train | seed11_final | 53 | 1,839.03 | -277.55 | 0.548 | 0.881 | 2022-11-12|2025-11-09 | 279.74 | 2022-11-15|2025-11-13 | -75.94 | 19 |
| Train | seed23_final | 53 | 1,755.48 | -263.64 | 0.522 | 0.935 | 2022-11-12|2025-11-09 | 225.62 | 2022-11-15|2025-11-13 | -73.13 | 20 |
| Validation | seed11_final | 11 | 533.12 | -8.71 | 1.000 | 1.000 | 2022-12-31|2025-12-28 | 373.56 | 2022-11-05|2025-11-02 | -6.23 | 5 |
| Validation | seed23_final | 11 | 440.23 | -27.21 | 1.000 | 1.000 | 2022-12-31|2025-12-28 | 361.02 | 2022-11-19|2025-11-15 | -27.12 | 5 |

## Verification

| check | status | detail |
|---|---|---|
| input_hashes_intact | PASS | 28 inputs, mismatches=0 |
| outputs_unique | PASS | {"ess_day_pnl.csv": {"rows": 1280, "duplicates": 0}, "fleet_day_pnl.csv": {"rows": 320, "duplicates": 0}, "paired_mapping_delta.csv": {"rows": 800, "duplicates": 0}, "cycle_cost_sensitivity.csv": {"rows": 200, "duplicates": 0}, "concentration_sensitivity.csv": {"rows": 800, "duplicates": 0}, "concentration_summary.csv": {"rows": 200, "duplicates": 0}} |
| pairing_complete | PASS | 160 checkpoint/date groups, incomplete=0 |
| pnl_identity | PASS | max |net - (revenue - purchase - cycle)| = 6.821e-13 |
| ess_fleet_identity | PASS | max |fleet - sum(ESS)| = 0.000e+00 over revenue, purchase, cycle and net |
| paired_delta_identity | PASS | max |delta_net - (delta_rev - delta_pur - delta_cycle)| = 7.958e-13 |
| original_net_reproduced | PASS | 6400 cells compared with the Phase 6D reconciliation, max |diff| = 0.000e+00 |
| bootstrap_reproduced | PASS | 40 groups, max |diff| = 1.812e-13 |
| cycle_cost_100_matches_existing | PASS | rate 100 rows equal the Phase 6 bootstrap summary on control mean, candidate mean, delta mean, median, positive share and both CI bounds |
| cycle_cost_linearity | PASS | max residual of delta(r) = delta(100) + (100 - r) x delta_energy: 2.842e-14 |
| no_checkpoint_or_replay_modification | PASS | 92 checkpoint/replay files hashed before and after the analysis, changed=0 |
| no_test_holiday_h2_read | PASS | every pair_id in the artifacts lies in the Train or Validation split of data/processed/v2_cy_jn_2025/common_split.json, and both are disjoint from its test and holiday blocks; the H2 price holdout stays sealed (h2_opened=False) |

## Answers

### 1. Where does the Z=10 gain come from - Revenue, Purchase or Cycle?

| split | checkpoint | dRevenue | -dPurchase | -dCycle | dNet | cycle term / dNet |
|---|---|---|---|---|---|---|
| Probe | seed11_final | -2.66 | 18.82 | 31.51 | 47.67 | 66% |
| Probe | seed23_final | -23.77 | 28.81 | 41.58 | 46.63 | 89% |
| Train | seed11_final | -16.14 | 10.00 | 35.60 | 29.46 | 121% |
| Train | seed23_final | -19.35 | 10.98 | 36.51 | 28.15 | 130% |
| Validation | seed11_final | -30.06 | 18.82 | 58.92 | 47.67 | 124% |
| Validation | seed23_final | -53.08 | 20.96 | 69.66 | 37.55 | 186% |

The three columns are the signed contributions that add up to dNet: `dNet = dRevenue - dPurchase - dCycle`. A negative dPurchase entry means Z=10 buys less, so it contributes positively.

### 2. At zero cycle cost, does Z=10 still have an advantage?

| split | checkpoint | Z2.5 net | Z10 net | mean d | median d | share > 0 | 95% CI |
|---|---|---|---|---|---|---|---|
| Probe | seed11_final | 353.19 | 369.35 | 16.16 | 0.000 | 0.625 | [-6.44, 47.24] |
| Probe | seed23_final | 355.10 | 360.14 | 5.04 | 0.000 | 0.625 | [-18.80, 32.47] |
| Train | seed11_final | 206.57 | 200.44 | -6.14 | 0.000 | 0.547 | [-24.72, 9.86] |
| Train | seed23_final | 251.31 | 242.95 | -8.37 | 0.000 | 0.509 | [-24.95, 6.29] |
| Validation | seed11_final | 183.73 | 172.49 | -11.24 | -0.000 | 0.455 | [-59.68, 34.89] |
| Validation | seed23_final | 273.61 | 241.50 | -32.11 | 0.000 | 0.545 | [-92.20, 12.42] |

On Train and Validation the zero-cost fleet mean delta is Train seed11_final -6.14 [-24.72, 9.86], Train seed23_final -8.37 [-24.95, 6.29], Validation seed11_final -11.24 [-59.68, 34.89], Validation seed23_final -32.11 [-92.20, 12.42] CNY/day. Negative in every group: True. Any CI excluding zero from above: False. So the gain at the frozen rate is a cost-avoidance effect, not an earnings improvement: the candidate keeps more because it cycles less, not because it sells more.

### 3. How does the result move across 0/50/100/150/200 CNY/MWh?

| split | checkpoint | 0 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|
| Probe | seed11_final | 16.16 | 31.92 | 47.67 | 63.43 | 79.18 |
| Probe | seed23_final | 5.04 | 25.84 | 46.63 | 67.42 | 88.21 |
| Train | seed11_final | -6.14 | 11.66 | 29.46 | 47.26 | 65.06 |
| Train | seed23_final | -8.37 | 9.89 | 28.15 | 46.41 | 64.66 |
| Validation | seed11_final | -11.24 | 18.22 | 47.67 | 77.13 | 106.59 |
| Validation | seed23_final | -32.11 | 2.72 | 37.55 | 72.38 | 107.21 |

The rate enters only through `delta(100) + (100 - r) * dEnergy`, so the whole shape is fixed by the mean change in throughput: a negative dEnergy means the candidate moves less charge-plus-discharge energy through the batteries, so it gains more the higher the rate. A positive dEnergy reverses that ordering.

- Probe seed11_final: CI excludes zero from 50 CNY/MWh
- Probe seed23_final: CI excludes zero from 50 CNY/MWh
- Train seed11_final: CI excludes zero from 100 CNY/MWh
- Train seed23_final: CI excludes zero from 100 CNY/MWh
- Validation seed11_final: CI excludes zero from 100 CNY/MWh
- Validation seed23_final: CI excludes zero from 150 CNY/MWh

### 4. Do the two seeds agree on Validation?

| checkpoint | mean d | median d | share > 0 | 95% CI |
|---|---|---|---|---|
| seed11_final | 47.67 | 0.000 | 0.727 | [1.17, 115.82] |
| seed23_final | 37.55 | 0.000 | 0.727 | [-2.34, 105.33] |

Same sign: True. Both CIs exclude zero: False.

### 5. Is the gain still positive after dropping the top three positive dates?

| split | checkpoint | full | drop 1 | drop 3 | drop 5 |
|---|---|---|---|---|---|
| Probe | seed11_final | 47.67 | 28.34 | -0.18 | -0.30 |
| Probe | seed23_final | 46.63 | 30.74 | -0.09 | -0.15 |
| Train | seed11_final | 29.46 | 24.65 | 17.97 | 11.55 |
| Train | seed23_final | 28.15 | 24.35 | 18.00 | 11.98 |
| Validation | seed11_final | 47.67 | 15.09 | -0.55 | -1.45 |
| Validation | seed23_final | 37.55 | 5.20 | -3.31 | -4.53 |

Mean fleet delta after removing the three largest positive dates: Probe seed11_final -0.18, Probe seed23_final -0.09, Train seed11_final 17.97, Train seed23_final 18.00, Validation seed11_final -0.55, Validation seed23_final -3.31 CNY/day. Still positive: Train seed11_final, Train seed23_final; negative: Probe seed11_final, Probe seed23_final, Validation seed11_final, Validation seed23_final.

### 6. Which ESS move against the fleet result?

| split | checkpoint | scope | fleet @0 | ESS @0 | vs fleet @0 | fleet @100 | ESS @100 | vs fleet @100 |
|---|---|---|---|---|---|---|---|---|
| Probe | seed11_final | ESS9 | 16.16 | 2.80 | same | 47.67 | 2.80 | same |
| Probe | seed11_final | ESS14 | 16.16 | 5.83 | same | 47.67 | 19.69 | same |
| Probe | seed11_final | ESS17 | 16.16 | 3.96 | same | 47.67 | 13.09 | same |
| Probe | seed11_final | ESS27 | 16.16 | 3.57 | same | 47.67 | 12.09 | same |
| Probe | seed23_final | ESS9 | 5.04 | 0.10 | same | 46.63 | 13.96 | same |
| Probe | seed23_final | ESS14 | 5.04 | 0.98 | same | 46.63 | 0.98 | same |
| Probe | seed23_final | ESS17 | 5.04 | 4.37 | same | 46.63 | 18.23 | same |
| Probe | seed23_final | ESS27 | 5.04 | -0.40 | opposite | 46.63 | 13.46 | same |
| Train | seed11_final | ESS9 | -6.14 | 4.88 | opposite | 29.46 | 1.42 | same |
| Train | seed11_final | ESS14 | -6.14 | -4.60 | same | 29.46 | 11.64 | same |
| Train | seed11_final | ESS17 | -6.14 | -5.43 | same | 29.46 | 9.64 | same |
| Train | seed11_final | ESS27 | -6.14 | -0.98 | same | 29.46 | 6.76 | same |
| Train | seed23_final | ESS9 | -8.37 | -2.19 | same | 28.15 | 8.13 | same |
| Train | seed23_final | ESS14 | -8.37 | 2.30 | opposite | 28.15 | 1.52 | same |
| Train | seed23_final | ESS17 | -8.37 | -1.37 | same | 28.15 | 8.92 | same |
| Train | seed23_final | ESS27 | -8.37 | -7.11 | same | 28.15 | 9.57 | same |
| Validation | seed11_final | ESS9 | -11.24 | 7.04 | opposite | 47.67 | 2.01 | same |
| Validation | seed11_final | ESS14 | -11.24 | -7.67 | same | 47.67 | 15.08 | same |
| Validation | seed11_final | ESS17 | -11.24 | -10.36 | same | 47.67 | 15.27 | same |
| Validation | seed11_final | ESS27 | -11.24 | -0.26 | same | 47.67 | 15.32 | same |
| Validation | seed23_final | ESS9 | -32.11 | -7.57 | same | 37.55 | 14.65 | same |
| Validation | seed23_final | ESS14 | -32.11 | 0.59 | opposite | 37.55 | 0.06 | same |
| Validation | seed23_final | ESS17 | -32.11 | -8.63 | same | 37.55 | 13.85 | same |
| Validation | seed23_final | ESS27 | -32.11 | -16.50 | same | 37.55 | 8.99 | same |

Units opposite the fleet sign at 0 CNY/MWh: 5 of 24. At 100 CNY/MWh: 0 of 24. Where a unit separates at zero cost it is gaining revenue under Z=10 while the fleet as a whole loses once the cycle cost is removed.

### 7. Any accounting non-closure or missing data?

| check | status | detail |
|---|---|---|
| input_hashes_intact | PASS | 28 inputs, mismatches=0 |
| outputs_unique | PASS | {"ess_day_pnl.csv": {"rows": 1280, "duplicates": 0}, "fleet_day_pnl.csv": {"rows": 320, "duplicates": 0}, "paired_mapping_delta.csv": {"rows": 800, "duplicates": 0}, "cycle_cost_sensitivity.csv": {"rows": 200, "duplicates": 0}, "concentration_sensitivity.csv": {"rows": 800, "duplicates": 0}, "concentration_summary.csv": {"rows": 200, "duplicates": 0}} |
| pairing_complete | PASS | 160 checkpoint/date groups, incomplete=0 |
| pnl_identity | PASS | max |net - (revenue - purchase - cycle)| = 6.821e-13 |
| ess_fleet_identity | PASS | max |fleet - sum(ESS)| = 0.000e+00 over revenue, purchase, cycle and net |
| paired_delta_identity | PASS | max |delta_net - (delta_rev - delta_pur - delta_cycle)| = 7.958e-13 |
| original_net_reproduced | PASS | 6400 cells compared with the Phase 6D reconciliation, max |diff| = 0.000e+00 |
| bootstrap_reproduced | PASS | 40 groups, max |diff| = 1.812e-13 |
| cycle_cost_100_matches_existing | PASS | rate 100 rows equal the Phase 6 bootstrap summary on control mean, candidate mean, delta mean, median, positive share and both CI bounds |
| cycle_cost_linearity | PASS | max residual of delta(r) = delta(100) + (100 - r) x delta_energy: 2.842e-14 |
| no_checkpoint_or_replay_modification | PASS | 92 checkpoint/replay files hashed before and after the analysis, changed=0 |
| no_test_holiday_h2_read | PASS | every pair_id in the artifacts lies in the Train or Validation split of data/processed/v2_cy_jn_2025/common_split.json, and both are disjoint from its test and holiday blocks; the H2 price holdout stays sealed (h2_opened=False) |

12 of 12 checks pass. No non-closure and no missing input: every paired row has both maps and four ESS, and the identities close to 6.821e-13.


## Verdict

`PHASE7A_ANALYSIS = PASS` - every accounting identity, input hash, bootstrap reproduction and base-rate agreement listed above holds.

`PHASE6_ORIGINAL_VERDICT = PARTIAL` (unchanged). Phase 7A explains what that result is made of; it does not upgrade it. The Phase 6 PASS condition was pre-registered on the re-cleared means and medians, not on a decomposition of them.

## What this does not show

- Not a re-cleared market. Every sensitivity row holds the Phase 6 dispatch and prices fixed.
- Not a policy trained under Z_CORE = 10.0.
- Not an independent-generalisation claim: the days are the same Train and Validation pairs Phase 6 used, the price series is the single 2025 source, and Test, Holiday and the H2 holdout were not read.
- Block-level quote-to-clearing transmission remains NOT_TESTED: the Phase 6 runner did not persist the per-block quote, gate and price series, and re-running the clearing is outside this phase.
