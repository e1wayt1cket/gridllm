# Phase 6 — frozen-policy paired market clearing

label: `NON_FORMAL_DIAGNOSTIC`; no formal map switch; no training.

**Verdict: `PARTIAL`** — The pre-registered PASS condition is not met. seed23_final on Validation: mean 37.55 CNY/day, 95% CI [-2.34, 105.33] includes zero. Every fleet median is within 0.001 CNY/day of zero, so the positive mean is carried by a minority of dates while the typical date is unchanged.

This is a frozen-policy counterfactual. The four actors were trained under `linear_core_v1`; the experiment changes only the inference quote map. It does not measure a policy trained under Z=10.0.

## What was run

| phase | split | dates | checkpoints | maps | day-runs |
|---|---|---:|---|---:|---:|
| 6A | Train probe | 8 | 4 | 2 | 64 |
| 6B | Train full | 53 | 2 | 2 | 212 |
| 6C | Validation | 11 | 2 | 2 | 44 |

Neither Test, Holiday, nor the H2 holdout was read.

## Physical, solver and accounting validation — all phases

| metric | 6A | 6B | 6C |
|---|---|---|---|
| day-runs | 64 | 212 | 44 |
| day_ok | 64/64 | 212/212 | 44/44 |
| clearing periods | 1536 | 5088 | 1056 |
| optimal clears | 1536 | 5088 | 1056 |
| fallback clears | 0 | 0 | 0 |
| LMP fallback periods | 0 | 0 | 0 |
| infeasible clears | 0 | 0 | 0 |
| voltage min (pu) | 0.940280 | 0.933177 | 0.945506 |
| voltage max (pu) | 1.023054 | 1.023054 | 1.020126 |
| voltage violation cells | 0 | 0 | 0 |
| max line utilization | 0.621871 | 0.814891 | 0.665103 |
| line overload cells | 0 | 0 | 0 |
| max terminal SOC error | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| max SOC band violation | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| max power-limit violation | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| max simultaneous charge/discharge | 1.432e-08 | 1.821e-08 | 7.409e-09 |
| max raw_profit vs ledger diff (CNY) | 0.000e+00 | 0.000e+00 | 0.000e+00 |

Voltage band: `0.93 <= V <= 1.05` pu (the SOCP clearing's own constraint). A `day_ok` row requires 24/24 optimal clears, no fallback, no LMP fallback, zero voltage and line violations, terminal SOC within 1e-5 of 0.5, raw SOC inside [0.1, 0.9], and the equipment power limit respected.

## Preflight

| check | value |
|---|---|
| control == linear_core_v1 (max abs diff) | 0.000e+00 |
| candidate self-tests | {'2.5': True, '10.0': True} |
| repeat same day (max payoff diff, CNY) | 0.000e+00 |
| distinct quotes per ESS | {'ESS9': {'charge': 18, 'discharge': 23}, 'ESS14': {'charge': 24, 'discharge': 24}, 'ESS17': {'charge': 24, 'discharge': 24}, 'ESS27': {'charge': 24, 'discharge': 24}} |
| actor hash unchanged | True |
| all_pass | True |

## Phase 6A — probe pairs

| checkpoint | round | role | control mean CNY | candidate mean CNY | mean delta CNY | median delta CNY | delta % | date-clustered 95% CI | share dates > 0 | dates |
|---|---|---|---|---|---|---|---|---|---|---|
| seed11_best | 1 | auxiliary_regression | 236.27 | 237.43 | 1.16 | 0.0001 | +0.49% | [-13.48, 17.78] | 0.625 | 8 |
| seed11_final | 10 | primary | 204.32 | 251.99 | 47.67 | 0.0002 | +23.33% | [9.68, 97.91] | 0.750 | 8 |
| seed23_best | 1 | auxiliary_regression | 242.78 | 249.82 | 7.04 | 0.0003 | +2.90% | [0.06, 14.87] | 0.875 | 8 |
| seed23_final | 10 | primary | 178.82 | 225.45 | 46.63 | 0.0006 | +26.08% | [9.37, 93.30] | 0.875 | 8 |

## Phase 6B — Train

| checkpoint | round | role | control mean CNY | candidate mean CNY | mean delta CNY | median delta CNY | delta % | date-clustered 95% CI | share dates > 0 | dates |
|---|---|---|---|---|---|---|---|---|---|---|
| seed11_final | 10 | primary | 14.31 | 43.77 | 29.46 | 0.0002 | +205.92% | [12.00, 49.06] | 0.755 | 53 |
| seed23_final | 10 | primary | 0.52 | 28.67 | 28.15 | 0.0002 | +5369.21% | [11.82, 45.62] | 0.717 | 53 |

## Phase 6C — Validation

| checkpoint | round | role | control mean CNY | candidate mean CNY | mean delta CNY | median delta CNY | delta % | date-clustered 95% CI | share dates > 0 | dates |
|---|---|---|---|---|---|---|---|---|---|---|
| seed11_final | 10 | primary | -5.93 | 41.74 | 47.67 | 0.0002 | +803.71% | [1.17, 115.82] | 0.727 | 11 |
| seed23_final | 10 | primary | -4.67 | 32.88 | 37.55 | 0.0001 | +803.82% | [-2.34, 105.33] | 0.727 | 11 |

## Per-ESS deltas (fleet decision unit)

A positive per-ESS delta is not a fleet improvement. The fleet row is the decision unit; the per-ESS rows are attribution only.

### Phase A

| checkpoint | scope | control mean CNY | candidate mean CNY | mean delta CNY | median delta CNY | 95% CI | share dates > 0 |
|---|---|---|---|---|---|---|---|
| seed11_best | fleet | 236.27 | 237.43 | 1.16 | 0.0001 | [-13.48, 17.78] | 0.625 |
| seed11_best | ESS9 | 43.12 | 43.24 | 0.12 | -0.0000 | [-0.41, 0.81] | 0.375 |
| seed11_best | ESS14 | 61.98 | 61.76 | -0.23 | 0.0000 | [-6.40, 5.61] | 0.750 |
| seed11_best | ESS17 | 65.62 | 60.13 | -5.49 | -0.0008 | [-13.81, -0.24] | 0.500 |
| seed11_best | ESS27 | 65.54 | 72.30 | 6.75 | 0.0001 | [-2.10, 19.63] | 0.750 |
| seed11_final | fleet | 204.32 | 251.99 | 47.67 | 0.0002 | [9.68, 97.91] | 0.750 |
| seed11_final | ESS9 | 51.91 | 54.70 | 2.80 | -0.0000 | [-0.05, 8.25] | 0.500 |
| seed11_final | ESS14 | 41.95 | 61.64 | 19.69 | 0.0001 | [3.96, 39.49] | 0.875 |
| seed11_final | ESS17 | 47.77 | 60.87 | 13.09 | 0.0001 | [-0.06, 32.00] | 0.750 |
| seed11_final | ESS27 | 62.70 | 74.78 | 12.09 | 0.0000 | [-0.65, 25.39] | 0.750 |
| seed23_best | fleet | 242.78 | 249.82 | 7.04 | 0.0003 | [0.06, 14.87] | 0.875 |
| seed23_best | ESS9 | 61.98 | 62.29 | 0.30 | 0.0000 | [-0.50, 1.46] | 0.625 |
| seed23_best | ESS14 | 69.56 | 61.64 | -7.91 | -0.1321 | [-17.69, -0.75] | 0.500 |
| seed23_best | ESS17 | 60.70 | 60.35 | -0.35 | 0.0001 | [-9.25, 9.95] | 0.625 |
| seed23_best | ESS27 | 50.55 | 65.54 | 14.99 | 0.0000 | [0.02, 35.23] | 0.625 |
| seed23_final | fleet | 178.82 | 225.45 | 46.63 | 0.0006 | [9.37, 93.30] | 0.875 |
| seed23_final | ESS9 | 48.48 | 62.44 | 13.96 | 0.0002 | [2.13, 28.21] | 0.875 |
| seed23_final | ESS14 | 40.96 | 41.94 | 0.98 | 0.0000 | [-0.04, 2.77] | 0.875 |
| seed23_final | ESS17 | 38.84 | 57.07 | 18.23 | 0.0002 | [3.83, 37.12] | 0.875 |
| seed23_final | ESS27 | 50.53 | 63.99 | 13.46 | 0.0001 | [-2.19, 33.37] | 0.750 |

### Phase B

| checkpoint | scope | control mean CNY | candidate mean CNY | mean delta CNY | median delta CNY | 95% CI | share dates > 0 |
|---|---|---|---|---|---|---|---|
| seed11_final | fleet | 14.31 | 43.77 | 29.46 | 0.0002 | [12.00, 49.06] | 0.755 |
| seed11_final | ESS9 | 5.49 | 6.91 | 1.42 | -0.0000 | [-0.89, 4.03] | 0.491 |
| seed11_final | ESS14 | -1.75 | 9.90 | 11.64 | 0.0001 | [5.15, 18.51] | 0.736 |
| seed11_final | ESS17 | -0.16 | 9.48 | 9.64 | 0.0001 | [3.08, 17.00] | 0.698 |
| seed11_final | ESS27 | 10.73 | 17.48 | 6.76 | 0.0000 | [1.87, 12.59] | 0.755 |
| seed23_final | fleet | 0.52 | 28.67 | 28.15 | 0.0002 | [11.82, 45.62] | 0.717 |
| seed23_final | ESS9 | 2.56 | 10.70 | 8.13 | 0.0001 | [3.08, 13.90] | 0.698 |
| seed23_final | ESS14 | -2.36 | -0.84 | 1.52 | 0.0001 | [0.14, 3.22] | 0.679 |
| seed23_final | ESS17 | -2.74 | 6.18 | 8.92 | 0.0001 | [3.55, 14.97] | 0.755 |
| seed23_final | ESS27 | 3.07 | 12.64 | 9.57 | 0.0001 | [2.88, 16.30] | 0.736 |

### Phase C

| checkpoint | scope | control mean CNY | candidate mean CNY | mean delta CNY | median delta CNY | 95% CI | share dates > 0 |
|---|---|---|---|---|---|---|---|
| seed11_final | fleet | -5.93 | 41.74 | 47.67 | 0.0002 | [1.17, 115.82] | 0.727 |
| seed11_final | ESS9 | 0.73 | 2.74 | 2.01 | 0.0000 | [0.00, 6.02] | 0.636 |
| seed11_final | ESS14 | -1.57 | 13.51 | 15.08 | 0.0002 | [0.08, 38.80] | 0.818 |
| seed11_final | ESS17 | -4.31 | 10.95 | 15.27 | 0.0002 | [0.45, 38.19] | 0.818 |
| seed11_final | ESS27 | -0.77 | 14.54 | 15.32 | 0.0001 | [1.36, 37.34] | 0.818 |
| seed23_final | fleet | -4.67 | 32.88 | 37.55 | 0.0001 | [-2.34, 105.33] | 0.727 |
| seed23_final | ESS9 | -1.70 | 12.94 | 14.65 | 0.0000 | [0.23, 37.37] | 0.727 |
| seed23_final | ESS14 | -2.28 | -2.22 | 0.06 | -0.0000 | [-1.09, 1.04] | 0.455 |
| seed23_final | ESS17 | -4.70 | 9.15 | 13.85 | 0.0000 | [0.08, 37.09] | 0.727 |
| seed23_final | ESS27 | 4.02 | 13.00 | 8.99 | 0.0000 | [-4.47, 32.32] | 0.818 |

## Date-level distribution of the fleet delta

The mean is not the typical date. A large block of dates has a small gain, and the aggregate gain is carried by a minority of dates; the tables below report that concentration instead of hiding it behind the mean. The median falls inside the near-zero block, which is why it is positive but far below the mean.

| phase | checkpoint | dates | delta > 0 | delta < 0 | |delta| < 0.01 CNY | sum positive CNY | sum negative CNY | max CNY | min CNY | top 10% of dates (share of positive total) |
|---|---|---|---|---|---|---|---|---|---|---|
| A | seed11_best | 8 | 5 | 3 | 4 | 48.49 | -39.20 | 48.49 | -35.95 | 48 (100%) |
| A | seed11_final | 8 | 6 | 2 | 4 | 382.26 | -0.90 | 183.00 | -0.90 | 183 (48%) |
| A | seed23_best | 8 | 7 | 1 | 4 | 57.27 | -0.97 | 29.23 | -0.97 | 29 (51%) |
| A | seed23_final | 8 | 7 | 1 | 4 | 373.47 | -0.46 | 157.84 | -0.46 | 158 (42%) |
| B | seed11_final | 53 | 40 | 13 | 19 | 1839.03 | -277.55 | 279.74 | -75.94 | 1176 (64%) |
| B | seed23_final | 53 | 38 | 15 | 20 | 1755.48 | -263.64 | 225.62 | -73.13 | 1075 (61%) |
| C | seed11_final | 11 | 8 | 3 | 5 | 533.12 | -8.71 | 373.56 | -6.23 | 455 (85%) |
| C | seed23_final | 11 | 8 | 3 | 5 | 440.23 | -27.21 | 361.02 | -27.12 | 402 (91%) |

## Phase 6A vs Phase 6B reproducibility

| metric | value |
|---|---|
| common checkpoint/map/date rows | 32 |
| max fleet income difference (CNY) | 0.000e+00 |
| identical | True |

Phase 6A probe pairs are a subset of the Train dates, so the same frozen checkpoint and map on the same date must reproduce the same clearing and settlement.

## Quote dynamics

One ESS-day is one ESS on one cleared date. The action is regenerated at every block from the current observation, so a single distinct quote means the policy chose the same quote all day, not that the day was swept once. 176 ESS-days have a single distinct charge quote; in 176 of them the charge gate is closed for all 24 blocks, which is why the quote never needs to move.

| phase | checkpoint | Z | ESS-days | min charge quotes | mean charge quotes | charge > 1 share | min discharge quotes | mean discharge quotes | discharge > 1 share | mean charge gate-open fraction |
|---|---|---|---|---|---|---|---|---|---|---|
| A | seed11_best | 2.5 | 32 | 1 | 18.2 | 0.750 | 6 | 20.0 | 1.000 | 0.750 |
| A | seed11_best | 10.0 | 32 | 1 | 18.2 | 0.750 | 24 | 24.0 | 1.000 | 0.750 |
| A | seed11_final | 2.5 | 32 | 10 | 21.7 | 1.000 | 15 | 21.7 | 1.000 | 0.944 |
| A | seed11_final | 10.0 | 32 | 16 | 22.9 | 1.000 | 24 | 24.0 | 1.000 | 0.944 |
| A | seed23_best | 2.5 | 32 | 1 | 18.2 | 0.750 | 19 | 23.4 | 1.000 | 0.750 |
| A | seed23_best | 10.0 | 32 | 1 | 18.2 | 0.750 | 24 | 24.0 | 1.000 | 0.750 |
| A | seed23_final | 2.5 | 32 | 1 | 18.2 | 0.750 | 4 | 13.8 | 1.000 | 0.750 |
| A | seed23_final | 10.0 | 32 | 1 | 18.2 | 0.750 | 24 | 24.0 | 1.000 | 0.750 |
| B | seed11_final | 2.5 | 212 | 10 | 21.9 | 1.000 | 15 | 22.1 | 1.000 | 0.948 |
| B | seed11_final | 10.0 | 212 | 16 | 23.0 | 1.000 | 24 | 24.0 | 1.000 | 0.948 |
| B | seed23_final | 2.5 | 212 | 1 | 18.2 | 0.750 | 2 | 13.7 | 1.000 | 0.750 |
| B | seed23_final | 10.0 | 212 | 1 | 18.2 | 0.750 | 23 | 24.0 | 1.000 | 0.750 |
| C | seed11_final | 2.5 | 44 | 9 | 21.8 | 1.000 | 12 | 21.9 | 1.000 | 0.952 |
| C | seed11_final | 10.0 | 44 | 16 | 23.1 | 1.000 | 24 | 24.0 | 1.000 | 0.952 |
| C | seed23_final | 2.5 | 44 | 1 | 18.2 | 0.750 | 3 | 13.2 | 1.000 | 0.750 |
| C | seed23_final | 10.0 | 44 | 1 | 18.2 | 0.750 | 24 | 24.0 | 1.000 | 0.750 |

## Magnitude and percentage

Percentage changes are reported only for completeness. In Phase 6B the control mean is close to zero, so a small absolute gain becomes a large percentage. The decision uses absolute CNY/day and the date-clustered CI, never the percentage column.

## Accounting

Per ESS and per day, `net_income_cny = market_revenue_cny - purchase_cost_cny - cycle_cost_cny`. The terminal inventory-value term is exactly zero by design: `participant_payoff.PayoffBreakdown` has no terminal term, and the terminal SOC is pinned at 0.5 by the day-end settlement. `raw_profit` from the environment is compared with the ledger every day; the maximum difference is reported for each phase in the validation table.

## What this does not show

- It does not show how a policy trained under Z=10.0 performs. The actors are frozen `linear_core_v1` actors; only the inference map changes.
- It does not read Test, Holiday, or H2. No independent generalisation claim is supported.
- Offline quote statistics are not payoffs; this report uses the settlement ledger only.
- A positive mean with a CI that includes zero is not a robust improvement.

## Provenance

- `phase6_config.json` freezes the pairs, checkpoints, maps and decision rule.
- `provenance_addendum.json` records the two stale hashes in the Phase 0-5 audit metadata. No historical file was modified.
- The 16 actor files in `checkpoints/` hash-match the original run directories and the audit copies.

## Files

`phase6_config.json`, `source_hashes.json`, `provenance_addendum.json`, `phase6_preflight.json`, `phase6A_smoke_results.csv`, `phase6B_train_results.csv`, `phase6C_validation_results.csv`, `daily_paired_deltas.csv`, `per_ess_payoff.csv`, `fleet_payoff.csv`, `physical_validation.csv`, `solver_validation.csv`, `bootstrap_summary.json`, `output_hashes.json`.

