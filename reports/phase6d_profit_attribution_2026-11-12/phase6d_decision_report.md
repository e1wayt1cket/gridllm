# Phase 6D — frozen-policy profit attribution

label: `NON_FORMAL_DIAGNOSTIC`; attribution only; no clearing, no training, no production change.

```text
PHASE6_ORIGINAL_VERDICT = PARTIAL
PHASE6D_ANALYSIS = PARTIAL
Z10_CANDIDATE = RETAINED
PRODUCTION_MAPPING = UNCHANGED
ACTION_SCHEMA = UNCHANGED
FORMAL_TRAINING = NOT_STARTED
OPTIMIZER_STEP = NONE
REPLAY_WRITE = NONE
COMMIT_PUSH = NONE
```

The Phase 6 verdict is preserved unchanged: `PARTIAL`. This stage explains the Phase 6 numbers; it does not re-adjudicate them.

## Scope and evidence

- Inputs: the 320 Phase 6 day-runs and their derived tables, read only.
- The four primary decision units are Seed 11 final / Train, Seed 23 final / Train, Seed 11 final / Validation, Seed 23 final / Validation. Seeds and splits are never pooled.
- Phase 6A best/final rows are auxiliary description only.
- Source integrity: `PASS` across 15 checks.
- Accounting: maximum identity residual 6.821e-13 CNY, fleet-minus-sum-ESS residual 0.000e+00 CNY, paired-delta identity residual 7.958e-13 CNY, paired-delta fleet-minus-sum 1.705e-13 CNY; tolerance 1e-06 CNY.

## Phase 6D-0 — source and accounting preflight

| check | status | detail |
|---|---|---|
| phase6_config | PASS | label=NON_FORMAL_DIAGNOSTIC, control=2.5, candidate=10.0 |
| output_hashes | PASS | 14 recorded files, mismatches=none |
| source_hashes | PASS | 43 recorded source/checkpoint files on disk, mismatches=none |
| phase6_input_sources_on_disk | PASS | 27 Phase 6 source inputs recorded, missing=none |
| checkpoint_actors | PASS | 16 actor files, mismatches=none |
| result_row_counts | PASS | counts={'A': 64, 'B': 212, 'C': 44}, expected={'A': 64, 'B': 212, 'C': 44} |
| no_duplicate_rows | PASS | duplicate keys=none |
| arm_pairing_complete | PASS | 160 checkpoint/date groups, incomplete=none |
| arm_metadata_identical | PASS | mismatched fields=none |
| day_ok_all | PASS | failed days=none |
| terminal_inventory_zero | PASS | max |term|=0.000e+00 |
| pairs_declared_in_config | PASS | 64 distinct dates, outside=none |
| phase6_preflight | PASS | all_pass=True |
| original_verdict_preserved | PASS | Phase 6 report states PARTIAL |
| bootstrap_config | PASS | config={'n_resamples': 1000, 'seed': 20261112, 'cluster': 'pair_id (date)', 'interval': 'percentile'} |

Overall source status: **`PASS`**.

## Phase 6D-1 — date contribution and concentration

| combo | n | mean CNY | median CNY | positive days | |delta| < 0.01 | S+ | S- | CR5+ | CR5- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seed11_final_train | 53 | 29.46 | 0.0002 | 40/53 | 19/53 | 1839.03 | 277.55 | 0.548 | 0.881 |
| seed23_final_train | 53 | 28.15 | 0.0002 | 38/53 | 20/53 | 1755.48 | 263.64 | 0.522 | 0.935 |
| seed11_final_validation | 11 | 47.67 | 0.0002 | 8/11 | 5/11 | 533.12 | 8.71 | 1.000 | 1.000 |
| seed23_final_validation | 11 | 37.55 | 0.0001 | 8/11 | 5/11 | 440.23 | 27.21 | 1.000 | 1.000 |

The mean is not a typical day. A positive mean with a median near zero means the aggregate gain comes from a minority of dates; the concentration ratios above quantify that tail.

### Top and bottom contributors

**seed11_final_train**

| tail | rank | date | delta CNY | share of tail |
|---|---:|---|---:|---:|
| top | 1 | 2022-11-12|2025-11-09 | 279.74 | 0.152 |
| top | 2 | 2022-12-13|2025-12-11 | 200.25 | 0.109 |
| top | 3 | 2022-12-10|2025-12-07 | 183.00 | 0.100 |
| bottom | 1 | 2022-11-15|2025-11-13 | -75.94 | 0.274 |
| bottom | 2 | 2022-10-26|2025-10-24 | -61.42 | 0.221 |
| bottom | 3 | 2022-12-26|2025-12-26 | -43.71 | 0.157 |

CR1+ = 0.152, CR3+ = 0.361, CR5+ = 0.548; CR1- = 0.274, CR3- = 0.652, CR5- = 0.881.

**seed23_final_train**

| tail | rank | date | delta CNY | share of tail |
|---|---:|---|---:|---:|
| top | 1 | 2022-11-12|2025-11-09 | 225.62 | 0.129 |
| top | 2 | 2022-10-23|2025-10-18 | 196.25 | 0.112 |
| top | 3 | 2022-12-30|2025-12-31 | 170.11 | 0.097 |
| bottom | 1 | 2022-11-15|2025-11-13 | -73.13 | 0.277 |
| bottom | 2 | 2022-12-26|2025-12-26 | -63.79 | 0.242 |
| bottom | 3 | 2022-10-26|2025-10-24 | -58.85 | 0.223 |

CR1+ = 0.129, CR3+ = 0.337, CR5+ = 0.522; CR1- = 0.277, CR3- = 0.743, CR5- = 0.935.

**seed11_final_validation**

| tail | rank | date | delta CNY | share of tail |
|---|---:|---|---:|---:|
| top | 1 | 2022-12-31|2025-12-28 | 373.56 | 0.701 |
| top | 2 | 2022-12-23|2025-12-24 | 81.36 | 0.153 |
| top | 3 | 2022-12-28|2025-12-29 | 73.91 | 0.139 |
| bottom | 1 | 2022-11-05|2025-11-02 | -6.23 | 0.716 |
| bottom | 2 | 2022-11-19|2025-11-15 | -2.47 | 0.284 |
| bottom | 3 | 2022-08-25|2025-08-26 | -0.00 | 0.000 |

CR1+ = 0.701, CR3+ = 0.992, CR5+ = 1.000; CR1- = 0.716, CR3- = 1.000, CR5- = 1.000.

**seed23_final_validation**

| tail | rank | date | delta CNY | share of tail |
|---|---:|---|---:|---:|
| top | 1 | 2022-12-31|2025-12-28 | 361.02 | 0.820 |
| top | 2 | 2022-12-28|2025-12-29 | 40.70 | 0.092 |
| top | 3 | 2022-12-23|2025-12-24 | 37.77 | 0.086 |
| bottom | 1 | 2022-11-19|2025-11-15 | -27.12 | 0.997 |
| bottom | 2 | 2022-11-05|2025-11-02 | -0.09 | 0.003 |
| bottom | 3 | 2022-08-25|2025-08-26 | -0.00 | 0.000 |

CR1+ = 0.820, CR3+ = 0.998, CR5+ = 1.000; CR1- = 0.997, CR3- = 1.000, CR5- = 1.000.

### Top-k removal sensitivity (descriptive only)

| combo | full mean | drop Top1 | drop Top3 | drop Top5 | drop Bottom1 | drop Bottom3 | drop Bottom5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed11_final_train | 29.46 | 24.65 | 17.97 | 11.55 | 31.49 | 34.85 | 37.62 |
| seed23_final_train | 28.15 | 24.35 | 18.00 | 11.98 | 30.10 | 33.75 | 36.21 |
| seed11_final_validation | 47.67 | 15.09 | -0.55 | -1.45 | 53.07 | 66.64 | 88.85 |
| seed23_final_validation | 37.55 | 5.20 | -3.31 | -4.53 | 44.01 | 55.03 | 73.37 |

Removing the best dates lowers the mean; removing the worst dates raises it. Neither replaces the pre-registered mean or the date-clustered CI, and no removed-date subsample is a new test set.

## Phase 6D-2 — ESS and leg attribution

| combo | ESS | mean delta CNY | positive days | negative days | share of fleet delta |
|---|---|---:|---:|---:|---:|
| seed11_final_train | fleet | 29.46 | 40 | 13 | 1.000 |
| seed11_final_train | ESS9 | 1.42 | 26 | 27 | 0.048 |
| seed11_final_train | ESS14 | 11.64 | 39 | 14 | 0.395 |
| seed11_final_train | ESS17 | 9.64 | 37 | 16 | 0.327 |
| seed11_final_train | ESS27 | 6.76 | 40 | 13 | 0.229 |
| seed23_final_train | fleet | 28.15 | 38 | 15 | 1.000 |
| seed23_final_train | ESS9 | 8.13 | 37 | 16 | 0.289 |
| seed23_final_train | ESS14 | 1.52 | 36 | 17 | 0.054 |
| seed23_final_train | ESS17 | 8.92 | 40 | 13 | 0.317 |
| seed23_final_train | ESS27 | 9.57 | 39 | 14 | 0.340 |
| seed11_final_validation | fleet | 47.67 | 8 | 3 | 1.000 |
| seed11_final_validation | ESS9 | 2.01 | 7 | 4 | 0.042 |
| seed11_final_validation | ESS14 | 15.08 | 9 | 2 | 0.316 |
| seed11_final_validation | ESS17 | 15.27 | 9 | 2 | 0.320 |
| seed11_final_validation | ESS27 | 15.32 | 9 | 2 | 0.321 |
| seed23_final_validation | fleet | 37.55 | 8 | 3 | 1.000 |
| seed23_final_validation | ESS9 | 14.65 | 8 | 3 | 0.390 |
| seed23_final_validation | ESS14 | 0.06 | 5 | 6 | 0.002 |
| seed23_final_validation | ESS17 | 13.85 | 8 | 3 | 0.369 |
| seed23_final_validation | ESS27 | 8.99 | 9 | 2 | 0.239 |

Per-ESS rows are attribution, not separate decisions. A negative ESS leg can coexist with a positive fleet delta, and the reverse.

### Fleet transfer check

| combo | fleet+ dates | fleet+ with a losing ESS | fleet- dates | fleet- with a winning ESS | all ESS same sign as fleet |
|---|---:|---:|---:|---:|---:|
| seed11_final_train | 40 | 19 | 13 | 7 | 27 |
| seed23_final_train | 38 | 16 | 15 | 13 | 24 |
| seed11_final_validation | 8 | 2 | 3 | 2 | 7 |
| seed23_final_validation | 8 | 6 | 3 | 3 | 2 |

### Historical negative examples (Round 1 auxiliary)

- seed11_best / ESS17: mean delta -5.49 CNY/day over 8 probe dates; fleet mean delta 1.16. Round 1 auxiliary only; the CI is in `bootstrap_reproduction.csv`.
- seed23_best / ESS14: mean delta -7.91 CNY/day over 8 probe dates; fleet mean delta 7.04. Round 1 auxiliary only; the CI is in `bootstrap_reproduction.csv`.

## Phase 6D-3 — quote-to-clearing attribution

The Phase 6 runner deleted its per-block quote, gate and LMP series before writing results, so block-level items are `NOT_TESTED`. Day-level aggregates below are the strongest available evidence.

| combo | ESS | charge open blocks (C → X) | discharge open blocks (C → X) | charge quote days changed | discharge quote days changed |
|---|---|---|---|---:|---:|
| seed11_final_train | ESS9 | 1006 → 1006 | 1272 → 1272 | 53/53 | 53/53 |
| seed11_final_train | ESS14 | 1272 → 1272 | 1272 → 1272 | 53/53 | 53/53 |
| seed11_final_train | ESS17 | 1272 → 1272 | 1272 → 1272 | 53/53 | 53/53 |
| seed11_final_train | ESS27 | 1272 → 1272 | 1272 → 1272 | 53/53 | 53/53 |
| seed23_final_train | ESS9 | 1272 → 1272 | 1272 → 1272 | 53/53 | 53/53 |
| seed23_final_train | ESS14 | 0 → 0 | 1272 → 1272 | 0/0 | 53/53 |
| seed23_final_train | ESS17 | 1272 → 1272 | 1272 → 1272 | 53/53 | 53/53 |
| seed23_final_train | ESS27 | 1272 → 1272 | 1272 → 1272 | 53/53 | 53/53 |
| seed11_final_validation | ESS9 | 213 → 213 | 264 → 264 | 11/11 | 11/11 |
| seed11_final_validation | ESS14 | 264 → 264 | 264 → 264 | 11/11 | 11/11 |
| seed11_final_validation | ESS17 | 264 → 264 | 264 → 264 | 11/11 | 11/11 |
| seed11_final_validation | ESS27 | 264 → 264 | 264 → 264 | 11/11 | 11/11 |
| seed23_final_validation | ESS9 | 264 → 264 | 264 → 264 | 11/11 | 11/11 |
| seed23_final_validation | ESS14 | 0 → 0 | 264 → 264 | 0/0 | 11/11 |
| seed23_final_validation | ESS17 | 264 → 264 | 264 → 264 | 11/11 | 11/11 |
| seed23_final_validation | ESS27 | 264 → 264 | 264 → 264 | 11/11 | 11/11 |

Across primary groups, quotes are comparable on 960 ESS-days and change on 960 of them. Quote changes are therefore observable at day level, but the block-level transmission from quote to cleared volume cannot be measured from the stored data.

### Single-charge-quote recheck

176 ESS-days have a single distinct charge quote; 176 of 176 have the charge gate closed for all 24 blocks. A constant quote there is the policy and gate outcome, not a mapping failure, and it carries no direct price-map exposure.

## Phase 6D-4 — bootstrap and robustness

Bootstrap reproduction: 40 groups, maximum absolute difference 0.000e+00, all match = True. Configuration is the Phase 6 one: 1000 resamples, seed 20261112, percentile interval, cluster = date.

| combo | mean | median | CI low | CI high | reproduced |
|---|---:|---:|---:|---:|---|
| seed11_final_train | 29.46 | 0.0002 | 12.00 | 49.06 | True |
| seed23_final_train | 28.15 | 0.0002 | 11.82 | 45.62 | True |
| seed11_final_validation | 47.67 | 0.0002 | 1.17 | 115.82 | True |
| seed23_final_validation | 37.55 | 0.0001 | -2.34 | 105.33 | True |

The Phase 6 decision rule is unchanged. Seed 23 final / Validation still has a CI lower bound below zero, so the original verdict remains `PARTIAL`.

## Phase 6D-5 — classification and answers

| combo | classification |
|---|---|
| seed11_final_train | Concentrated improvement |
| seed23_final_train | Concentrated improvement |
| seed11_final_validation | Concentrated improvement |
| seed23_final_validation | Concentrated improvement |

No primary combination is broad-based. Each has a positive mean carried by a minority of dates, a large near-zero block, and a negative tail.

### Required questions

1. **Do the four primary groups reproduce the Phase 6 statistics?** `PASS` — 40 bootstrap groups match to 0.000e+00 CNY.
2. **Does the concentration match the original report?** `PASS` — the original report already showed medians near zero; this stage adds the top/bottom tail shares.
3. **Largest single date in each group.**
   - seed11_final_train: 2022-11-12|2025-11-09 contributes 279.74 CNY, 15.2% of the positive total.
   - seed23_final_train: 2022-11-12|2025-11-09 contributes 225.62 CNY, 12.9% of the positive total.
   - seed11_final_validation: 2022-12-31|2025-12-28 contributes 373.56 CNY, 70.1% of the positive total.
   - seed23_final_validation: 2022-12-31|2025-12-28 contributes 361.02 CNY, 82.0% of the positive total.
4. **Mean after removing the best and worst dates.**
   - seed11_final_train: full 29.46; drop Top1 24.65, Top3 17.97, Top5 11.55; drop Bottom1 31.49, Bottom3 34.85, Bottom5 37.62 CNY/day.
   - seed23_final_train: full 28.15; drop Top1 24.35, Top3 18.00, Top5 11.98; drop Bottom1 30.10, Bottom3 33.75, Bottom5 36.21 CNY/day.
   - seed11_final_validation: full 47.67; drop Top1 15.09, Top3 -0.55, Top5 -1.45; drop Bottom1 53.07, Bottom3 66.64, Bottom5 88.85 CNY/day.
   - seed23_final_validation: full 37.55; drop Top1 5.20, Top3 -3.31, Top5 -4.53; drop Bottom1 44.01, Bottom3 55.03, Bottom5 73.37 CNY/day.
5. **Negative tail.** S- and CR5- are reported above for every primary group; none of them is zero.
6. **ESS attribution.** The fleet delta is the sum of the four ESS deltas; the transfer table counts the dates on which a fleet gain contains a losing ESS and the reverse.
7. **Share of ESS-days with the charge gate closed all day.** 176 single-charge-quote ESS-days, all with the charge gate closed for 24 blocks; the full gate-closed share per ESS-day is in `gate_exposure_summary.csv`.
8. **How much quote change reaches clearing?** Day-level quote changes exist on 960 comparable ESS-days; the block-level clearing response is `NOT_TESTED` because block series were not stored.
9. **Where the gain comes from.** In every primary group the mean market-revenue delta and the mean cycle-cost delta are both negative; cycle cost falls by more than revenue. The positive net is therefore a throughput/cost-avoidance effect: the candidate cycles less. The largest positive dates are dominated by large throughput reductions, and on some of them revenue rises as well. This is not a revenue improvement.
10. **Is the evidence sufficient to design a new-map training experiment?** See the recommendation section.

### Component decomposition of the fleet delta

| combo | market revenue | purchase cost | cycle cost | net |
|---|---:|---:|---:|---:|
| seed11_final_train | -16.14 | -10.00 | -35.60 | 29.46 |
| seed23_final_train | -19.35 | -10.98 | -36.51 | 28.15 |
| seed11_final_validation | -30.06 | -18.82 | -58.92 | 47.67 |
| seed23_final_validation | -53.08 | -20.96 | -69.66 | 37.55 |

The same pattern holds in every primary group: mean market revenue falls, purchase cost falls, cycle cost falls, and the net gain is the part of the cycle-cost reduction that the revenue loss does not consume. The top-day table shows the extreme cases, where the candidate cuts charge and discharge volume by one to two MWh on a day. Component moves on a single date can be large and cancel; the net delta, not the dominant component, is the payoff.

## Recommendation

**`ADDITIONAL_EVIDENCE_REQUIRED`**

The gain is concentrated and cost-side. In every primary group the median is within 0.001 CNY/day of zero, the top 5 positive dates carry 52-100% of the positive total, and removing the top 3 positive dates turns both Validation means negative. The positive net comes from lower cycle cost as throughput falls, not from higher market revenue, and Seed 23 final / Validation keeps a 95% CI that includes zero. Day-level quote changes are visible, but the block-level quote-to-clearing chain was not persisted, so the mechanism cannot be verified at the block level. Design work on a new mapping may proceed only after the trace gap is closed by a separately approved run; the current candidate is not a production candidate.

This is not an approval to switch the production mapping or to start training. Any follow-up must be a separately reviewed protocol with a new mapping version, a new action schema, new checkpoint/replay manifests and independent validation.

## NOT_TESTED

- block-level gate state agreement between arms: the Phase 6 runner persisted only day aggregates; no block series exist on disk, and re-running clearing is out of scope.
- block-level quote change counts and magnitudes: the Phase 6 runner persisted only day aggregates; no block series exist on disk, and re-running clearing is out of scope.
- block-level clearing volume and LMP responses: the Phase 6 runner persisted only day aggregates; no block series exist on disk, and re-running clearing is out of scope.
- the first block at which the two arms diverge: the Phase 6 runner persisted only day aggregates; no block series exist on disk, and re-running clearing is out of scope.
- per-block quote-to-clearing mediation: the Phase 6 runner persisted only day aggregates; no block series exist on disk, and re-running clearing is out of scope.

## Files

`phase6d_config.json`, `input_inventory.csv`, `source_integrity_report.json`, `accounting_reconciliation.csv`, `daily_contribution_rank.csv`, `return_distribution_summary.csv`, `concentration_sensitivity.csv`, `top_bottom_contributors.csv`, `per_ess_decomposition.csv`, `per_leg_exposure.csv`, `fleet_transfer_analysis.csv`, `accounting_component_summary.csv`, `gate_exposure_summary.csv`, `quote_change_summary.csv`, `quote_to_clearing_trace.csv`, `dynamic_path_attribution.csv`, `top_bottom_day_mechanism.csv`, `single_quote_day_recheck.csv`, `bootstrap_reproduction.csv`, `bootstrap_reproduction.json`, `robustness_summary.csv`, `phase6d_decision_report.md`, `source_hashes.json`, `output_hashes.json`.

