# Critic joint state–action coverage (P0)

Read-only audit of the two saved replay buffers against the validation winter
states where the long-horizon critic mis-ranks quotes. No training, no
checkpoint write, no H2, no legacy test. Data:
`critic_joint_coverage_targets.csv`, `critic_joint_coverage_states.json`,
`critic_joint_coverage_summary.json`.

## Checkpoints and buffer

| checkpoint | step | transitions | episodes | distinct training dates |
|---:|---:|---:|---:|---:|
| R2 | 2544 | 2544 | 106 | 53 |
| R10 | 12720 | 12720 | 530 | 53 |

The buffer stores the observation at each decision, the executed quote, the
hard gate and the gate probability. `episodes.csv` maps buffer index
`i -> i // 24` to the training pair, so the nearest transition can be named by
date and block instead of only by distance.

The three audit days are **validation** days and never enter the buffer:
2022-12-13|2025-12-11, 2022-12-30|2025-12-31, 2022-12-31|2025-12-28. The two
controls are validation non-winter days: 2022-07-23|2025-07-13 and
2022-10-26|2025-10-24.

## Normalization

The joint state has 36 dimensions: for each ESS, `load_feat`, `re_feat`, `soc`,
`avg_other_dev`, `all_dev_std`, `ema_dev`, `price_trend`, `pred_lmp_norm`
(8 × 4 = 32); plus the four features that are identical for all agents,
`last_lmp_norm`, `block_pos`, `avg_lmp_hist_norm`, `slr_norm` (4). The two
opponent features are per agent because they exclude self, and the three
forecaster features are per agent because each agent holds its own EMA.

Distances use fixed, interpretable scales, then divide the Euclidean distance
by `sqrt(36)`:

| feature | scale |
|---|---:|
| load, renewable | 1.0 (already in [0,1]) |
| SOC | 0.25 (one quarter of the SOC range) |
| opponent deviation / spread | 1.0 (already normalized) |
| price features | 1.0 (already divided by `PRICE_REF_CNY_MWH`) |
| block position | 1/24 (one block) |

The summary also reports a z-score distance using the buffer's per-feature
mean and standard deviation. The fixed-scale version is the primary one; the
z-score version leads to the same ranking but over-weights SOC when the buffer
SOC variance is small. Action distance uses the quote divided by the leg upper
bound (charge 150, discharge 300) and the hard gate as 0/1.

## Material states

`nn state` is the distance to the nearest buffer transition from a **different
episode**; `pct` is its percentile in the buffer's own nearest-neighbour
distribution (1.0 = farther than 99% of buffer states). `k50 act` is the
minimum joint quote distance among the 50 nearest states. `joint` counts
transitions with block ±1, target SOC ±0.10 and target quote ±10% of the high
simultaneously; `hi k50` counts transitions at ≥0.86 × high among those 50.

| ckpt | day | block | ESS | leg | actor quote | best quote | worst quote | ΔG spread | ρ(Q,ΔG) | nn state | pct | nearest date/b | k50 act | joint | hi k50 |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 10 | 12-13 | 11 | ESS14 | charge | 0.0 | 112.5 | 0.0 | 36.0 | −0.60 | 0.234 | 0.99 | 12-09 b11 | 0.138 | 71 | 3 |
| 10 | 12-13 | 11 | ESS27 | charge | 0.0 | 112.5 | 37.5 | 17.8 | −0.90 | 0.234 | 0.99 | 12-09 b11 | 0.138 | 49 | 13 |
| 10 | 12-13 | 11 | ESS27 | discharge | 211.3 | 0.0 | 300.0 | 30.6 | −0.14 | 0.234 | 0.99 | 12-09 b11 | 0.138 | 75 | 1 |
| 10 | 12-13 | 20 | ESS14 | discharge | 1.4 | 150.0 | 300.0 | 24.4 | +0.54 | 0.132 | 0.93 | 10-20 b20 | 0.067 | 34 | 0 |
| 10 | 12-13 | 20 | ESS27 | discharge | 87.1 | 0.0 | 300.0 | 27.1 | +0.03 | 0.132 | 0.93 | 10-20 b20 | 0.067 | 18 | 0 |
| 10 | 12-13 | 2 | ESS27 | discharge | 285.1 | 0.0 | 285.1 | 16.2 | −0.77 | 0.020 | 0.61 | 10-11 b2 | 0.016 | 239 | 10 |
| 10 | 12-30 | 11 | ESS14 | discharge | 299.2 | 0.0 | 299.2 | 56.1 | −0.94 | 0.216 | 0.99 | 10-20 b11 | 0.008 | 55 | 19 |
| 10 | 12-30 | 11 | ESS27 | discharge | 282.0 | 0.0 | 300.0 | 62.7 | −0.77 | 0.216 | 0.99 | 10-20 b11 | 0.008 | 9 | 8 |
| 2 | 12-30 | 11 | ESS14 | discharge | 0.0 | 0.0 | 300.0 | 19.5 | +1.00 | 0.180 | 0.93 | 10-20 b11 | 0.051 | 32 | 0 |
| 2 | 12-30 | 11 | ESS27 | discharge | 51.7 | 0.0 | 225.0 | 17.8 | −0.71 | 0.180 | 0.93 | 10-20 b11 | 0.051 | 137 | 0 |

`best quote` is the quote with the highest actual day ΔG; `worst quote` the
lowest. The ranking column is the within-group Spearman of `min(Q1,Q2)` against
the full-day ΔG.

The phase-15 calibration grid used blocks 2/11/20. The winter-loss
reconstruction puts the worst R10 blocks at 12-30 b10 and 12-31 b17, which that
grid did not include. The coverage audit adds them:

| ckpt | day | block | ESS | leg | actor quote | nn state | pct | nearest date/b | k50 act | joint | hi k50 | hi at time |
|---:|---|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| 10 | 12-31 | 17 | ESS14 | discharge | 272.5 | 0.170 | 0.97 | 12-17 b17 | 0.063 | 43 | 4 | 100 |
| 10 | 12-31 | 17 | ESS27 | discharge | 258.4 | 0.170 | 0.97 | 12-17 b17 | 0.063 | 20 | 4 | 152 |
| 2 | 12-31 | 17 | ESS14 | discharge | 0.0 | 0.114 | 0.81 | 12-19 b17 | 0.029 | 169 | 0 | 0 |
| 2 | 12-31 | 17 | ESS27 | discharge | 49.4 | 0.114 | 0.81 | 12-19 b17 | 0.029 | 125 | 0 | 0 |

The 10 material groups above are therefore a **lower bound** on the material
state set, not a complete census. The calibration recheck can only rank the
states in its grid; the winter 15-minute audit is the source for b10/b17.

## Aggregate

| slice | rows | mean nn state | rows within buffer median nn | mean k50 action | rows with joint time+SOC+quote | rows with high quote in k50 |
|---|---:|---:|---:|---:|---:|---:|
| winter | 96 | 0.146 | 16/96 | 0.047 | 96/96 | 35/96 |
| control | 16 | 0.123 | 0/16 | 0.045 | 16/16 | 7/16 |

Buffer nearest-neighbour quantiles (different episode): R2 median 0.028,
95th 0.198; R10 median 0.0076, 95th 0.146. The z-score version gives R2 median
0.181 and R10 median 0.117.

## Findings

1. **The state is covered only at the tail.** Every audit state has at least
   one buffer transition with the same block, a close SOC and a close quote,
   but the number is small. At the mid-day material states the state itself is
   at the 0.93–0.99 percentile of the buffer's own nearest-neighbour
   distribution, and the nearest transition is usually a different winter
   training day at the same block (12-09, 12-17, 12-19, 12-21 or 10-20), not
   a same-day state. The early-block states are much closer (0.020–0.076).
2. **Joint action coverage is thin where the loss is.** At 12-30 b11 the
   time+SOC+quote count is 55 for ESS14 and 9 for ESS27; the high-quote count
   among the 50 nearest states is 19 and 8. At 12-13 b11 it is 71/49 and 3/13.
   At 12-13 b20 the high-quote count in the 50 nearest states is 0 for both
   units. High quotes are not absent from the buffer — the R10 buffer has
   2,307 and 1,276 transitions at ≥0.86 × high for ESS14 and ESS27 discharge —
   but they cluster at other blocks and other SOCs, not at the material
   state's block and SOC.
3. **Coverage alone does not explain the failure.** At 12-13 b2 ESS27 the
   state is at the 0.61 percentile (distance 0.020), there are 239 joint
   transitions and 10 high-quote neighbours, and the critic still anti-ranks
   the quote (ρ = −0.77) with the actor at the worst quote. Conversely, at
   12-13 b20 ESS14 there are no high-quote neighbours in the 50 nearest
   states, yet the critic ranks correctly (ρ = +0.54) and the actor is near
   truthful. The critic's failure is therefore a calibration problem first
   and a conditional-coverage problem second.
4. **The gap is distributional, not marginal.** A marginal quote histogram
   would call the action space covered. The conditional audit shows that the
   joint (block, SOC, quote) region the policy actually visits in the winter
   loss states is sparse, and that the critic fails even where it is not.
5. **Implication for a repair.** Improving only the replay action distribution
   would not fix the b2 state; improving only the critic target would leave the
   b11/b10 states with very few high-quote examples. The D design should do
   both: targeted quote exploration on high-price training blocks and a critic
   sampling/weighting rule that values those transitions, with the validation
   states kept out of training.

## Boundary

The buffer does not record the day id per transition; the mapping uses the
episode order in `episodes.csv` (`i // 24`), which is exact for this run. The
audit reads only training/validation days; H2 was not read and the legacy test
was not run.
