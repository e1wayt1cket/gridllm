# Phase 19-A: critic calibration on training-day states

Read-only. No training, no checkpoint write, no replay addition, no H2 and no
legacy test. Data: `phase19_critic_training_calibration_rows.csv`,
`phase19_critic_training_calibration.json`, `phase19_replay_coverage.csv`.
Code: `src/audit_phase19_critic_calibration.py`.

## Split membership of the five phase-18 cases

All five cases are **NEAR validation** days. They are offline cases only and
contribute no training label.

| pair | block | unit | NEAR | FULL96 |
|---|---:|---|---|---|
| 2022-12-13 \| 2025-12-11 | 3 | ESS14 | validation | test |
| 2022-12-13 \| 2025-12-11 | 12 | ESS27 | validation | test |
| 2022-12-30 \| 2025-12-31 | 10 | ESS14 | validation | train |
| 2022-12-30 \| 2025-12-31 | 11 | ESS14 | validation | train |
| 2022-12-31 \| 2025-12-28 | 17 | ESS14 | validation | train |

The D+A pilot trained on the NEAR 53 training days and selected checkpoints on
the NEAR 11 validation days, so none of the five was a training day for this
run. The FULL96 assignment differs for two of the three dates; that does not
change the NEAR role and does not make them trainable here.

## Training-day sample

Three NEAR training days (`2022-12-17`, `2022-12-24`, `2022-12-29`) × blocks 8
and 14 × ESS14/ESS27 × both legs × a seven-point quote grid plus the actor
quote. Each row is a matched counterfactual: same joint state, only the target
leg's quote changes, then the frozen deterministic actor continues to the day
end. The critic sees Q on the same state-action pair; the label is the
discounted B+C2 training return and the full-day ΔG.

| checkpoint | rows | states |
|---|---:|---:|
| D+A best (step 1272) | 192 | 24 |
| D+A final (step 3816) | 186 | 24 |

## Pooled and within-state ranking

| metric | D+A best | D+A final |
|---|---:|---:|
| Q vs training return, Pearson | +0.463 | +0.011 |
| Q vs training return, Spearman | +0.244 | −0.085 |
| Q vs full-day ΔG, Pearson | **+0.001** | **−0.253** |
| Q vs full-day ΔG, Spearman | **+0.012** | **−0.263** |
| MAE, CNY-equivalent | 304.5 | 109.4 |
| RMSE, CNY-equivalent | 308.2 | 127.2 |
| states where Q-top = return-top | **4 / 24** | **4 / 24** |
| mean within-state Spearman(Q, return) | −0.081 | −0.038 |
| mean within-state Spearman(dQ/dq, return) | −0.152 | +0.061 |
| local gradient points toward the best-return quote | 98 / 168 (58.3%) | 78 / 162 (48.1%) |

The positive pooled Q-vs-training correlation for D+A best is a between-state
level effect. It does not survive holding the state fixed: within-state ranking
is at or below chance, and against the economic quantity (full-day ΔG) the
pooled correlation is zero or negative.

## Grouped by unit, leg and quote band

D+A best:

| slice | Pearson | Spearman | mean Q (CNY) | mean return (CNY) |
|---|---:|---:|---:|---:|
| ESS14 | −0.017 | −0.060 | −345.4 | −6.5 |
| ESS27 | −0.115 | −0.056 | −246.5 | +23.6 |
| charge | +0.448 | +0.266 | −296.5 | +7.3 |
| discharge | +0.481 | +0.228 | −295.3 | +9.8 |
| low quote (< h/3) | +0.526 | +0.307 | −287.1 | +10.8 |
| mid quote | +0.467 | +0.182 | −297.0 | +8.3 |
| high quote (> 2h/3) | +0.416 | +0.234 | −303.3 | +6.6 |

D+A final:

| slice | Pearson | Spearman |
|---|---:|---:|
| ESS14 | +0.064 | −0.002 |
| ESS27 | −0.026 | −0.034 |
| charge | +0.019 | −0.105 |
| discharge | +0.005 | −0.040 |
| low quote | +0.039 | −0.091 |
| mid quote | −0.002 | −0.161 |
| high quote | +0.004 | −0.045 |

The correlation does not concentrate in one unit, one leg or one quote band.
The high-quote band is not better ranked than the low band; the failure is
throughout the covered range.

## Material training states (return spread > 10 CNY)

Four of 24 states per checkpoint are material. The mean best-quote gain over
the actor across all 24 states is only 1.9 CNY (best) and 2.2 CNY (final)
because most training-day states are nearly flat; the material states are:

| checkpoint | day | block | unit | leg | actor q | best q | actor ret | best ret | gain | spread | Q-top | rho |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best | 12-17 | 14 | ESS14 | charge | 19.2 | 50.0 | 13.1 | 13.1 | 0.1 | 12.3 | 150.0 | −0.02 |
| best | 12-17 | 14 | ESS27 | charge | 36.8 | 0.0 | 55.2 | 55.4 | 0.2 | 79.2 | 100.0 | +0.05 |
| best | 12-24 | 8 | ESS14 | discharge | 106.5 | 0.0 | −35.2 | −21.3 | 13.9 | 13.9 | 50.0 | −0.19 |
| best | 12-24 | 8 | ESS27 | discharge | 73.4 | 0.0 | +7.2 | +22.8 | 15.6 | 15.6 | 0.0 | +0.31 |
| final | 12-17 | 14 | ESS14 | charge | 0.0 | 0.0 | −17.0 | −17.0 | 0.0 | 11.8 | 125.0 | −0.50 |
| final | 12-17 | 14 | ESS27 | charge | 76.1 | 0.0 | −33.7 | −17.8 | 15.8 | 15.8 | 75.0 | +0.45 |
| final | 12-24 | 8 | ESS14 | discharge | 248.6 | 0.0 | −131.0 | −119.8 | 11.2 | 11.2 | 248.6 | −0.38 |
| final | 12-24 | 8 | ESS27 | discharge | 300.0 | 0.0 | −99.4 | −83.6 | 15.9 | 15.9 | 0.0 | +0.48 |

One of four material states has the Q-top at the return-top for each
checkpoint. The same pattern as the validation cases is reproduced on training
days: the actor often sits at a high discharge quote, the best quote is 0, and
the critic does not rank 0 first.

## Twin critics

| checkpoint | mean abs(Q1−Q2) | median | p95 | states with gap > 0.01 (scaled) |
|---|---:|---:|---:|---:|
| D+A best | 0.0268 (26.8 CNY) | 0.0209 | 0.0655 | 24 / 24 |
| D+A final | 0.0163 (16.3 CNY) | 0.0088 | 0.0572 | 24 / 24 |

The target uses `min(Q1,Q2)`. The twin gap is comparable to or larger than the
material gains (11–16 CNY on training days, 21–139 CNY on the validation
cases), so the ordering is not stable under the min target.

## Replay coverage and action extrapolation

D+A best buffer: 1272 transitions (53 episodes, one exposure per day).
D+A final buffer: 3816 transitions (159 episodes, three exposures per day).

"Grid coverage" is the fraction of the eight calibration quotes for which the
buffer contains at least one transition from the same unit within ±10% of the
leg's high bound, ±3 blocks and ±0.10 SOC.

| checkpoint | mean grid coverage | executed quote p05 / median / p95, ESS14 discharge |
|---|---:|---|
| D+A best | 87.0% | 0.0 / 35.0 / 258.0 |
| D+A final | 98.2% | 0.0 / 133.4 / 300.0 |

The high-quote region is not missing from the replay; in D+A final the median
ESS14 discharge quote in the buffer is 133.4. The residual extrapolated points
are the extreme low and top grid points at a few states, not the region where
the material failures sit. The defect is therefore a ranking error on covered
states, not a missing action region.

The nearest-state distance (z-scored observation L2) is 0.3–1.7, so the
calibration states are inside the state distribution as well.

## What this answers

1. **Is the critic's failure validation-only?** No. It reproduces on NEAR
   training days: Q-top equals the return-top in 4/24 states, and the local
   gradient points toward the best quote only 58% (best) / 48% (final) of the
   time, at or below chance.
2. **Is the action region absent from the replay?** No. The covered fraction of
   the quote grid is 87–98%; the high-quote region is the most frequent region
   in the final buffer.
3. **Is the label scale wrong?** No. Q is compared in its own training units;
   ×1000 gives CNY-equivalent. The MAE (109–305 CNY) is larger than the
   material gains, so the critic is not a usable absolute predictor either, but
   the decisive check is the within-state ordering, not the level.
4. **Is the twin target stable?** No. Both twins disagree on ordering in every
   state; the gap is of the same order as the effect being ranked.

## Boundary

The rollout continues the frozen deterministic actor from each state; it is
not the four-agent dynamic equilibrium. The quote grid is a single-block
counterfactual family, not an all-day fixed-quote policy. Only NEAR training
days contribute labels; the five phase-18 cases remain offline-only.
