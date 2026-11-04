# Critic return calibration recheck (P0)

Read-only recheck of the phase-15 rollout rows. No training, no clearing, no
checkpoint write. Data: `critic_return_calibration_rows.csv`,
`critic_return_calibration_groups.csv`,
`critic_return_calibration_summary.json`.

## What is compared

The critic is trained on `reward / fixed_scale`, with `fixed_scale = 1000`.
`Q` is therefore in reward units and `Q * 1000` is the critic's CNY-equivalent.
The same-state, same-action rollout return is

```
training_return = (gamma ** (23 - block) * full_day_delta
                   + discounted_C2) / 1000
```

because the B reward is paid only at block 23 and the C2 shaping is paid every
block. This is the object the critic's Bellman target estimates. The audit also
reports two other objects that are **not** the training return:

* `future_payoff`: the undiscounted sum of the remaining raw policy block
  profits, with no B1 and no C2;
* `future_minus_b1`: the remaining policy payoff minus the remaining B1 payoff.

They are useful for economics, but substituting them for the training return
would compare different definitions, not different scales.

## Overall

| comparison (n=499) | Pearson | Spearman |
|---|---:|---:|
| Q vs training return (same units) | −0.566 | −0.570 |
| Q×1000 vs training return CNY | −0.566 | −0.570 |
| Q×1000 vs remaining settlement | −0.025 | +0.019 |
| Q×1000 vs remaining B1 counterfactual | +0.016 | +0.240 |
| Q×1000 vs full-day ΔG | −0.477 | −0.420 |
| Q×1000 vs discounted settlement | −0.046 | −0.014 |

The phase-15 headline is reproducible: the pooled Q is anti-correlated with the
training return. The settlement and B1 columns are near zero because they are a
different object, not because the scale is wrong.

## Where the pooled sign comes from

| slice | n | Pearson Q vs training | Spearman Q vs training | mean error Q×1000 − training (CNY) |
|---|---:|---:|---:|---:|
| checkpoint R2 | 253 | −0.389 | −0.434 | −507.4 |
| checkpoint R10 | 246 | −0.143 | −0.089 | −2.2 |
| winter days | 409 | −0.703 | −0.754 | −270.5 |
| control days | 90 | +0.484 | +0.403 | −203.3 |
| early blocks (<8) | 135 | −0.695 | −0.789 | −321.1 |
| mid blocks (8–15) | 226 | −0.429 | −0.361 | −254.2 |
| late blocks (>15) | 138 | −0.672 | −0.674 | −203.9 |
| ESS14 | 202 | −0.724 | −0.702 | −414.1 |
| ESS27 | 207 | −0.682 | −0.572 | −130.3 |
| ESS17 | 44 | +0.436 | +0.238 | −417.4 |
| ESS9 | 46 | +0.030 | +0.408 | +1.4 |

The sign flips by region: positive on the control days and ESS17, strongly
negative on the winter days and ESS14/27. R2 also carries a large negative
offset (−507 CNY) while R10 is approximately centred (−2 CNY). A single pooled
correlation therefore describes a level difference between checkpoints as much
as a ranking failure.

## Material states (actual ΔG spread > 10 CNY)

Within-group Spearman is the decisive check. `best` is the quote with the
highest actual day ΔG; `worst` the lowest.

| ckpt | day | block | ESS | leg | actor | best | worst | ΔG spread | rho Q,ΔG | rank inversions | twin disagreements |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 10 | 12-13 | 11 | ESS14 | charge | 0.0 | 112.5 | 0.0 | 36.0 | −0.60 | 70% | 40% |
| 10 | 12-13 | 11 | ESS27 | charge | 0.0 | 112.5 | 37.5 | 17.8 | −0.90 | 90% | 20% |
| 10 | 12-13 | 11 | ESS27 | discharge | 211.3 | 0.0 | 300.0 | 30.6 | −0.14 | 60% | 27% |
| 10 | 12-13 | 20 | ESS14 | discharge | 1.4 | 150.0 | 300.0 | 24.4 | +0.54 | 27% | 27% |
| 10 | 12-13 | 20 | ESS27 | discharge | 87.1 | 0.0 | 300.0 | 27.1 | +0.03 | 47% | 20% |
| 10 | 12-13 | 2 | ESS27 | discharge | 285.1 | 0.0 | 285.1 | 16.2 | −0.77 | 80% | 20% |
| 10 | 12-30 | 11 | ESS14 | discharge | 299.2 | 0.0 | 299.2 | 56.1 | −0.94 | 93% | 0% |
| 10 | 12-30 | 11 | ESS27 | discharge | 282.0 | 0.0 | 300.0 | 62.7 | −0.77 | 80% | 27% |
| 2 | 12-30 | 11 | ESS14 | discharge | 0.0 | 0.0 | 300.0 | 19.5 | +1.00 | 0% | 50% |
| 2 | 12-30 | 11 | ESS27 | discharge | 51.7 | 0.0 | 225.0 | 17.8 | −0.71 | 73% | 7% |

The material rows have 57 quote-level rows in 10 groups. The critic is
anti-ranked in 7 of 10 groups, near-flat in one, and correctly ranked in two
(12-13 b20 ESS14 and 12-30 b11 ESS14 at R2). The actor is exactly at the worst
quote in 3 of the 10 (12-13 b11 ESS14 charge, 12-13 b2 ESS27 discharge,
12-30 b11 ESS14 discharge); the 12-30 b11 ESS27 discharge actor is 18 CNY/MWh
below the worst end, and the remaining groups sit in the middle or at the
truthful end. The state-dependent sign is
important: a high quote is not universally wrong; it is wrong in the states
where the fleet's truthful offer already clears the block.

## Twin critics

| statistic | value |
|---|---:|
| mean abs(Q1 − Q2) | 0.0427 |
| median abs(Q1 − Q2) | 0.0073 |
| 95th percentile abs(Q1 − Q2) | 0.200 |
| groups with at least one twin rank disagreement | 85/88 |
| material groups with at least one twin rank disagreement | 9/10 |

The twins agree on the level far more than on the ordering: the median gap is
small, but in almost every group at least one pair of quotes is ordered
differently by Q1 and Q2. The target uses `min(Q1,Q2)`, so this disagreement
is exactly the quantity that can flip an ordering.

## Absolute error

| slice | mean | median | MAE | RMSE | 5th | 95th |
|---|---:|---:|---:|---:|---:|---:|
| all vs training | −258.4 | −160.4 | 310.4 | 427.2 | −855.7 | 150.4 |
| all vs settlement | −289.3 | −207.3 | 298.1 | 402.6 | −816.7 | 32.9 |
| material vs training | −79.4 | +24.6 | 159.5 | 289.7 | −809.3 | 182.2 |
| material vs settlement | −208.3 | −118.5 | 208.3 | 298.8 | −762.7 | −60.9 |

The units are consistent (CNY after ×1000), but the critic is not a usable
absolute predictor: a MAE of 310 CNY on a day whose material losses are
10–130 CNY per block is larger than the effect it is supposed to rank.

## Answers

1. **Is the comparison on the same scale?** Yes for Q vs the discounted B+C2
   training return: both are reward units, and ×1000 converts to CNY. It is
   **not** the same object as the undiscounted settlement or the B1
   counterfactual; those must be reported separately and are.
2. **Does the critic predict the return?** No. Pooled correlation is −0.566;
   within-group mean Spearman is −0.008; material groups are mostly
   anti-ranked. The critic fits its training batch, not this return.
3. **Is it only an offset?** No. R2 has a large negative offset, but the sign
   reversal by winter/control and by ESS survives a level shift, and the
   within-group ordering is wrong in the material states.
4. **Twin critics:** they disagree on ordering in 9 of 10 material groups, so
   the `min` target is not a stable value estimate there.

## Boundary

The rollout uses the current deterministic policy after the audited action and
the current critic; it is not a target-actor evaluation. The material
definition is the phase-15 one (ΔG spread > 10 CNY).

The calibration grid used blocks 2/11/20. The winter-loss reconstruction puts
the worst R10 blocks at 12-30 b10 and 12-31 b17, which this grid does not
contain. The 10 material groups therefore under-sample the states that lose the
most; the winter 15-minute audit covers those blocks directly. A future
calibration must use the winter-loss blocks, not only the phase-15 grid.

No H2, no legacy test, no training, no checkpoint write.
