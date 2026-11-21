# D2 stratified joint state-action replay (P0)

Implemented in `src/da_replay.py`. Data: `da_gate_sampling_probabilities.csv`,
`da_gate_buffer_coverage.json`, `da_gate_sampler_critic_loss.json`,
`da_gate_material_calibration.{csv,json}`.

## Joint cell

Each transition gets one cell from features available at the decision:

| dimension | bands |
|---|---|
| block | 0–7, 8–15, 16–23 |
| price | `avg_lmp_hist_norm` <0.5, 0.5–1.0, ≥1.0 |
| load | mean `load_feat` <0.5 / ≥0.5 |
| SOC | mean SOC <0.5 / ≥0.5 |
| discharge quote | max quote/300 <0.33, 0.33–0.66, ≥0.66 |
| gate | no open discharge gate / at least one |
| reward magnitude | max |scaled reward| >0.02 / not |

That is 432 nominal cells; the pilot observed 124 non-empty cells with counts
1–215 (median 14). The cell is built from OBS_V6 and the executed action only;
no future price or LMP enters it.

## Sampling law

```
p_i = 0.5 / N + 0.5 * p_strat,i
p_strat,cell ∝ count_cell ** 0.5
p_strat,i <= 5 / N   (cap)
```

The mixture is the declared training distribution. The critic target is the
conditional Q, so the mixture changes the loss weighting rather than the
estimand; no importance weight is applied to the loss. The uniform-ratio
`(1/N)/p_i` is recorded for audit, not used as a weight.

Pilot quantities (final buffer, 3,816 transitions):

| quantity | value |
|---|---:|
| non-empty cells | 124 |
| cell count min / median / max | 1 / 14 / 215 |
| per-transition p min / max | 1.92e-4 / 7.90e-4 |
| uniform-ratio min / max | 0.332 / 1.368 |
| effective sample size | 3,453 |
| ESS fraction | 0.905 |

Per-episode sampler logs are in `da_gate_sampler_stats.jsonl`; the per-transition
cell, count and probability are in `da_gate_sampling_probabilities.csv`.

## Coverage

| buffer | transitions | non-empty cells | ESS14 high ≥0.86h | ESS27 high ≥0.86h |
|---|---:|---:|---:|---:|
| old R10 | 12,720 | 125 | 2,307 (18.1%) | 1,276 (10.0%) |
| new final | 3,816 | 124 | 705 (18.5%) | 252 (6.6%) |

The raw joint-cell count is essentially unchanged; the new buffer is shorter
(159 vs 530 episodes). The exploration adds high quotes at ESS14 but ESS27's
high-quote share is lower than the old buffer. The sampler reweights the rare
cells (ESS fraction 0.905), but it does not create raw coverage by itself.

## Critic loss under the sampling modes

Post-hoc, with the critic, target critic and target actors fixed:

| checkpoint | uniform | mixture | stratified only |
|---|---:|---:|---:|
| best (round 1, 1,272 transitions) | 0.00255 | 0.00297 | 0.00213 |
| final (3,816) | 0.00115 | 0.00102 | 0.00084 |

The stratified-only sample has the lowest batch loss, but the losses are small
and the distributions differ; this is **not** evidence that the critic is
better calibrated. The material-state ranking is the evidence.

## Material-state ranking

The pilot's selected checkpoint was evaluated on the phase-15 material blocks
plus the winter-loss blocks (13 states, 52 state×ESS×leg groups). A group is
material when its actual day ΔG spread across the quote grid exceeds 10 CNY.

| statistic | pilot best |
|---|---:|
| groups / material / flat | 52 / 15 / 31 |
| mean Spearman(Q, ΔG) | +0.109 |
| median Spearman | +0.200 |
| material mean Spearman | +0.128 |
| material groups negative | 5 / 15 |

The old R10 had a mean near 0 and 7 of 10 material groups negative; the state
sets are not identical (the pilot adds the winter-loss blocks), so this is a
direction, not a controlled comparison. The critic ranking is **better on this
material set**: 10 of 15 are non-negative and the mean is positive. The
critic loss comparison is not used as the success criterion.

## What this does and does not show

- The sampler is traceable, the probabilities sum to one, the cap and ESS are
  recorded, and the batch fields match their buffer indices.
- The critic's material-state ordering improved on the evaluated set.
- The actor still chose mid quotes that lose the block (see the pilot report);
  a better critic ranking did not by itself change the selected action.

## Boundary

No validation or H2 data entered the replay. The sampling distribution is
declared; the importance ratio is recorded but not applied. The pilot's
selected checkpoint is round 1.
