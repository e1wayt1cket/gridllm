# Next single-seed pilot protocol (Phase 16)

Protocol only. No training is started in this phase.

## Purpose

Test whether the D + A repair changes the two confirmed failures:

1. the R10 actor sits at 258–299 CNY/MWh in winter blocks where that quote
   produces zero accepted discharge and a 98–128 CNY block loss;
2. the critic anti-ranks the long-horizon quote value in the material states
   (7 of 10 material groups, mean within-group Spearman ≈ 0).

The pilot is exploratory/development only. It does not support a generalisation
claim; H2 and the legacy seen test stay sealed.

## Fixed design

| item | value |
|---|---|
| seed | 11 (same seed as R2/R10, so the comparison is same-seed) |
| training days | the existing `JN-Base-NEAR` train split, 53 days |
| validation days | the existing 11 validation days, used only for model selection and reporting |
| test / H2 / legacy | not read, not run |
| reward | B + C2, unchanged |
| gamma / scale / terminal | 0.99 / 1000 / SOC 0.5 at the day-end window, unchanged |
| action | candidate C quote map, discharge and charge highs 300/150 |
| gate | per leg, ST Bernoulli, init p=0.5, temperature warmup 4→1 over the first 106 episodes, evaluation τ=1 |
| critic | twin, unchanged loss and target; stratified sampler (D2) |
| exploration | D3, epsilon 0.10 on the discharge leg in upper-quintile-price blocks |
| budget | 530 episodes = 10 rounds × 53 days; checkpoint every round |
| checkpoint | full schema v2: actors, targets, critics, critic targets, optimizers, RNG, counters, replay, exposure, reward/config/data hashes |
| run directory | new `runs/DEV_PILOT_D_A_seed11/`; R2/R10 are read-only |
| model selection | validation mean fleet ΔG at each round; select the best round and also report the final round |
| formal claim | none |

If D4 is activated, the auxiliary set is generated from training states only,
labelled `model_generated_counterfactual`, and stored outside the real replay.

## Baselines

- R2 (round 2, step 2544) and R10 (round 10, step 12720) of
  `EXPLORATORY_NEAR_B_C2_GATE_seed11_v2`, read-only.
- B1 terminal-neutral truthful on the same days.
- The four rule arms B2A/B2B are not re-run; their existing benchmark is the
  reference.

## Required reports

For the selected checkpoint and the final checkpoint:

1. **Settlement:** per-day ΔG vs B1, mean/median, number of positive days,
   and the three winter days separately. Do not report the pooled mean without
   the winter/control split.
2. **Gate:** per-gate p(1−p), open frequency, deterministic and sampled joint
   modes on the fixed reference states, permanent-gate flags, and the fraction
   of decisions with p < 0.01 or p > 0.99.
3. **Quote distribution:** per unit/leg mean, quantiles, the share at
   ≥0.86h and ≥0.99h, and the accepted MWh in those bins.
4. **Critic calibration:** long-horizon Q vs discounted B+C2 return on
   training-only held-out states (acceptance) and on validation (reporting
   only); material-group Spearman, twin disagreement, and the B/C2 component
   split.
5. **Winter 15-minute audit:** repeat the same q=0/100/150/258/300 grid on the
   three winter days and the worst blocks; report accepted MWh and payoff per
   period, not a block average.
6. **Provenance:** seed, training dates, data hashes, config hash, checkpoint
   tree hash, sampler and temperature schedule, and the exploration log.

## Stop rule

Do **not** increase the budget if any of the following holds at the selected
checkpoint:

- the deterministic joint mode count on the fixed reference states is ≤2 and
  the sampled modes do not recover (the R1/R10 collapse pattern);
- the material-group mean Spearman is negative, or at least half of the
  material groups remain anti-ranked;
- the actor still sits at ≥0.86h with zero accepted discharge in a high-price
  block where q=0/100 would have cleared;
- the 15-minute audit shows the same q=258–300 → zero acceptance pattern at the
  winter loss blocks.

In those cases the next step is a diagnosis, not a larger run. Only if the
material-group ranking is non-negative and the high-quote region is no longer
the loss source does a budget increase make sense.

## What the pilot must not do

- No validation replay, no H2, no legacy test, no old-test-based selection.
- No change to B, C2, gamma, fixed scale, terminal SOC, clearing engine or
  action bounds.
- No overwrite of R2/R10 or any prior checkpoint.
- No claim that a model-generated counterfactual is independent market data.
- No entropy bonus or REINFORCE switch as part of this pilot.

## Decision after the pilot

The pilot answers one question: does the D + A change move the winter
high-quote loss and the critic's material-state ordering without changing the
economic objective? If yes, the next step is a multi-seed protocol with a fresh
holdout. If no, the finding is that the current state/action/reward design is
still the blocker, and the budget should not be expanded.

## Boundary

This phase does not start the pilot, does not modify the trainer, and does not
read H2 or the legacy test. The protocol is ready to be implemented as a
separate run after review.
