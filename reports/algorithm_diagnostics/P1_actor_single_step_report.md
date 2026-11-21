# P1-B — actor-only update diagnostic

Protocol `DIAGNOSTIC_PLAN_V1`, phase P1-B (read-only).
Written 2026-11-08. Interpreter
`C:\Users\26036\.conda\envs\energy_env\python.exe` (CPython 3.10.20),
torch `2.11.0+cpu`, numpy `1.24.3`, gurobipy `13.0.1`.

Collector: `src/diagnose_actor_update.py` (new, unpinned). Artifacts:

- `reports/algorithm_diagnostics/P1_evidence.json` — provenance, guards,
  per-replicate step logs, per-day and per-unit summaries;
- `reports/algorithm_diagnostics/P1_actor_update_results.csv` — 256 rows, one
  per (seed, arm, replicate, pair), 39 columns including the 25-point SOC path
  of each unit;
- `reports/algorithm_diagnostics/P1_run.log` — progress and the final verdict;
- `P1_mapping_regression.md` — the P1-A half of the phase.

Command:

```
PYTHONPATH=src "C:/Users/26036/.conda/envs/energy_env/python.exe" \
  src/diagnose_actor_update.py
```

Result: `verdict: PASS_P1_ACTOR_DIAGNOSTIC`, exit 0, 2,197 s wall clock.
`overall_classification: INCONCLUSIVE` — the two checkpoints disagree, which
the pre-registered rule maps to `INCONCLUSIVE`.

## Method

| item | value |
|---|---|
| checkpoints | seed 11 best `runs/FORMAL_1b47e0ed_seed11_best` step 1,272 / episode 53 / τ 2.5 / buffer 1,272; seed 23 best `runs/FORMAL_1b47e0ed_seed23_best` step 5,088 / episode 212 / τ 1.0 / buffer 5,088 |
| days | the 8 pre-registered Train pairs, 3 summer / 3 autumn / 2 winter, season-stratified before measurement |
| arms | 0 (baseline), 1, 5 and 10 actor updates |
| replicates | 5 per arm; only the sampled batch varies (`seed = 10000·seed + 100·arm + rep`) |
| batch | 128 transitions, one `DAReplayBuffer.sample` per replicate, reused for that replicate's 1/5/10 steps |
| update | the actor branch of `MATD3BC2Gate.update` only: each of the four actors forwards its own observation, the actor-loss map replaces that agent's action in the batch, twin-critic `-min(Q1,Q2).mean()`, clip 1.0, Adam step |
| not run | no critic step, no target soft update, no `remember()`, no `save()`, no optimizer reset |
| rollout | full 24-block day with `select_action(train=False)`, B1 cache settlement, terminal SOC pinned at 0.5, identical to `train_da_gate.eval_day` |
| restore | actors and Adam state deep-copied before the arms and reloaded before every arm and every replicate |

### Provenance

| field | seed 11 | seed 23 |
|---|---|---|
| step / episode | 1,272 / 53 | 5,088 / 212 |
| `actor_loss_map` / `quote_map` | `linear_core_v1` / `linear_core_v1` | same |
| `obs_spec` / `action_spec` | `v6_12d` / `storage_quote_v3` | same |
| gate temperature | 2.5 | 1.0 |
| replay buffer length | 1,272 | 5,088 |
| rollout equals `eval_day` | **PASS**, max abs diff 0.0 | **PASS**, max abs diff 0.0 |

The rollout comparison covers `G_policy`, `G_b1`, `delta`, charge, discharge,
cycles, SOC extrema, gate mean probability, gate open frequency and both
saturation counters; every one agreed exactly.

### Guards

| guard | seed 11 | seed 23 |
|---|---|---|
| critic and critic-target parameters bitwise equal after every arm | **PASS** | **PASS** |
| actor-target parameters bitwise equal after every arm | **PASS** | **PASS** |
| replay length before / after | 1,272 / 1,272 | 5,088 / 5,088 |
| checkpoint tree sha256 before / after | unchanged | unchanged |
| rollout matches `eval_day` | **PASS** | **PASS** |

`checkpoint_trees_unchanged: true` over both checkpoint trees. The only writes
are the CSV, the JSON and the log under `reports/algorithm_diagnostics/`.

## Results

All values are CNY per day, higher is better, relative to arm 0 (the restored
policy on the same day). A negative value means the update made the fleet earn
less than the un-updated checkpoint.

### Fleet ΔG

| seed | arm | fleet mean | 95% CI (registered: day × replicate) | days better | n |
|---|---|---|---|---|---|
| 11 | 1 | **−0.218** | [−0.397, −0.070] | 1 / 8 | 40 |
| 11 | 5 | **−5.130** | [−8.153, −2.431] | 4 / 8 | 40 |
| 11 | 10 | **−6.111** | [−9.501, −3.055] | 2 / 8 | 40 |
| 23 | 1 | +0.375 | [−0.072, +0.894] | 5 / 8 | 40 |
| 23 | 5 | +2.115 | [+0.083, +4.055] | 5 / 8 | 40 |
| 23 | 10 | +1.970 | [−1.351, +5.245] | 4 / 8 | 40 |

Bootstrap: percentile, 10,000 resamples, unit day × replicate, as
pre-registered in `thresholds.json`.

### Per-day means

The fleet mean is not spread across the days. Most days are exactly zero to
machine precision, and the sign of the effect is not uniform where it is
non-zero.

Seed 11:

| pair | season | arm 1 | arm 5 | arm 10 |
|---|---|---|---|---|
| 2022-07-23 \| 2025-07-13 | summer | −0.00 | 0.00 | −0.00 |
| 2022-08-14 \| 2025-08-17 | summer | −0.00 | 0.00 | 0.00 |
| 2022-08-17 \| 2025-08-19 | summer | −0.00 | −0.00 | −0.00 |
| 2022-10-13 \| 2025-10-13 | autumn | −0.13 | −0.07 | −0.35 |
| 2022-10-15 \| 2025-10-12 | autumn | 0.00 | 0.00 | −0.00 |
| 2022-10-17 \| 2025-10-17 | autumn | −0.00 | 0.00 | 0.00 |
| 2022-12-09 \| 2025-12-09 | winter | −1.57 | −15.27 | −25.79 |
| 2022-12-10 \| 2025-12-07 | winter | −0.04 | −25.70 | −22.74 |

Seed 23:

| pair | season | arm 1 | arm 5 | arm 10 |
|---|---|---|---|---|
| 2022-07-23 \| 2025-07-13 | summer | +1.83 | +12.68 | +12.68 |
| 2022-08-14 \| 2025-08-17 | summer | −0.00 | −0.00 | −0.00 |
| 2022-08-17 \| 2025-08-19 | summer | 0.00 | −0.00 | −0.00 |
| 2022-10-13 \| 2025-10-13 | autumn | +0.23 | +6.71 | +19.12 |
| 2022-10-15 \| 2025-10-12 | autumn | 0.00 | 0.00 | 0.00 |
| 2022-10-17 \| 2025-10-17 | autumn | −0.00 | 0.00 | 0.00 |
| 2022-12-09 \| 2025-12-09 | winter | −0.17 | −9.55 | −15.92 |
| 2022-12-10 \| 2025-12-07 | winter | +1.12 | +7.08 | −0.12 |

At arm 0 the fleet earns 0.000 CNY on the days marked zero: the clearing gives
those units no throughput at the restored policy (`charge_mwh = discharge_mwh =
0`), so a quote change cannot move them. Four of the eight days are inert for
seed 11 and three of eight for seed 23. The effective sample is therefore 4–5
informative days, not 40 independent observations.

Clustering the bootstrap by day (a post-hoc sensitivity, **not** the registered
interval) widens the intervals:

| seed | arm | mean | 95% CI, days resampled |
|---|---|---|---|
| 11 | 1 | −0.218 | [−0.609, −0.005] |
| 11 | 5 | −5.130 | [−12.158, −0.009] |
| 11 | 10 | −6.111 | [−14.640, −0.044] |
| 23 | 1 | +0.375 | [−0.021, +0.882] |
| 23 | 5 | +2.115 | [−2.351, +6.478] |
| 23 | 10 | +1.970 | [−4.386, +8.738] |

Under day clustering, seed 23's positive arm-5 effect is no longer separated
from zero, and seed 11's negative effects remain negative with intervals that
reach close to zero. The classification below uses the registered interval;
this sensitivity is reported because the registration treats five replicates of
the same day as independent when they share that day's load, price and clearing.

### Per-ESS ΔG

| seed | arm | ESS9 | ESS14 | ESS17 | ESS27 |
|---|---|---|---|---|---|
| 11 | 1 | −0.00 | −0.17 | −0.00 | −0.04 |
| 11 | 5 | −2.76 | −1.15 | +0.03 | −1.26 |
| 11 | 10 | −2.35 | −2.12 | +0.04 | −1.68 |
| 23 | 1 | +0.04 | −0.01 | −0.00 | +0.35 |
| 23 | 5 | −1.16 | +0.97 | −0.14 | +2.45 |
| 23 | 10 | −1.62 | +3.47 | −1.91 | +2.03 |

Per-unit 95% intervals are in `P1_evidence.json` under
`seed{11,23}.summary.<arm>.per_unit_ci`. On seed 11 every unit except ESS17 is
negative at arms 5 and 10. On seed 23 the units move in opposite directions:
ESS14 and ESS27 gain while ESS9 and ESS17 lose.

### Batch Q

`q_eval_before` / `q_eval_after` are `min(Q1,Q2)` at the replicate's own
sampled batch states, with the critic and the recorded joint actions held
fixed and only this actor's action replaced. The critic is in eval mode, so
the number is deterministic; the batch differs across replicates, so only the
within-replicate change is interpretable.

| seed | arm | mean ΔQ | replicates positive | ΔQ / \|Q\| |
|---|---|---|---|---|
| 11 | 1 | −0.00022 | 2 / 5 | −0.09% |
| 11 | 5 | +0.00107 | 5 / 5 | +0.44% |
| 11 | 10 | +0.00114 | 4 / 5 | +0.46% |
| 23 | 1 | +0.00009 | 4 / 5 | +0.14% |
| 23 | 5 | +0.00004 | 3 / 5 | +0.07% |
| 23 | 10 | +0.00025 | 5 / 5 | +0.40% |

Q is in reward units (fixed scale 1,000), so +0.00114 corresponds to about
1.4 CNY/day through the return identity `γ^23 · delta / 1000`, since
`γ^23 = 0.99^23 = 0.794`. The actor moves
the batch Q slightly upward on both checkpoints; on seed 11 the realised
payoff moves down by 6.1 CNY/day at arm 10. The Q change stays below the 0.5%
level the pre-registered direction probe treats as a signal, so this is a lead
for P2 and not a finding of this phase.

### Gradients

Mean over every step of every replicate:

| seed | arm | ‖∇actor‖ | ‖∇gate‖ | ‖∇price‖ | cos(gate, price) |
|---|---|---|---|---|---|
| 11 | 1 | 0.0427 | 2.4e-04 | 2.19e-03 | +0.755 |
| 11 | 5 | 0.0406 | 2.4e-04 | 2.28e-03 | +0.795 |
| 11 | 10 | 0.0392 | 2.5e-04 | 2.26e-03 | +0.767 |
| 23 | 1 | 0.0129 | 1.9e-05 | 4.83e-04 | +0.806 |
| 23 | 5 | 0.0084 | 2.0e-05 | 4.53e-04 | +0.816 |
| 23 | 10 | 0.0084 | 1.8e-05 | 4.62e-04 | +0.851 |

The gate-logit component is one to two orders of magnitude smaller than the
price-logit component and the two point in the same direction. The actor is
therefore moving prices, not gate modes: the straight-through signal to a gate
logit is `p(1-p)/τ` times the quote level, and at τ 2.5 and 1.0 that product is
small compared with the direct price gradient. Gradient clipping at 1.0 never
activates.

### Gate probability, quotes, saturation, SOC

| seed | arm | mean gate prob | charge quote | discharge quote | lower-edge legs/day | upper-edge legs/day |
|---|---|---|---|---|---|---|
| 11 | 0 | 0.939 | 30.37 | 152.00 | 22.8 | 6.9 |
| 11 | 1 | 0.939 | 30.60 | 152.80 | 22.8 | 7.4 |
| 11 | 5 | 0.943 | 32.03 | 158.15 | 22.7 | 10.8 |
| 11 | 10 | 0.946 | 32.75 | 164.06 | 22.3 | 19.1 |
| 23 | 0 | 0.875 | 1.76 | 80.31 | 48.5 | 0.0 |
| 23 | 1 | 0.876 | 1.78 | 81.01 | 48.4 | 0.0 |
| 23 | 5 | 0.875 | 2.12 | 85.85 | 48.1 | 0.0 |
| 23 | 10 | 0.876 | 2.67 | 92.03 | 48.0 | 0.0 |

Quotes are the mean declared price over open legs only; a closed leg is
reported as 0. Over ten update steps the open discharge quote moves +7.9%
(seed 11) and +14.6% (seed 23), and the number of leg-decisions pinned at the
upper edge rises on seed 11 from 6.9 to 19.1 per day out of 192. Gate
probabilities barely move, consistent with the gradient table.

| seed | arm | charge MWh/day | discharge MWh/day | cycles/day | SOC min | SOC max | terminal SOC error |
|---|---|---|---|---|---|---|---|
| 11 | 0 | 0.938 | 0.794 | 0.541 | 0.16 | 0.90 | 0.0 |
| 11 | 10 | 0.890 | 0.753 | 0.514 | 0.16 | 0.90 | 0.0 |
| 23 | 0 | 1.384 | 1.172 | 0.799 | 0.10 | 0.90 | 0.0 |
| 23 | 10 | 1.312 | 1.111 | 0.757 | 0.10 | 0.90 | 0.0 |

SOC extrema are unchanged by the updates and every day ends at 0.5 to machine
precision. Throughput falls by about 5% on both checkpoints while the payoff
change has opposite signs, so the two are not linked by a common mechanism.
The 25-point SOC path of every unit and every day is in the CSV.

## Classification

Applying the pre-registered rule to the registered intervals:

| seed | class | rule that fired |
|---|---|---|
| 11 | **CUMULATIVE_DEGRADATION** | means are non-increasing over arms 1, 5, 10 (−0.218, −5.130, −6.111) and arm 10 ≤ −1.0 with its 95% CI excluding 0 ([−9.501, −3.055]) |
| 23 | **INCONCLUSIVE** | arm 1 CI includes 0; arm 5 is positive with a CI excluding 0 but arm 10 is not ≤ −1.0 (no short-term/long-term split) and arm 1 is not significantly positive (no consistent improvement); arm 10 CI includes 0; not every arm CI includes 0 |
| overall | **INCONCLUSIVE** | the two checkpoints disagree; `thresholds.json` maps that to `INCONCLUSIVE` |

Seed 11's arm-1 interval excludes zero but its magnitude is below the 1.0
CNY/day effect floor, so the immediate-degradation rule does not fire; the
cumulative rule does.

## What this establishes

**PASS** — over one to ten actor-only updates, with the critic, the target
networks, the replay buffer and the market conditions fixed, the fleet's
full-day payoff on eight Train days:

- falls monotonically on the seed 11 best checkpoint, significantly at arms 5
  and 10;
- does not move significantly on the seed 23 best checkpoint;
- is concentrated on a few days and a few units, with several days entirely
  inert and with unit-level effects of opposite sign on seed 23.

**PASS** — the actor step does ascend the batch Q slightly on both checkpoints
(+0.1% to +0.5%), and the gradient is price-dominated; the gate logits are
essentially stationary. No guard failed and no original artifact changed.

## What this does not establish

- **NOT_TESTED** — the auxiliary direction probe (quote displacements 5/10/20/
  40 CNY/MWh). It was not run: it needs a normalised gradient direction and a
  controlled displacement, and the same Q-versus-payoff question is answered
  over a wider and more interpretable range by the pre-registered P2
  one-at-a-time grid. This is a deviation from the plan and is recorded as
  such; nothing in P3 may cite a direction-probe result.
- **NOT_TESTED** — that ten updates on one reused batch represent the training
  trajectory. They do not: each arm resamples one batch and applies the same
  actor loss to it repeatedly, which is more aggressive than a training step.
- **NOT_TESTED** — the voltage and line cells per arm. The rollout computed them
  and the rollout matched `eval_day`, but the column was not persisted, so no
  per-arm statement is available.
- **NOT_TESTED** — any root cause. One checkpoint degrades and one does not;
  the two differ in gate temperature (2.5 vs 1.0), in buffer size (1,272 vs
  5,088) and in policy, so the disagreement cannot be attributed to a single
  variable. P1 alone is below the two-line threshold P3 requires.
- **INFERRED** — nothing; every value above was measured in this run.

## Deviations and incidents

1. **Arm 0 added.** `thresholds.json` pre-registers arms 1/5/10; arm 0 is the
   measured baseline and enters no classification rule.
2. **`CONSISTENT_IMPROVEMENT` added** to the class vocabulary at the user's
   direction; no arm met it.
3. **Direction probe deferred**, as stated above.
4. **Day-clustered interval added post hoc** as a sensitivity only; the
   classification uses the registered day × replicate interval.
5. **A first attempt was killed by system memory pressure** before writing any
   artifact; no measurement from it is used. The collector was then changed to
   append each replicate's CSV rows and dump the evidence after every
   replicate, and the phase was rerun from the beginning.

## Not tested / open

- direction probe (see above);
- per-arm voltage and line cells;
- Test and Holiday blocks, untouched by design; H2 sealed;
- long-horizon behaviour beyond ten updates.

## Exit

**`PASS_P1_ACTOR_DIAGNOSTIC`** — all guards passed, all 256 rollouts and both
rollout-equivalence checks are recorded, and no frozen artifact changed. The
scientific classification is **`INCONCLUSIVE`**: seed 11 degrades cumulatively,
seed 23 does not, and the phase does not by itself identify a root cause.

P2 is not started. Stopping for acceptance.
