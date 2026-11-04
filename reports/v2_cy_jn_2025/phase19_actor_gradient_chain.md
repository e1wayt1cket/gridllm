# Phase 19-B: actor gradient chain

Read-only. No training, no checkpoint write, no H2 and no legacy test. Data:
`phase19_actor_gradient_chain.json` and `phase19_actor_gradient_chain.csv`.
Code: `src/audit_phase19_actor_gradient_chain.py`.

## The two named states

| pair | block | unit | NEAR role |
|---|---:|---|---|
| 2022-12-30 \| 2025-12-31 | 10 | ESS14 | validation, offline case only |
| 2022-12-13 \| 2025-12-11 | 12 | ESS27 | validation, offline case only |

Two NEAR training states (`2022-12-24` b8 ESS14, `2022-12-29` b14 ESS27) are
added so the result is not a property of the validation days. Both checkpoints
are audited: D+A best (step 1272) and D+A final (step 3816).

## A concrete implementation mismatch: the actor loss uses the v2 map

`DAGateTrainer` executes and targets through `da_quote_map.forward_policy_v3`
(`linear_core_v1`), but it inherits `MATD3BC2Gate.update`. That update maps the
actor output with `truthful_gate_actor_adapter.forward_policy`, which is the
**v2 sigmoid** branch:

```
quote = gate * high * sigmoid(price_logit)            # update path (v2)
quote = gate * high * g(price_logit)                  # execution, target (v3)
```

The replay stores the v3 executed quote, the critic is trained on that quote,
and the target actor produces v3 quotes. Only the actor loss is differentiated
through the v2 map. The phase-16 ST-gradient audit documented the sigmoid
price branch; the D+A integration table says "v3 map; derivative audited". The
two statements cannot both describe this code: the actor gradient is taken
through a map the actor does not use.

### Size of the mismatch

At the evaluated quote the two maps differ by 2.7–10.2 CNY/MWh on these
states. More importantly, the derivative differs:

| state | v3 dq/dz charge, discharge | v2 dq/dz charge, discharge |
|---|---:|---:|
| 12-30 b10 ESS14 (best) | 29.4, 1.65 | 16.9, 17.2 |
| 12-13 b12 ESS27 (best) | 16.0, 43.6 | 15.7, 5.3 |

The logit-gradient comparison, same state and same gate draw:

| checkpoint / state | cos(v2 path, v3 path) | norm ratio v2/v3 | sign flip |
|---|---:|---:|---|
| best 12-30 b10 ESS14 | +0.986 | 1.06 | no |
| best 12-13 b12 ESS27 | +0.916 | 0.87 | no |
| best 12-24 b8 ESS14 (train) | +0.999 | 0.58 | no |
| best 12-29 b14 ESS27 (train) | **−0.892** | 0.54 | **yes, price legs** |
| final 12-30 b10 ESS14 | +0.997 | 0.31 | no |
| final 12-13 b12 ESS27 | **−0.973** | 0.29 | **yes, price legs** |
| final 12-24 b8 ESS14 (train) | +0.999 | 0.77 | no |
| final 12-29 b14 ESS27 (train) | +0.992 | 1.48 | no |

The direction is usually close, but it is not the direction of the executed
map; at ESS27 states it can reverse. A one-step update through the wrong map is
therefore not guaranteed to be a small perturbation.

## The chain is connected, unclipped, and the optimizer is active

At both named states the path `actor logit -> gate -> quote -> critic -> loss`
is connected: every price logit has a nonzero gradient, and the gate logits
receive the straight-through surrogate. The recorded values at 12-30 b10 ESS14
(D+A best):

| quantity | value |
|---|---|
| gate probability (charge, discharge) | 0.871, 0.939 |
| hard gate sampled in training | 1, 1 |
| ST surrogate p(1−p) | 0.113, 0.057 |
| logit gradient (v2 path) | 7.2e-4, 2.0e-3, 5.6e-3, 1.9e-2 |
| total gradient norm | 0.0201 |
| clip threshold / coefficient | 1.0 / 1.0 (no clipping) |
| Adam steps on the actor | present, all layers |

At 12-13 b12 ESS27 (D+A best) the same checks pass: gradient norm 0.0179, no
clipping, Adam state present. Gradient clipping is not blocking these updates.
The `policy_delay=2` gate is not implicated: the checkpoint counters show
thousands of actor updates, and the audit replays the exact actor-loss branch.

At D+A final one gate is closed in two states (charge probability 0.29 / 0.40,
deterministic gate 0). The closed price leg then has exactly zero actor
gradient, which is expected from `quote = gate * ...`; the gate logit still
receives the surrogate. This is a hard-gate effect, not a broken tensor path.

## Critic gradient with respect to the action

At the actor's executed quote, `dQmin/dquote` (scaled Q per CNY/MWh):

| checkpoint / state | charge | discharge | Q-top quote | actor quote | best-return quote |
|---|---:|---:|---:|---:|---:|
| best 12-30 b10 ESS14 | +2.1e-4 | +2.3e-4 | 0.0 | 136.8 | 0.0 |
| best 12-13 b12 ESS27 | −1.6e-4 | +2.1e-4 | 0.0 | 81.2 | 0.0 |
| final 12-30 b10 ESS14 | +8.4e-5 | +1.0e-4 | 150.0 | 278.7 | 95.0 / 0.0 |
| final 12-13 b12 ESS27 | −7.4e-5 | +1.4e-4 | 150.0 | 300.0 | 0.0 |

At 12-30 b10 ESS14 (D+A best) the critic's local slope at 136.8 is **positive**,
i.e. it points to a higher quote, while the within-state Q-top is 0 and the
matched gain from 0 is 138.9 CNY. The actor is following a local gradient that
is inconsistent with the critic's own value ordering; it has no way to cross
the non-concave gap to 0 by a first-order step.

## Virtual one-step update

A deep copy of the actor and its Adam state is stepped once; the original
checkpoint is untouched. The executed quote is then recomputed with the v3
evaluation map. Quoted in CNY/MWh.

| checkpoint / state | charge before → after | discharge before → after | path |
|---|---:|---:|---|
| best 12-30 b10 ESS14 | 18.96 → 18.88 | 136.82 → 135.94 | code (v2) |
| best 12-30 b10 ESS14 | 18.96 → 18.98 | 136.82 → 135.95 | intended (v3) |
| best 12-13 b12 ESS27 | 37.46 → 37.28 | 81.23 → 80.78 | code (v2) |
| best 12-13 b12 ESS27 | 37.46 → 37.09 | 81.23 → 80.90 | intended (v3) |
| final 12-30 b10 ESS14 | 0.00 → 0.00 | 278.68 → 276.34 | code (v2) |
| final 12-30 b10 ESS14 | 0.00 → 0.00 | 278.68 → 275.59 | intended (v3) |
| final 12-13 b12 ESS27 | 76.01 → 77.65 | 300.00 → 300.00 | code (v2) |
| final 12-13 b12 ESS27 | 76.01 → 75.10 | 300.00 → 300.00 | intended (v3) |

The per-step movement is small. At 12-30 b10 ESS14 the correct-map step lowers
the discharge quote by less than 1 CNY/MWh, against a 136.8 CNY/MWh gap to the
Q-top. A local, gradient-limited step of this size cannot repair the quote at
the sampling rate used in the pilot; the fix is not "run the same training
longer".

## Adam state at the checkpoint

D+A best, 12-30 b10 ESS14 (final-layer rows):

| output row | weight row norm | Adam exp_avg norm | sqrt(exp_avg_sq) mean |
|---|---:|---:|---:|
| gate charge | 0.506 | 7.3e-4 | 9.9e-4 |
| gate discharge | 0.552 | 1.3e-3 | 1.4e-3 |
| price charge | 0.502 | 3.2e-3 | 2.7e-3 |
| price discharge | 0.475 | 1.6e-3 | 3.5e-3 |

D+A final, same state:

| output row | weight row norm | Adam exp_avg norm | sqrt(exp_avg_sq) mean |
|---|---:|---:|---:|
| gate charge | 0.573 | 7.2e-5 | 5.3e-4 |
| gate discharge | 0.572 | 1.0e-3 | 7.8e-4 |
| price charge | 0.550 | 1.2e-4 | 1.5e-3 |
| price discharge | 0.479 | 8.2e-3 | 2.2e-3 |

The optimizer has accumulated momentum, and the price-discharge row is the
largest in both checkpoints. The optimizer is not frozen.

## Finding

The actor chain is mechanically sound: connected, unclipped, Adam active, no
NaN/Inf, and a virtual step executes. Two defects remain:

1. **Implementation error:** the training objective is differentiated through
   the v2 sigmoid map while execution and target use v3. The gradient is not
   the gradient of the executed action, and the direction can reverse at ESS27
   states.
2. **Actor/critic geometry:** the actor takes a small first-order step and, at
   12-30 b10 ESS14, follows a positive local slope away from the critic's own
   better value basin at 0.

Neither of these is a gate defect. Both are reachable from the same actor-loss
branch, which is what Phase 20 must correct or re-specify.

## Boundary

The audit replays one state at a time with the deterministic evaluation gate;
it is not a training episode. Gate draws in the virtual update are seeded for
reproducibility; the single-seed result describes the path, not the sampling
distribution. No checkpoint was written or modified.
