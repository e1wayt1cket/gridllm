# Phase 20 diagnostic decision

Read-only diagnosis of the D+A checkpoints and the corrected actor loss. No
training, no checkpoint change, no H2 and no legacy test. Evidence:
`phase20_oracle_surrogate.md`, `phase20_bellman_twin_gap.md`,
`phase20_own_action_sensitivity.md`, `phase20_action_routing.md`, and their
JSON/CSV files.

## Component verdicts

| component | verdict | category |
|---|---|---|
| actor-loss quote map | v3 map now used; update == execution under the same gate draw | **code error, fixed and tested** |
| update-path gate temperature | uses `current_gate_temperature()`; no fixed 1.0 | **code error, fixed** |
| critic ranking on covered states | Q-top equals return-top in 6/15 informative groups (best), 1/15 (final); median Spearman +0.238 / −0.143 | **estimation / calibration** |
| twin gap | 23.8 CNY median (best), 6.3 CNY (final); larger than the material spread only in best | **critic training design** |
| target smoothing | 4.2 CNY (best), 0.6 CNY (final) median effect; rank correlation 0.88–0.94 | **not the explanation** |
| ESS14 own-action collapse | does not reproduce in eval mode on 9 states; final median own-share 0.273, range 0.115–0.473 | **measurement artifact in Phase 19** |
| action routing / indices | ESS order, leg order, widths, target and critic routes all match v3; v2 calls in update = 0 | **clean** |
| first-order actor step | moves <1 CNY at the named states against 100+ CNY gaps; oracle slope often flat | **consequence of non-local value geometry** |

## What is a code error and what is not

The only unambiguous implementation error was the actor-loss map. It is fixed
by one override point, tested against the execution path, and recorded in the
checkpoint metadata. The same fix also removes the fixed evaluation gate
temperature from the update.

The critic gap is **not** a target-arithmetic error. The target closes on the
code's own formula, and the smoothing effect is too small to explain the
distance between the twins. The twin gap is a learned-parameter disagreement,
large in the best checkpoint and smaller in the final checkpoint. The oracle
shows that the critic's local direction is wrong in 9/15 informative groups
for both checkpoints, so the value surface is not usable as a first-order
descent direction even where the action region is covered.

## What this rules out

- A target-noise or twin-aggregation change (candidate 002): smoothing is not
  the source of the gap, and the twin gap does not persistently dominate the
  action-return spread.
- A sampler or credit-assignment change (candidate 003): the ESS14 own-action
  collapse does not reproduce once the critic is evaluated without dropout and
  the state set is expanded. The earlier 0.004–0.022 numbers are a
  dropout-dependent local measurement, not a stable credit signal.
- A network-structure change: no evidence supports it, and the task book
  forbids it without a localized, reproducible defect.

## What remains open

The critic's ranking/calibration on material states is weak. It is now
characterized (twin gap, TD residual, oracle direction, own/cross sensitivity,
action coverage from Phase 19) but not repaired. No minimal, evidence-supported
fix was found. This is a **residual model risk**, not a deterministic code
defect. The formal run should be judged by whether the larger budget changes
the critic, not by the Phase 19 checkpoint's ranking.

## Split boundary

12-13 and 12-30 are validation days under `near/split.json`, the split used by
the D+A checkpoints, and training days under `common_split`, the formal split.
They are offline diagnostic states only. 12-31 is validation under both. No
diagnostic state contributes a label to any formal run.

## Decision

Proceed with candidate 001 (the map fix). Do not open candidate 002 or 003.
Record the critic ranking as an explained residual risk in the readiness
decision. No training budget or data change is made.

## Boundary

No training, checkpoint write, replay addition, H2 read or legacy test run
occurred. The oracle is a piecewise-linear proxy; the replay residual is
off-policy; the own-action measurement is local.
