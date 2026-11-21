# Phase 20-D: own-action versus cross-action sensitivity

Read-only. No training, no checkpoint write, no H2 and no legacy test. Data:
`phase20_own_action_sensitivity.{json,csv}`. Code:
`src/audit_phase20_own_action_sensitivity.py`.

## Method

At a fixed joint state, perturb one unit's quote and differentiate the target
critic's `min(Q1,Q2)` with respect to that quote. The absolute sensitivity is
scaled Q per CNY/MWh; the box-normalized sensitivity multiplies it by the leg's
box width. Own-share is the squared own normalized sensitivity divided by the
sum over all four units and both legs. Nine states are used: the four Phase 19
training states and the five named validation cases. The critic is evaluated
without dropout so best and final use the same mask.

## Results

| checkpoint | unit | median own-share | range | median own/cross ratio |
|---|---|---:|---:|---:|
| D+A best | ESS14 | 0.157 | 0.052–0.202 | 0.431 |
| D+A best | ESS27 | 0.374 | 0.073–0.374 | 0.773 |
| D+A final | ESS14 | **0.273** | **0.115–0.473** | 0.613 |
| D+A final | ESS27 | 0.237 | 0.049–0.266 | 0.557 |

Per ESS14 state (final checkpoint): 0.139 (12-24 b8), 0.115 (12-29 b14),
0.244 (12-30 b10), 0.273 (12-30 b11), 0.313 (12-13 b3), 0.473 (12-31 b17).

## The Phase 19 collapse does not reproduce

Phase 19 reported ESS14's final own-action share at 0.004–0.022 and treated it
as a credit-assignment asymmetry. The expanded state set and the deterministic
critic evaluation do **not** reproduce it: every ESS14 state is above 0.11,
and the final checkpoint's median own-share is **higher** than the best
checkpoint's, not lower. The likely cause of the difference is the online
critic's dropout: the Phase 19 measurement ran the critic in training mode,
while this audit evaluates the same local derivative without dropout. A
dropout-dependent local share is not a stable credit signal.

## Consequence

No sampler or credit-assignment candidate is opened. A change justified only by
the earlier 0.004–0.022 numbers would be a change justified by a measurement
artifact. The remaining ESS14 weakness is visible in the action and oracle
results, not in a reproducible own-action collapse.

## Boundary

The measurement is local, at the executed action, and is not a computed
contribution. The four training states use the `near/split.json` role of the
D+A run; the named cases are validation there and training days under
`common_split`. They are offline diagnosis only.
