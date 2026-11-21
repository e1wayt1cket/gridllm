# Phase 19-C: cross-ESS gradient conflict

Read-only. No training, no checkpoint write, no H2 and no legacy test. Data:
`phase19_cross_ess_gradient.json` and `phase19_cross_ess_gradient.csv`.
Code: `src/audit_phase19_cross_ess_gradient.py`.

## Structure first

The plan asks whether ESS14's gradient is cancelled by other ESS through
shared parameters. The code does not have that structure:

* `trainer.actors[name]` is a separate `GateActor` per agent. Across the four
  agents, **0 parameter tensors are shared** (all six pairwise intersections are
  empty; `parameter_tensors_shared = 0`).
* `trainer.critics[name]` is also a separate `CentralizedCritic` per agent.
  Each is centralized in its *input* (4×12 observations + 4×2 executed quotes),
  but its parameters are not shared with the other critics.

So a cross-ESS actor-parameter conflict cannot exist in this implementation.
The testable counterpart is the credit each centralized critic gives to each
unit's action, and the conflict between the two legs inside one actor's shared
trunk. Both are measured below on four NEAR training states
(`2022-12-24` b8, `2022-12-29` b14, ESS14 and ESS27).

## Critic credit to each unit's action

`own_action_share` is the squared norm of `dQ_i/da_i` divided by the sum of
squared norms over the four units. It is a local sensitivity, not a computed
contribution, but it answers the credit question directly.

D+A best:

| state | critic ESS14 | critic ESS27 | own share of ESS9 | ESS17 |
|---|---:|---:|---:|---:|
| 12-24 b8 ESS14 | 0.246 | 0.071 | 0.182 | 0.021 |
| 12-24 b8 ESS27 | 0.362 | 0.190 | 0.493 | 0.144 |
| 12-29 b14 ESS14 | 0.436 | 0.203 | 0.354 | 0.189 |
| 12-29 b14 ESS27 | 0.436 | 0.203 | 0.354 | 0.189 |

D+A final:

| state | critic ESS14 | critic ESS27 | own share of ESS9 | ESS17 |
|---|---:|---:|---:|---:|
| 12-24 b8 ESS14 | **0.022** | 0.503 | 0.047 | 0.019 |
| 12-24 b8 ESS27 | 0.022 | 0.503 | 0.047 | 0.019 |
| 12-29 b14 ESS14 | **0.004** | 0.439 | 0.036 | 0.050 |
| 12-29 b14 ESS27 | 0.004 | 0.439 | 0.036 | 0.050 |

In D+A final, ESS14's own critic barely responds to ESS14's action (0.4–2.2%
of the squared sensitivity); most of its Q variation is attributed to ESS27
(36–37%) and ESS17 (34–43%). ESS27's own critic gives ESS27 44–50%. That
asymmetry matches the phase-18 observation that ESS27 is near its conditional
optimum while ESS14 is far.

This is not a shared-parameter cancellation. It is a centralized-critic credit
asymmetry: the actor's own learning signal is small because its critic's value
is mostly a function of other units' actions at those states.

## Within-actor trunk: charge versus discharge

Each actor's two output legs share `net.0` and `net.2`. The cosine between the
per-leg gradients through that trunk is:

| checkpoint | mean cos(charge, discharge) | range | mean cos(gate head, price head) |
|---|---:|---:|---:|
| D+A best | +0.098 | −0.044 … +0.214 | −0.208 |
| D+A final | 0.000 | 0.000 … 0.000 | +0.168 |

The legs are close to orthogonal at D+A best, not in conflict. At D+A final the
cosine is exactly zero because the charge gate is closed at those states and
the closed price leg has no gradient; the zero is a closed-gate effect, not a
measured conflict. The gate and price heads are also close to orthogonal.

## Does any stable cross-ESS conflict reproduce?

No.

* There is no shared actor parameter to cancel through.
* The trunk conflict measure is small and does not have the sign pattern needed
  to explain a unit-specific failure.
* The one stable asymmetry is the credit share: in D+A final ESS14's own action
  is almost absent from its own critic's Q, while ESS27's is present.

A network-structure change is therefore **not** supported by this evidence.
The credit asymmetry is a property of the trained critics, not of the
architecture, and it must be addressed (if at all) on the critic/credit side,
not by merging or separating actor trunks.

## Bound on the claim

Four training states and two checkpoints are not enough to turn the credit
asymmetry into a general law. The measurement is local (`dQ/da` at the
executed action), the Q used is `min(Q1,Q2)` at the mean, and the states are
plain deterministic rollouts. It is sufficient to reject a shared-parameter
conflict claim and to keep a credit/critic hypothesis open; it is not
sufficient to redesign the network.

## Boundary

No checkpoint was written. The actor-leg cosine is computed through the v3
(intended) map, because the execution and target use v3; the actor loss in the
current code uses v2 and that mismatch is recorded in
`phase19_actor_gradient_chain.md`.
