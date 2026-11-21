# P1-A — v3 quote-map mapping regression

Protocol `DIAGNOSTIC_PLAN_V1`, phase P1-A (read-only).
Written 2026-11-08. Interpreter
`C:\Users\26036\.conda\envs\energy_env\python.exe` (CPython 3.10.20),
torch `2.11.0+cpu`, numpy `1.24.3`, gurobipy `13.0.1`.

Collector: `src/diagnose_actor_update.py` (new, unpinned), function
`mapping_regression`. Command:

```
PYTHONPATH=src "C:/Users/26036/.conda/envs/energy_env/python.exe" \
  src/diagnose_actor_update.py --mapping-only
```

Result: `mapping_all_pass: true`, exit 0, **27 / 27 checks pass**.
The full P1-B actor-step run writes the same mapping block into
`P1_evidence.json`; the numbers below were read from the `--mapping-only`
pass, which performs the identical computation and writes no artifact.

## Preconditions

| item | verdict | value |
|---|---|---|
| 19 live `source_sha256` | **PASS** | all 19 match on disk, `mismatch []` |
| 6 pinned tests/audits in the live manifest | **PASS** | all 6 match (`test_da_quote_mapping.py`, `test_train_da_gate_guards.py`, `test_train_da_gate_resume.py`, `test_training_contract.py`, `test_frozen_guard.py`, `audit_phase20_action_routing.py`) |
| `thresholds.json` unchanged | **PASS** | sha256 `12549ac67a879f1bec4de4e49c082839ab0abd6b02e43ff254029f569ab54592` |
| seed 11 best | **PASS** | `runs/FORMAL_1b47e0ed_seed11_best`, step 1,272, episode 53, `actor_loss_map=linear_core_v1`, `quote_map=linear_core_v1`, `obs_spec=v6_12d` |
| seed 23 best | **PASS** | `runs/FORMAL_1b47e0ed_seed23_best`, step 5,088, episode 212, same three fields |

Read directly from each checkpoint's `manifest.json`. Code identity and
checkpoint identity are therefore both established before any mapping test.

## Route table

Call sites, by symbol rather than line:

| path | site | call | verdict |
|---|---|---|---|
| execution | `rl_da_gate.DAGateTrainer._policy_action` | `da_quote_map.forward_policy_v3(..., train=False)` | **PASS** |
| actor loss | `rl_da_gate.DAGateTrainer._actor_loss_quotes` | `da_quote_map.forward_policy_v3(..., train=True, gate_temperature=trainer.current_gate_temperature())` | **PASS** |
| critic action inside the actor loss | `rl_b_c2_gate.MATD3BC2Gate.update` | `_actor_loss_quotes(policy_out)` | **PASS** |
| target actor | `rl_da_gate.DAGateTrainer._target_quotes` | `da_quote_map.target_quotes_v3(...)` | **PASS** |
| critic target action inside the critic loss | `rl_b_c2_gate.MATD3BC2Gate.update` | `_target_quotes(self.actor_targets[...])` | **PASS** |
| retired v2 map | `rl_da_gate` module source | no `forward_policy(` occurrence | **PASS** |

Checked by source inspection (`inspect.getsource`) of `_policy_action`,
`_actor_loss_quotes`, `_target_quotes` and `MATD3BC2Gate.update`.

## Call audit in one real update

`audit_phase20_action_routing.update_call_audit` instruments one
`MATD3BC2Gate.update()` on a synthetic buffer:

| field | value |
|---|---|
| `v2_calls_in_update` | 0 |
| `v3_calls_in_update` | 20 |
| `actor_loss` | 1.689203068614006 (finite) |

Twenty = 4 actor-loss forwards (one per agent) + 16 target forwards (4 agents
× 2 critics in the twin-target bootstrap). The v2 map is not reachable from an
update. The scalar `actor_loss` depends on the freshly initialised critic, so
it is a finiteness check, not a value to compare against the earlier standalone
audit run (1.6761), whose initialisation drew different weights from the
global RNG stream.

## Storage-routing checks (`audit_phase20_action_routing.routing_checks`)

| check | value |
|---|---|
| ESS order identical in expectation, trainer and real env | `ESS9, ESS14, ESS17, ESS27` |
| actor output dimension | 4 |
| quote dimension | 2 |
| critic input dimension | 56 (4 × 12 obs + 4 × 2 actions) |
| closed charge gate quote | exactly `0.0` |
| open discharge quote | 297.0 |
| target route equals the v3 map | true |
| target route finite under injected noise | true |
| same leg bounds as v3 `(150, 300)` | true |

## Map numerics

### Finite difference against the analytic derivative

`map_fraction` at seven points, central difference with `h = 1e-6`:

| z | finite difference | `map_derivative` | abs error |
|---|---|---|---|
| -10.0 | 4.740097e-07 | 4.740097e-07 | 2.9e-16 |
| -2.5 | 0.19599856 | 0.196 | 1.4e-06 |
| -1.0 | 0.19600000 | 0.196 | 2.0e-11 |
| 0.0 | 0.19600000 | 0.196 | 7.9e-12 |
| 1.0 | 0.19600000 | 0.196 | 7.9e-12 |
| 2.5 | 0.19599856 | 0.196 | 1.4e-06 |
| 10.0 | 4.740097e-07 | 4.740097e-07 | 5.9e-15 |

The linear core is exactly 0.196; the quadratic tails decay to ~4.7e-07 at
|z| = 10. `autograd_matches_map_derivative`: true (atol 1e-10).

### Closed gate

With logits `[-2.0, 3.0, 1.0, 1.0]` and `train=False`:

- the closed charge leg's quote is exactly `0.0`;
- the closed leg's price logit receives gradient `0.0` — the gate multiplies
  the quote, so a shut leg cannot be moved by its own price gradient;
- the open discharge leg's price logit receives 58.8.

A closed gate sets the declared price to zero; it does not remove the leg from
the clearing. That behaviour is the same one phase 25 observed in the formal
runs.

### Straight-through gate gradient

In train mode `forward_policy_v3` builds
`gate = hard + (probs - probs.detach())`, so the surrogate gradient reaching a
gate logit is `probs * (1 - probs) / tau * price_quote`, **not**
`p(1-p) * price_quote`. The test at `tau = 2.0`, logits
`[0.4, -0.3, 1.5, -1.5]`, `price_quote[0] = 119.1` measures a maximum absolute
error of `8.88e-16` against that expression.

The frozen formal gate temperature starts at 4.0, so at step 0 the effective
gate gradient is a quarter of the `p(1-p)` magnitude; at the two best
checkpoints it is `tau = 2.5` (seed 11, episode 53) and `tau = 1.0` (seed 23,
episode 212). `da_quote_map._STGate` exists but is never called; the gradient
comes from the subtraction above. This is a measured property of the map, not a
routing defect — it is recorded because P1-B moves gate logits and the step
size depends on it.

### Determinism and path identity

| check | result |
|---|---|
| same logits + same torch seed → same gate draw | true |
| same logits + same torch seed → same quotes | true |
| `target_quotes_v3` == eval-mode `forward_policy_v3` | true (bitwise) |
| `critic_action_v3` == eval-mode `forward_policy_v3` | true (bitwise) |
| environment execution (`select_action`) == direct map call | true |
| actor-loss map with the trainer's own temperature == direct v3 call | true |
| trainer gate temperature at a fresh trainer | 4.0 |

## What this establishes and what it does not

Established (**PASS**): every path that quotes a price — execution, actor loss,
critic action inside the actor loss, target actor, critic target — is the v3
`linear_core_v1` map; the v2 map is absent from the module and uncalled during
an update; the map's derivative, straight-through gradient, closed-gate
behaviour and determinism match their analytic values to 1e-10 or better.

Not established:

- **NOT_TESTED** — that this routing holds *during* the P1-B actor steps, which
  run in the full P1-B pass and are guarded there by the actor-loss route
  checks plus the checkpoint manifest fields.
- **NOT_TESTED** — any claim about learning quality. A correct map says nothing
  about whether ascending the critic improves the payoff; that is P1-B and P2.
- **INFERRED** — nothing in this report; every value above was measured in this
  session.

## Exit

**`PASS_P1_MAPPING`** — 27 / 27 checks, 0 v2 calls in an update, code and
checkpoint identity pinned. The later full run reproduced the same mapping
block inside `P1_evidence.json` (`mapping.all_pass: true`, 27 checks), so the
routing result holds for the collector process that produced the P1-B
measurements as well. The phase's actor-step half is in
`P1_actor_single_step_report.md`.
