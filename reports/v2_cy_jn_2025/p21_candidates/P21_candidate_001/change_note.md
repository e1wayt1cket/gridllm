# P21 candidate 001: actor-loss v3 quote map

## Hypothesis

Phase 19 established that `DAGateTrainer` executes and targets through
`da_quote_map.forward_policy_v3` (`linear_core_v1`), but inherits
`MATD3BC2Gate.update`, whose actor loss maps the actor output through
`truthful_gate_actor_adapter.forward_policy` (v2 sigmoid). The actor gradient is
therefore not the gradient of the action it executes. The inherited update also
uses the fixed evaluation gate temperature (1.0) instead of the training
warmup schedule.

## Change

- `src/rl_b_c2_gate.py`: the actor-loss quote map is now one override point,
  `_actor_loss_quotes(policy_out)`. The base class keeps the v2 map.
- `src/rl_da_gate.py`: `DAGateTrainer` overrides the point with
  `da_quote_map.forward_policy_v3(..., train=True,
  gate_temperature=self.current_gate_temperature())`.
- `src/rl_da_gate.py`: checkpoint metadata records
  `actor_loss_map: linear_core_v1` and
  `actor_loss_gate_temperature: current_gate_temperature`.
- `src/train_da_gate.py`: shared engineering used by both the control and the
  candidate: common-split selection, fixed training subset, early-stop rule,
  block trace, run-directory guard, resume validation and log truncation.
  These do not change the reward, observation, gate, network or action space.

The control run (`P21_control_smoke_seed11`) was produced after the shared
engineering and before the map fix. The candidate differs from it in the actor
loss map (and its metadata) only.

## Tests

| test | result |
|---|---|
| `tests/test_da_quote_mapping.py` | 7 passed |
| `tests/test_rl_b_c2_gate.py` | 13 passed |
| `tests/test_truthful_gate_actor_adapter.py` | 6 passed |
| `tests/test_train_da_gate_guards.py` | 4 passed |
| `src/audit_phase20_action_routing.py` | all_pass = true, `v2_calls_in_update = 0` |

The equality test compares the update-path quote to the execution-path quote
under the same gate draw, over gate open/closed, both legs, batched input and
low/mid/high price logits. The finite-difference test checks the v3 map
derivative. The base trainer regression keeps the v2 map.

## Smoke result

Pre-registered protocol: `reports/v2_cy_jn_2025/phase21_smoke_protocol.json`.
Common split train days 2022-08-14, 10-13, 11-01, 12-17, 12-27; seed 11; four
rounds = 20 episodes; four validation passes; no test, holiday or H2.

| run | best round | validation mean fleet ΔG | positive days |
|---|---:|---:|---:|
| control (pre-fix) | 2 | −25.0 | 6 / 11 |
| candidate 001 | 4 | −18.5 | 7 / 11 |

The candidate passes every hard gate: exact 20 episodes / 480 block
interactions, 0 clearing fallbacks, 0 voltage/line violations, terminal SOC
0.5, B1 settlement identity closed, no test/H2. Quote medians and high-quote
shares remain comparable to the control; no new corner collapse is present.
The pre-registered non-inferiority rule is satisfied.

## Scope

The smoke is a controlled diagnostic, not a generalisation result. The selected
checkpoint is not a formal model. No historical checkpoint, report, data file
or manifest was modified. The split roles of the named Phase 20 states remain:
12-13 and 12-30 are validation under `near/split.json` (the D+A training split)
and training under the formal `common_split`; the smoke uses neither.
