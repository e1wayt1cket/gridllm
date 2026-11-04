# Phase 20-E: action routing and update-path call audit

Read-only. One in-memory update, no checkpoint write, no H2 and no legacy test.
Data: `phase20_action_routing.json`. Code:
`src/audit_phase20_action_routing.py`.

## Checks

| check | result |
|---|---|
| ESS order: expected / trainer / real env | PASS (ESS9, ESS14, ESS17, ESS27) |
| actor output width | PASS (4: gate charge, gate discharge, price charge, price discharge) |
| quote width | PASS (2: charge, discharge) |
| critic input width | PASS (4 × 12 observations + 4 × 2 quotes = 56) |
| closed gate quote | PASS (exactly 0) |
| open discharge quote | PASS (positive) |
| target route equals v3 with the configured noise | PASS |
| finite-difference dQ/dquote vs autograd | PASS (both legs, eval mode) |
| v2 calls during one actor update | **0** |
| v3 calls during one actor update | positive |
| actor loss finite | PASS |

## Notes

- The online critic has a dropout layer. The finite-difference check evaluates
  the critic without dropout so both sides of the difference use the same mask;
  training-time critic gradients remain stochastic. The routing identity is
  deterministic in eval mode.
- `build_env_for_pair` attaches `ACTION_STORAGE_QUOTE_V2` as environment
  metadata, while the trainer declares `ACTION_STORAGE_QUOTE_V3`. The bounds and
  truthful point are identical; the environment uses the spec only for bounds.
  This is a metadata difference, not a numeric action difference. The formal
  action schema recorded in the checkpoint is v3.
- The update audit patches both the base v2 map and the v3 map counters. The
  zero v2 count is the direct evidence that the actor-loss branch no longer
  calls the sigmoid map.

## Answer

Every D+A route that produces a critic action or an actor-loss quote uses the
same ESS order, leg order, tensor shape and v3 map. The actor loss calls the v3
map and does not call the v2 map. The routing audit passes.

## Boundary

The update is an in-memory one-step call on a synthetic buffer; it is not a
training episode. No checkpoint or historical run was modified.
