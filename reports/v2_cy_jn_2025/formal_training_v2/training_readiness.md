# Training readiness — protocol version 2

**training_readiness_status = GO_FORMAL_TRAINING**

This version differs from the original frozen protocol in one enforced rule:
early stopping cannot trigger before 10 completed training rounds with
validation. Reward, observation, action, actor, critic, gate, quote map,
optimizer settings, exploration, replay, gamma and model selection are
unchanged. The original `formal_training_ready` directory and its hashes are
untouched.

## Frozen budget

| item | value |
|---|---|
| minimum rounds | 10 |
| maximum rounds | 15 |
| episodes per seed | 530–795 |
| block steps per seed | 12,720–19,080 |
| early-stop patience | 3 |

Early stop is checked only after a round's validation has been recorded. At and
after round 10 the original patience rule applies; a new best at round 10
resets the streak. The run never exceeds the 15-round budget.

## Verification

- Guard suite: 13 passed, including no stop at rounds 3–9, stop at round 10
  when patience is met, new-best reset at round 10, normal stop after round 10,
  the 15-round cap, and resume restoration.
- Targeted suite: 27 passed (guards, checkpoint roundtrip, checkpoint schema,
  quote map); 0 failed.
- Data manifests: 53/11/11/5, all disjoint, legacy label matches, H2 sealed.

See `pretraining_validation_results.json` and
`phase24_budget_guard_tests.json`.
