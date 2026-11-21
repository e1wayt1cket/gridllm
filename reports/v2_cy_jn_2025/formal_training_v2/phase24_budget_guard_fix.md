# Minimum-budget guard fix and refreeze

**Status: fix implemented, guard suite passing, protocol version 2 frozen,
Seed 11 rerun launched from initialization.**

## Defect

The original training entry applied `early_stop_patience` without any floor.
Seed 11 reached its best validation at round 3 and stopped at round 6, which is
below the pre-registered minimum of 10 exposures per training day (530
episodes, 12,720 block steps). The run was recorded as
`PROTOCOL_DEVIATION_EARLY_STOP_BELOW_MINIMUM` and is preserved read-only.

## Change

Only the early-stop decision and the plumbing that carries the frozen numbers
changed.

`src/train_da_gate.py`:

- new `should_early_stop(round_idx, evaluations, patience, min_rounds)`:
  returns false while `round_idx < min_rounds`; at or after the floor it
  applies the existing `no_improvement_streak >= patience` rule;
- `round_idx` counts rounds whose validation has been recorded, so a partial
  round cannot satisfy the floor;
- new `episode_budget(order_size, exposures, max_episodes=0)` and
  `completed_rounds(episodes, order_size)` pure helpers; the resume path uses
  `completed_rounds` and still discards a partial round at the next boundary;
- new `--min-rounds` argument (default 0, preserving non-formal behaviour);
- `early_stop` and the run manifest record `min_rounds` alongside `patience`.

`src/run_formal_training.py`:

- `--config-dir` selects a frozen version directory;
- the run id, `--exposures`, `--early-stop-patience` and `--min-rounds` are
  read from that directory's config rather than hardcoded, so the entry cannot
  drift from the frozen numbers.

`src/make_formal_training_manifests.py`:

- `--out` writes a new version directory instead of the original;
- the config payload adds `protocol_version: 2` and
  `budget.minimum_rounds: 10`;
- finalize records the actual run date.

No change was made to reward, observation, action, actor, critic, gate, quote
map, learning rate, batch size, replay buffer, gamma, exploration, model
selection, or data isolation. Fourteen of the fifteen code-manifested sources
have byte-identical hashes; only `train_da_gate.py` changed.

## Behaviour at the boundary

| situation | result |
|---|---|
| rounds 3–6 without improvement | no early stop |
| rounds 7–9 without improvement | no early stop |
| round 10 completed, patience met | early stop allowed |
| round 10 sets a new best | best updated, streak reset, no stop |
| best at round 10, rounds 11–13 non-improving | stop at round 13 |
| no early stop | exactly 15 rounds, never more |
| resume | completed rounds and patience streak restored |

## Tests

`python -m pytest tests/test_train_da_gate_guards.py -v` → 13 passed.
The targeted suite (guards, checkpoint roundtrip, checkpoint schema, quote map)
→ 27 passed, 0 failed. Details:
`phase24_budget_guard_tests.json`.

Coverage limit: these are pure-decision and budgeting tests. They do not
re-execute a real multi-round training loop; the wiring is a single call to
`should_early_stop` immediately after validation. The live rerun is the
end-to-end check.

## Refreeze

New version directory: `reports/v2_cy_jn_2025/formal_training_v2/`.

| hash | value |
|---|---|
| new config payload | `2148a5777ea6f16a4f5f26730e04ec5cfa4e3d3476c0b50586fe203ba5e4f68f` |
| previous config payload | `b21a97c15d16aae5ed0f8bd599e54a2f8aafb6e687bca2b63b06bd5a8da8e1e7` |
| new `train_da_gate.py` | `a722b2f9266a6ab5dfe7cceeccf3b16ae6f01c3e47cf6c50672f245c9eb6c79c` |
| previous `train_da_gate.py` | `c32b7fec591c5f0e2a7b42b91778f0931217d5c60207c3644e385a54e564b9b6` |

The original `formal_training_ready` directory was not overwritten. Its config
payload, code-manifest payload and data-manifest payload still match the
recorded hashes; see `phase24_refreeze_manifest.json`.

Preflight for the new version:
`python src/run_formal_training.py --check --config-dir reports/v2_cy_jn_2025/formal_training_v2`
→ `all_pass: true`, `GO_FORMAL_TRAINING`.

## Seed 11 rerun

New run id: `FORMAL_2148a577_seed11`. It starts from initialization, uses
`--min-rounds 10 --early-stop-patience 3 --exposures 15`, and does not read or
warm-start from any previous checkpoint. It is running; the completion report
is generated after it finishes.
