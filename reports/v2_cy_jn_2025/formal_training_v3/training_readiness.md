# Training readiness — protocol version 3

**training_readiness_status = GO_FORMAL_TRAINING**
Config hash `1b47e0ed`, prefix used by every run id: `FORMAL_1b47e0ed_seed<N>`.

Version 3 changes the run guards and the launcher only. Reward, observation,
action, actor, critic, gate, quote map, optimizer settings, exploration, replay,
gamma, model selection, data split and seeds are unchanged. The v2 directory
(`2148a577`) and `formal_training_ready` (`b21a97c1`) are untouched.

## What changed from v2

| item | v2 | v3 |
|---|---|---|
| `--min-rounds` default | 0 | 10 |
| round ceiling | none in code | `--max-rounds`, clamped in the loop and checked after validation |
| resume restart | next full-round boundary (partial round discarded) | the checkpoint's exact episode, keeping the round's order offset |
| resume stop rule | re-read from the CLI | compared with the manifest; a missing key refuses |
| resume evaluations | every row in `eval_history.json` | filtered to completed rounds |
| best on resume | rebuilt from history and snapshots | the manifest's `best` first |
| launcher pins | launcher unpinned | launcher, contract, auditor and generator pinned |
| launcher code list | its own copy | the generator's `code_payload()` |

## Frozen budget

| item | value |
|---|---|
| minimum rounds | 10 |
| maximum rounds | 15 |
| episodes per seed | 530–795 |
| block steps per seed | 12,720–19,080 |
| early-stop patience | 3 |

Rounds 1–9 can never stop. Round 10 is the first round at which a satisfied
patience may stop, and it is evaluated before that decision. Round 15 is the
last round that runs.

## Verification

| check | result |
|---|---|
| `src/run_formal_training.py --check` | `all_pass: true`, `budget_contract_ok: true`, exit 0 |
| `src/training_contract.py --check` | `ok: true`, `violations: []` |
| guard suite | 20 passed |
| resume suite | 8 passed |
| contract + frozen guard | 41 passed |
| full suite | 1429 passed, 1 failed, 53 deselected |
| data manifests | 53/11/11/5, disjoint, legacy label matches, H2 sealed |

The one full-suite failure is
`tests/test_formal_ensemble.py::test_every_consumer_reads_the_declared_comparison_set`.
It fails at `src/make_rot_comparison_figures.py:28` with
`module 'matplotlib' has no attribute 'use'`, and `import matplotlib` in this
interpreter reports `__file__ = None`. It reproduces in isolation, touches no
file changed here, and is an environment defect in the plotting dependency.
It is recorded, not worked around.

## Not covered

- The resume path has unit and helper coverage but no live resume: no run was
  started, and a real mid-round resume is only exercised from Phase 25 onward.
- `--dry-run` prints the launch command; the launcher's success path still
  starts training, so it remains unexercised until a seed is launched.
- TD-01 (the `train.log` hash is taken before the completion line) is unchanged,
  so v3 runs will still audit as `PASS_WITH_NOT_TESTED`.
- TD-02 (one infeasible clearing aborts a seed) is unchanged.

See `pretraining_validation_results.json`.
