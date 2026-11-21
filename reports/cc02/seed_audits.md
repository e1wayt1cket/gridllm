# CC-02 seed audits

Command for each run (read-only; stdout redirected into this directory):

```
PYTHONPATH=src python src/audit_run.py --run-dir runs/FORMAL_2148a577_seed<N>
  > reports/cc02/seed_audits/FORMAL_2148a577_seed<N>.json
```

`reports/cc01/audit/` was not written. The existing seed 11 evidence there was
only read for comparison.

## Results

| seed | exit | status | rounds | episodes | block steps | best round | verdict |
|---|---|---|---|---|---|---|---|
| 11 | 0 | `completed_early_stop` | 10 | 530 | 12,720 | 3 | `PASS_WITH_NOT_TESTED` |
| 23 | 0 | `completed_early_stop` | 10 | 530 | 12,720 | 1 | `PASS_WITH_NOT_TESTED` |
| 37 | 0 | `completed_early_stop` | 10 | 530 | 12,720 | 1 | `PASS_WITH_NOT_TESTED` |
| 53 | 1 | `failed` | 0 | 28 | 672 | — | `NO_GO_RUN_FAILED` |

Budget read by the auditor: 10–15 rounds, 530–795 episodes, 12,720–19,080
block steps; minimum 10 rounds.

The only non-`PASS` check on seeds 11/23/37 is `train_log_hash_matches:
INFERRED` (TD-01: the log was hashed before the completion line was
appended). The physical and isolation checks all passed: zero voltage
violations, zero line overloads, zero LMP fallbacks, terminal SOC error 0.0,
`test_evaluation: null`, `legacy_seen_test_run: false`, `h2_holdout_read:
false`.

Seed 53's failing checks are `run_terminated_cleanly`,
`rounds_at_least_minimum`, `episodes_in_budget`, `block_steps_in_budget`,
`best_checkpoint_exists`, `final_checkpoint_exists`, `best_is_not_final`. It
stopped at episode 28 of 530 after an infeasible clearing (TD-02); the shortfall
is not softened into a budget finding.

## Seed 11 cross-check against `reports/cc01/audit/`

The freshly computed audit equals the existing CC-01 audit in every field
except `run_dir` (absolute path then, relative path now). All checks,
counts, `best_value` and verdict are identical.

## Independent review

`experiment-auditor` re-derived seed 11 from the run artifacts and returned
`PASS_WITH_NOT_TESTED`, matching the machine audit. It could not recompute
hashes (read-only tools) and could not compare `.pt` bytes; those checks rest
on `run_manifest.json`, the checkpoint inventory and the finalize evidence.
