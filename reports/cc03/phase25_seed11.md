# Phase 25 — seed 11 under protocol v3

Run: `runs/FORMAL_1b47e0ed_seed11`, config `1b47e0ed`, interpreter
`energy_env` (Python 3.10.20, torch 2.11.0).

The run was launched twice. The first launch (2026-11-06, anaconda python) and
the second were both killed by the harness for system memory; the second had
reached episode 221 with a round-4 checkpoint. It was continued by hand with
`--resume` and completed. See `phase25_launch.md`.

## Audit

`scripts/audit_run.ps1 -RunDir runs/FORMAL_1b47e0ed_seed11` →
`PASS_WITH_NOT_TESTED`.

| item | value |
|---|---|
| status | `completed_early_stop` |
| rounds | 10 of 10–15 |
| episodes | 530 of 530–795 |
| block steps | 12,720 of 12,720–19,080 |
| updates | 12,593 |
| early stop | round 10, the minimum round |
| best | round 1, -35.288 |
| final | `checkpoints/step_00012720` |
| voltage / line violations | 0 / 0 |
| LMP fallbacks | 0 |
| terminal SOC max error | 0.0 |

Every check `PASS` except `train_log_hash_matches`, which is `INFERRED` for
TD-01 — the log is hashed before the completion line is appended. That defect
predates this protocol and was not changed.

## The resume path, exercised for real

This is the first live evidence for the Phase 24 fix, which until now had only
unit coverage.

```
RESUME from checkpoint step 5088 at episode 212; partial-round=False
```

| claim | evidence |
|---|---|
| the restart used the checkpoint's episode, not a ceiling | the log records episode 212, and `episodes.csv` is contiguous 1…530 with no duplicate |
| the round keeps its order offset | round 5 starts at `2022-08-23\|2025-08-21`, and rounds 4 and 5 have identical 53-pair order |
| every round exposes every training day exactly once | 10 rounds × 53 distinct pairs each |
| the stop rule was restored, not re-read | the run stopped at round 10, not earlier, with `min_rounds: 10` recorded in `run_manifest.json` |

`partial-round=False` because the checkpoint sat exactly at a round boundary:
the nine episodes after it had never been checkpointed, so they were truncated
and re-run. The mid-round branch of `resume_position` therefore still has unit
coverage only.

## Observation for Phase 26

Validation is highest at round 1 and degrades from there, in both protocols:

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| v2 seed 11 | -50.1 | -48.6 | -20.8 | -51.6 | -41.6 | -52.6 | -63.5 | -72.3 | -84.5 | -89.8 |
| v3 seed 11 | -35.3 | -61.4 | -77.0 | -93.7 | -80.9 | -70.4 | -66.9 | -101.9 | -93.7 | -95.0 |

v2's best is round 3; v3's is round 1. Both end about 45 points below their
best, and both are `completed_early_stop` at the minimum round because the
patience rule fires as soon as it is allowed to.

This is one seed. It is not a result about the method — it is the shape the
critic diagnostics in Phase 26 exist to explain: whether the critic's
long-horizon value estimate is drifting, or the policy is moving away from the
truthful anchor for a reward that is not payoff.

## Not established

Nothing here bears on `pi_ai`, `delta_pi` or whether the agent beats the
baseline. Four seeds are still missing, and the test block stays
`LEGACY_SEEN_TEST`.
