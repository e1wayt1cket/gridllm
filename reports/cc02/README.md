# CC-02 — engineering evidence for the formal training contract

Authority: MASTER_PLAN_V1 §2, §3, §9. CC-02 is a verification phase. It does not
change an algorithm, a frozen file, a run or a manifest, and it does not start
or resume training.

## Verdict

| # | item | verdict | evidence |
|---|---|---|---|
| 1 | `independent-reviewer` / `experiment-auditor` are read-only at runtime | `PASS` | `agent_readonly_probe.md`: both report `Read, Grep, Glob, SubagentHandback`; write, shell and subagent canaries all `*_UNAVAILABLE`; no canary file created |
| 2 | frozen-file hook denies a frozen path and allows a normal one | `PASS` | `agent_readonly_probe.md` §4; hook event returns `permissionDecision: "deny"` for `runs/FORMAL_2148a577_seed53/run_manifest.json` and nothing for a non-frozen path |
| 3 | guard tests | `PASS` | `python -m pytest tests/test_training_contract.py tests/test_frozen_guard.py tests/test_train_da_gate_guards.py -v` → 43 passed |
| 4 | seed 11 / 23 / 37 audit | `PASS_WITH_NOT_TESTED` | `seed_audits.md`, `seed_audits/*.json`: 10 rounds, 530 episodes, 12,720 block steps each; the one non-passing check is `train_log_hash_matches: INFERRED` (TD-01) |
| 5 | seed 53 audit | `NO_GO_RUN_FAILED` | `seed_audits/FORMAL_2148a577_seed53.json`: 28 episodes, 672 block steps, no best or final checkpoint |
| 6 | seed 53 root cause | `NOT_REPRODUCED` (`INFERRED`) | `seed53_diagnosis.md`, `seed53_replay.json` |
| 7 | formal gate | `PASS` | `run_formal_check.json`: `all_pass: true`, `budget_contract_ok: true`, `status: GO_FORMAL_TRAINING`, `config_sha256 2148a577…`, no violations |
| 8 | launcher exit codes | `PASS` | `launcher_verification.md`: 1 missing seed, 3 populated run, 3 through `Start-Process`, 10 preflight failure, 11 contract failure |
| 9 | minimum-round guard runs in the training loop | `PASS` | `phase24_verdict.md`: `src/train_da_gate.py:109` + call at 892–893, probe `GUARD_CHECKS_PASS`, three runs stopped at round 10 |
| 10 | resume restores all counters and state | `FAIL` | `phase24_verdict.md`: patience and `min_rounds` not in the resume `checks` dict; partial round rounded up so 25 training days lose one exposure (TD-05) |
| 11 | no frozen artifact changed | `PASS` | `guardrail_before.json` and `guardrail_after.json` both `identical: true` over 11 seed trees, 3 unsnapshotted trees, 2 manifest dirs and 18 protected files; `run_inventory` 2117 files with no added, removed or modified entry |
| 12 | no training started, resumed or rerun | `PASS` | `runs/FORMAL_2148a577_seed71` does not exist; `git_status_after.txt` adds only `reports/cc02/**` and `src/diagnose_seed53_clearing.py` |

CC-02 overall: **PASS with two recorded `FAIL`s that are protocol findings, not
execution failures.** Items 10 blocks entry to Phase 25/26 until the user either
commissions a fix or accepts the risk, per the confirmed disposition. The
`--max-episodes` finding below is a third item needing a user ruling.

## What each verdict is based on

The four things stay separate, as the reporting rule requires.

1. **Code exists.** `src/train_da_gate.py` contains `should_early_stop` and the
   resume block; `src/diagnose_seed53_clearing.py` contains the replay harness.
2. **Code ran.** The probe, the four audits, the launcher tests and the
   diagnosis script were all executed; their exit codes and stdout are recorded
   in this directory.
3. **A test covers it and passed.** 43 tests passed. They cover the guard's
   arithmetic and the frozen hook, not a real resume.
4. **The scientific conclusion follows.** No scientific conclusion is drawn in
   CC-02. Nothing here bears on `pi_ai`, `delta_pi` or whether the agent
   improves on the baseline; the three completed seeds are still one protocol,
   four seeds short of the planned five.

## Protocol findings needing a user decision

- **TD-05** — resume does not validate `patience` / `min_rounds` and rounds a
  partial round up. Fixing it edits a pinned file and needs a new candidate and
  a re-freeze.
- **TD-06** — `src/audit_phase22_resume.py` writes into the superseded
  `formal_training_ready/` directory. Not run in CC-02.
- **TD-07** — `src/run_formal_training.py` is not pinned, and the recorded
  seed 11/23/37/53 launch commands omit `--max-episodes`, which the frozen rule
  says makes them not a formal launch. Numerically inert (53 × 15 = 795), but
  literal.
- **Seed 53 disposition** — keep, rerun under a new run-id, or substitute a new
  seed. The seed list is frozen in `formal_training_config.yaml`, so a
  substitution needs a re-freeze. Recommendation in `seed53_diagnosis.md`:
  keep the failed run, launch 71, decide on a fifth seed later.

## Not tested

Each of these has no evidence and no workaround inside the CC-02 boundary.

| item | why | risk |
|---|---|---|
| `launch_seed.ps1` success path | it launches training | the wrapper's happy path is only exercised via the refusal and failure branches |
| a real resume, full round or partial round | `--resume` truncates files in the frozen run tree | TD-05's two defects are static-source findings, not observed behaviour |
| runtime behaviour when a resume supplies different `patience` / `min_rounds` | same | a silently different stop rule is possible |
| a stop attempt before round 10 in a live run | would need a run | only the pure function and the three completed runs were observed |
| seed 53's original stochastic trajectory | the failure checkpoint's RNG is post-failure and the actions were never logged | the classification is `INFERRED`; the mechanism hypothesis is unproven |
| solver non-determinism beyond repeated calls | no infeasible input was ever captured | `SOLVER_TRANSIENT` was not testable |
| positive proof that an agent shell write is refused | neither agent has a shell tool | read-only behaviour rests on the frontmatter tool list (TD-03) |

Nothing was `BLOCKED`: the Gurobi licence was valid, the agents were available,
and the formal gate returned `GO_FORMAL_TRAINING` on the first attempt.

## Files

| file | contents |
|---|---|
| `agent_readonly_probe.md` | three-layer agent probe, hook test, TD-03 boundary |
| `seed_audits.md`, `seed_audits/*.json` | per-run machine audit for seeds 11/23/37/53 |
| `seed53_diagnosis.md`, `seed53_replay.json` | classification, limitation, mechanism hypothesis, disposition options, full replay capture |
| `launcher_verification.md` | gate, dry run, five exit-code tests, the `--max-episodes` finding |
| `phase24_verdict.md` | §3 matrix, guard probe, resume findings |
| `run_formal_check.json` | `run_formal_training.py --check` output |
| `guardrail_before.json`, `guardrail_after.json` | frozen-baseline re-hash at start and end |
| `run_inventory_before.json`, `run_inventory_after.json` | size and mtime of every file under `runs/` and `snapshots/` |
| `git_status_before.txt`, `git_status_after.txt` | `git status --porcelain -uall`, start and end |
| `seed53_sweep40.log` | 40-seed exploration sweep output |

`TECHNICAL_DEBT.md` gained TD-05, TD-06 and TD-07. It is untracked in git, so
its edit does not appear in `git_status_after.txt`.

## Reproduce

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/preflight.ps1
python src/run_formal_training.py --check
python -m pytest tests/test_training_contract.py tests/test_frozen_guard.py tests/test_train_da_gate_guards.py -v
python src/audit_run.py --run-dir runs/FORMAL_2148a577_seed11     # and seed23, seed37, seed53
python src/run_formal_training.py --seed 71 --dry-run
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_seed.ps1              # exit 1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/launch_seed.ps1 -Seed 53     # exit 3
PYTHONPATH=src python src/diagnose_seed53_clearing.py --run-dir runs/FORMAL_2148a577_seed53 --sweep-count 40
```

`src/diagnose_seed53_clearing.py` is the only new source file. It reads the run
directory, disables `gp.Model.write` so an infeasible model cannot dump an
`.ilp` into the tree, and writes only under `reports/cc02/` with `--write`.

## Stop conditions

None was triggered. No frozen hash changed, no run or snapshot was written, no
`.ilp` was produced, no canary was created, `runs/FORMAL_2148a577_seed71` was
never created, and no training process was started.
