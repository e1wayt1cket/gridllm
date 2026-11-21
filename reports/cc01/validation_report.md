# Validation report

Phase CC-01. Verdict vocabulary: PASS / FAIL / NOT_TESTED / INFERRED / BLOCKED.

The four levels are kept apart: a file existing is not the same as it running, and
running is not the same as being tested.

## 1. Frozen set unchanged — PASS

`reports/cc01/frozen_audit.json` holds the pre-implementation SHA256 of the 18
protected files, the two manifest directories and every `runs/` / `snapshots/`
tree named by either config. The `after` block was recomputed at the end of the
phase with the same digest rule
(`sha256(json.dumps({path: sha256(file)}, sort_keys=True))`).

| Scope | Count | Identical |
|---|---|---|
| Protected files (15 sources + 3 pinned tests/audits) | 18 | yes |
| Manifest directories | 2 | yes |
| Seed run / `_best` / snapshot trees | 11 | yes |

`identical: true`, `drift: []`. The seed trees hold 212 + 62 + 11 files for
seed 11, 209 + 22 + 11 for seeds 23 and 37 (run, `_best`, snapshot), and 29 + 0
for the failed seed 53. Each tree is compared by per-file digest, not by count.

The superseded config's runs (`FORMAL_b21a97c1_seed11`, `_best`, snapshot; 133
+ 62 + 7 files) were not in the before snapshot. Their current digests are
recorded under `after.unsnapshotted_trees` as a baseline for a later phase; no
before/after claim is made for them.

The 18 protected files also match what the code manifest records —
`code_sources_match: true` and `code_tests_match: true` in section 3.

## 2. Budget contract — PASS

`src/training_contract.py` re-derives every total from `minimum_rounds`,
`train_days` and `episode.blocks`. On the live config:

```text
PYTHONPATH=src python src/training_contract.py --check
{"ok": true, "config_dir": "reports\\v2_cy_jn_2025\\formal_training_v2", "violations": []}
exit 0
```

A mutated copy with `episodes_per_seed_min: 531` instead of `530` produces two
named violations and exit 1:

```text
min_episodes_mismatch      formal_training.budget.episodes_per_seed_min      expected 530   actual 531
min_block_steps_mismatch   formal_training.budget.block_steps_per_seed_min   expected 12744 actual 12720
```

`tests/test_training_contract.py` covers each invariant by name (12 tests).

## 3. Formal launcher — PASS

Default `--check`, with no `--config-dir`:

```text
PYTHONPATH=src python src/run_formal_training.py --check
status GO_FORMAL_TRAINING   all_pass true   budget_contract_ok true   config_sha256 2148a577
exit 0
```

The superseded directory does not fall back:

```text
PYTHONPATH=src python src/run_formal_training.py --check --config-dir reports/v2_cy_jn_2025/formal_training_ready
status NO_GO   declared_status GO_FORMAL_TRAINING   all_pass false
code_sources_match false   code_tests_match false   budget_contract_ok false
contract_violations [{code: minimum_rounds_not_integer}]
exit 1
```

`status` is now downgraded to `NO_GO` whenever any check fails; the status the
manifest declared at freeze time is reported separately as `declared_status`.
Previously a failing directory echoed `GO_FORMAL_TRAINING`.

Dry run of the next seed, which prints the command and starts nothing:

```text
PYTHONPATH=src python src/run_formal_training.py --seed 71 --dry-run
python src/train_da_gate.py --split common --seed 71 --exposures 15 --early-stop-patience 3 \
  --min-rounds 10 --max-episodes 795 --run-id FORMAL_2148a577_seed71 --trace-blocks
exit 0
```

`command()` reads `minimum_rounds`, `exposures_per_day_max` and
`episodes_per_seed_max` from the same file `verify()` validated, so the command
cannot disagree with the contract that authorised it. There is no second copy of
the budget in the launcher.

## 4. Write guard — PASS

`.claude/hooks/protect_frozen.py` is exercised directly over stdin, so the result
does not depend on whether the current session reloaded `settings.json`.

| Path | Result |
|---|---|
| `src/train_da_gate.py` | deny |
| `tests/test_train_da_gate_guards.py` | deny |
| `reports/v2_cy_jn_2025/formal_training_v2/formal_training_config.yaml` | deny |
| `reports/v2_cy_jn_2025/formal_training_ready/formal_training_config.yaml` | deny |
| `runs/FORMAL_2148a577_seed11/train.log` | deny |
| `runs/FORMAL_2148a577_seed53/run_manifest.json` | deny |
| `runs/FORMAL_b21a97c1_seed11/train.log` | deny |
| `snapshots/FORMAL_b21a97c1_seed11/round_001.json` | deny |
| `src/training_contract.py` | allow, no output |
| `reports/cc01/README.md` | allow, no output |

The superseded config's runs were initially allowed. The guard now derives the
protected run ids once per manifest directory, so both configs' runs are frozen.
`tests/test_frozen_guard.py` covers this (18 cases: the allow/deny matrix,
relative paths, traversal, non-`Edit`/`Write` tools, missing `file_path`,
malformed JSON).

Denial is exit 0 with `hookSpecificOutput.permissionDecision = "deny"`; a guard
crash that cannot even emit that JSON falls back to exit 2.

Fail-closed was checked on a throwaway copy of the guard in an empty project
tree, where no manifest exists: `src/anything_new.py` and
`runs/FORMAL_x_seed11/train.log` are denied, `reports/cc01/README.md` is allowed.
With a manifest present, only the pinned files are denied and unrelated sources
stay writable — that is the difference the two zones are for.

## 5. Read-only auditor — PASS

`src/audit_run.py` reads a run directory and writes nothing unless `--write` is
passed, in which case the evidence goes to `reports/cc01/audit/<run-id>.json`,
outside the run.

| Run | Reported | Expected |
|---|---|---|
| `runs/FORMAL_b21a97c1_seed11` | `NO_GO_BUDGET_PROTOCOL`, 6 rounds, 318 episodes, 7632 block steps | reproduce the seed 11 incident |
| `runs/FORMAL_2148a577_seed11` | `PASS_WITH_NOT_TESTED`, 10 rounds, 530 episodes, 12720 block steps | the rerun passes, with the log-hash check `INFERRED` |
| `runs/FORMAL_2148a577_seed53` | `NO_GO_RUN_FAILED`, 28 episodes | a crashed run, not a budget finding |
| `runs/FORMAL_2148a577_seed23` | `PASS_WITH_NOT_TESTED`, 10 rounds, 530 episodes | same shape as seed 11 |
| `runs/FORMAL_2148a577_seed37` | `PASS_WITH_NOT_TESTED`, 10 rounds, 530 episodes | same shape as seed 11 |

The auditor compares the run against the live contract, not against numbers
copied into the script. `finalize_run.ps1` refused a directory without
`run_manifest.json` (exit 13) and wrote
`reports/cc01/audit/FORMAL_2148a577_seed11.json` for the finished run.

## 6. PowerShell entrypoints — PASS

All five parse under `[Parser]::ParseFile` with no errors, and the ones that can
be exercised without training were run:

| Script | Check | Result |
|---|---|---|
| `preflight.ps1` | full run | python 3.12.7, Gurobi 13.0.2, contract ok, `GO_FORMAL_TRAINING`, exit 0 |
| `launch_seed.ps1` | no `-Seed` | PowerShell rejects the mandatory parameter, exit 1 |
| `launch_seed.ps1` | actual launch | **NOT_TESTED** — would start training, forbidden in this phase |
| `run_tests.ps1 -Level Unit` | run | 41 passed |
| `run_tests.ps1 -Level Full` | run | see section 8 |
| `audit_run.ps1` | missing directory | exit 12 |
| `audit_run.ps1` | real run id | prints evidence, exit 0 |
| `finalize_run.ps1` | missing directory | exit 12 |
| `finalize_run.ps1` | directory without manifest | exit 13 |

`launch_seed.ps1` is untested end to end on purpose. Its preflight half is the
`preflight.ps1` run above, and its launch half is the `--dry-run` command in
section 3, which the wrapper passes through unchanged.

## 7. Read-only agents — NOT_TESTED

`.claude/agents/independent-reviewer.md` and `.claude/agents/experiment-auditor.md`
exist and parse; their frontmatter declares `tools: Read, Grep, Glob` and no
Bash. Dispatching `experiment-auditor` from this session returned
`Agent type 'experiment-auditor' not found`, because agent definitions are read
when a session starts and this phase created them mid-session.

**What is verified:** the definition, the tool allowlist and the instruction
text. **What is not:** that the runtime honours the allowlist for these two
agents, and that the auditor reaches the right verdict on a real run. The
`src/audit_run.py` verdicts in section 5 are the same checks, executed by a
script, so the substance is covered; the read-only delivery channel is not.

**Cleared by:** a new session, then dispatching `experiment-auditor` on
`runs/FORMAL_2148a577_seed11` and confirming it cannot write.

## 8. Full test suite — PASS

```text
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_tests.ps1 -Level Full
======== 1404 passed, 53 deselected, 17 warnings in 421.28s (0:07:01) ========
exit 0
```

The 53 deselected are the `slow` marker exclusion in `pytest.ini`, unchanged by
this phase. The phase adds 30 tests: 12 in `tests/test_training_contract.py` and
18 in `tests/test_frozen_guard.py`.

## 9. Open items

| Item | Verdict | Why |
|---|---|---|
| `train.log` hash | `INFERRED` | frozen writer order, `TECHNICAL_DEBT.md` TD-01 |
| `launch_seed.ps1` end-to-end | `NOT_TESTED` | would start training |
| Agent dispatch | `NOT_TESTED` | definitions load at session start |
| Shell writes to frozen files | not covered | guard matches `Edit|Write` only, TD-03 |
