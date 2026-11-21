# Phase CC-01 — engineering framework upgrade

Status: `CC01_PASS_FROZEN_CODE_PRESERVED`

One acceptance item is `NOT_TESTED`: the two read-only agents were created in
this session, so the runtime does not register them until a session starts.
Their definitions parse and declare `Read, Grep, Glob`; the checks they would
judge are the same ones `src/audit_run.py` already runs. See
`validation_report.md` section 7.

This phase turns the formal-training rules that previously lived only in prose
into executable checks, and adds a read-only audit path that is separate from the
session that produced the work. It does not change the training algorithm, the
reward, the observation, the action space, the gate, the critic, the actor, the
optimizer or the budget semantics.

## Rulings

| Question | Ruling |
|---|---|
| Freeze boundary | `OPTION_1_NO_MODIFICATION_TO_FROZEN_TRAINING_CODE` — none of the 15 frozen sources, the 3 pinned tests/audits, the existing manifests, checkpoints, replay buffers or logs were modified |
| Live manifest directory | `OPTION_1` — `formal_training_v2` is the default; `formal_training_ready` is `SUPERSEDED_READ_ONLY` |
| PowerShell entrypoints | `OPTION_2_FIVE_POWERSHELL_ENTRYPOINTS` |
| `.claude` granularity | `OPTION_2_MEDIUM_SCOPE` — 4 rules, 2 skills, 2 agents, no permanent implementer agent |
| Hook scope | `OPTION_1_PRETOOLUSE_WRITE_PROTECTION_ONLY` — `Edit|Write` only, no Bash interception |
| Superseded manifest in hook | Protected as well |
| Deliverables | Merged into the five files listed below |
| Agent permissions | Read-only allowlist — `Read, Grep, Glob`, no Bash |

No new training was started. No git commit or push was made in this phase.

## What changed

| Path | Kind | Purpose |
|---|---|---|
| `src/training_contract.py` | new, unfrozen | Executable budget and data invariants; `--check` exits 1 on any violation |
| `tests/test_training_contract.py` | new | One test per named violation plus the CLI exit code |
| `src/run_formal_training.py` | modified, unfrozen | Default manifest directory moved to v2; `budget_contract_ok` in `verify()`; `--max-episodes` on the launch command |
| `src/make_formal_training_manifests.py` | modified, unfrozen | Writes v2 by default, refuses to overwrite an existing manifest file, `--check` no longer creates the directory, writes `manifest_generation.json` |
| `src/audit_run.py` | new, unfrozen | Read-only evidence collector for one run directory |
| `.claude/hooks/protect_frozen.py` | new | PreToolUse guard; protected set derived from the manifests at call time |
| `.claude/settings.json` | new | Registers the guard on `Edit|Write` |
| `tests/test_frozen_guard.py` | new | Allow/deny matrix for the guard, 18 tests |
| `.claude/rules/*.md` | new (4) | Training contract, data governance, model constraints, reporting |
| `.claude/skills/*/SKILL.md` | new (2) | `training-preflight`, `experiment-audit` |
| `.claude/agents/*.md` | new (2) | `independent-reviewer`, `experiment-auditor` |
| `scripts/*.ps1` | new (5) | `preflight`, `launch_seed`, `run_tests`, `audit_run`, `finalize_run` |
| `CLAUDE.md` | modified | Index of the above plus `@` imports of the four rules |

## Role matrix

| Role | May do | May not do |
|---|---|---|
| Implementing session | Write unfrozen code, run scripts, produce evidence files | Edit a frozen file; start training in this phase |
| `independent-reviewer` | Read code, tests, hashes, reports; accept or reject | Write anything; run any command |
| `experiment-auditor` | Read a run directory and the evidence written by `audit_run.py`; issue a verdict | Write anything; run any command |
| `protect_frozen.py` | Deny `Edit`/`Write` on the frozen set | Judge intent; intercept shells |

The scripts produce the evidence (hashes, directory listings, pytest output);
the agents only read it. That split is what keeps the review independent of the
session that wrote the code.

## Config audit

The live manifest emits `GO_FORMAL_TRAINING` with config hash `2148a577`; the
superseded one emits `b21a97c1` and is refused by default. `training_contract`
re-derives every budget number from `minimum_rounds`, `train_days` and
`episode.blocks`, so a hand-edited total can no longer disagree with its own
derivation.

| Field | Value | Derived from |
|---|---|---|
| `budget.seeds` | 11, 23, 37, 53, 71 | — |
| `budget.minimum_rounds` | 10 | — |
| `budget.exposures_per_day_min` | 10 | equals `minimum_rounds` |
| `budget.exposures_per_day_max` | 15 | — |
| `budget.episodes_per_seed_min` | 530 | `10 × 53` train days |
| `budget.episodes_per_seed_max` | 795 | `15 × 53` |
| `budget.block_steps_per_seed_min` | 12720 | `530 × 24` |
| `budget.block_steps_per_seed_max` | 19080 | `795 × 24` |
| `validation.early_stop_patience` | 3 | must be ≥ 1 |

Run state on disk at the end of the phase, as reported by `audit_run.py`:

| Run | Status | Verdict |
|---|---|---|
| `runs/FORMAL_2148a577_seed11` | `completed_early_stop`, 10 rounds, 530 episodes | `PASS_WITH_NOT_TESTED` (log hash, see `TECHNICAL_DEBT.md`) |
| `runs/FORMAL_2148a577_seed23` | `completed_early_stop`, 10 rounds, 530 episodes | `PASS_WITH_NOT_TESTED` (log hash) |
| `runs/FORMAL_2148a577_seed37` | `completed_early_stop`, 10 rounds, 530 episodes | `PASS_WITH_NOT_TESTED` (log hash) |
| `runs/FORMAL_2148a577_seed53` | `failed` — clearing failure at episode 28 | `NO_GO_RUN_FAILED` |
| `runs/FORMAL_b21a97c1_seed11` | `completed_early_stop`, 6 rounds, 318 episodes | `NO_GO_BUDGET_PROTOCOL` |
| seed 71 | not launched | — |

Seed 53 did not complete. Its directory is frozen evidence and must stay as it
is; its verdict is a failed run, not a budget finding.

## Related

- `reports/cc01/validation_report.md` — evidence per acceptance item.
- `reports/cc01/frozen_audit.json` — before/after SHA256 of the frozen set.
- `reports/cc01/manifest_directory_audit.md` — v2 versus superseded directory.
- `TECHNICAL_DEBT.md` — open defects, with the condition that clears them.
