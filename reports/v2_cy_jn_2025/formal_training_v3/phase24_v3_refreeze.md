# Phase 24 — budget-guard repair and v3 re-freeze

Verdict: **PASS**, with three `NOT_TESTED` items and one environment `BLOCKED`
listed at the end.

## Hashes

| set | config hash | prefix | status |
|---|---|---|---|
| v3 (live) | `1b47e0ed0e5e469a510a77c6afb9b058aba3e41b90bd98734f40593645ffd2ff` | `1b47e0ed` | `finalized`, `GO_FORMAL_TRAINING`, `protocol_version: 3` |
| v2 (historical) | `2148a577…` | `2148a577` | unchanged, read-only |
| ready (superseded) | `b21a97c1…` | `b21a97c1` | unchanged, read-only |

Every v3 run id is `FORMAL_1b47e0ed_seed<N>`, so no v2 directory is touched.

## What changed

Code, in four places.

1. `src/train_da_gate.py`, guards. `--min-rounds` now defaults to
   `DEFAULT_MIN_ROUNDS = 10`; `--max-rounds` was added and is enforced twice —
   `chunk_end` is clamped to `max_rounds × round length`, and after a round's
   validation the round cap is checked before the patience rule. The cap check
   runs after validation so round 15 is still evaluated and can still take the
   best checkpoint.
2. `src/train_da_gate.py`, resume. `completed_rounds` is now a floor.
   `resume_position` returns the checkpoint's exact episode with the round's
   order offset; the loop computes `round_start = (round_idx - 1) × 53` every
   round and takes the pair from `pair_index`, so a partial round resumes at
   `order[28]` instead of restarting at `order[0]`. `resume_rule_checks` adds
   `early_stop_patience`, `min_rounds` and `max_rounds`; a missing key is a
   mismatch. `filter_evaluations` drops rows written after the checkpoint.
   `restore_best` prefers the manifest's `best`, then history plus snapshots,
   then the sentinel; the failure path now records `best` too.
3. `src/run_formal_training.py`. Passes `--max-rounds` from
   `exposures_per_day_max`; defaults to the v3 directory; builds its code
   payload from `make_formal_training_manifests.code_payload()` instead of its
   own copy (the TD-07 root cause).
4. `src/make_formal_training_manifests.py`, `src/training_contract.py`,
   `src/audit_run.py`, `.claude/hooks/protect_frozen.py`. `protocol_version: 3`;
   19 pinned sources (the 15 plus the launcher, contract, auditor and
   generator); 6 pinned tests/audits; the default config directory is v3; the
   hook resolves `(v3, v2, ready)` and reads source pins from v3 alone.

Tests and docs: `tests/test_train_da_gate_guards.py` (20), new
`tests/test_train_da_gate_resume.py` (8), `tests/test_training_contract.py`
(now pins v3 and the launcher command), `tests/test_frozen_guard.py` (v3
deny matrix), `scripts/run_tests.ps1`, `CLAUDE.md`,
`.claude/rules/training_contract.md`, `.claude/agents/experiment-auditor.md`,
`.claude/skills/training-preflight/SKILL.md`, `TECHNICAL_DEBT.md`
(TD-05/TD-07 closed, TD-08 added).

Nothing in the reward, observation, action, actor, critic, gate, quote map,
optimizer, exploration, replay, gamma, split, seeds, terminal SOC or clearing
engine changed.

## Verification

| # | check | verdict | evidence |
|---|---|---|---|
| 1 | `run_formal_training.py --check` | `PASS` | exit 0, `all_pass: true`, `budget_contract_ok: true`, `status: GO_FORMAL_TRAINING`, `config_sha256 1b47e0ed…` |
| 2 | `training_contract.py --check` | `PASS` | exit 0, `ok: true`, `violations: []` |
| 3 | guard suite | `PASS` | `pytest tests/test_train_da_gate_guards.py -q` → 20 passed |
| 4 | resume suite | `PASS` | `pytest tests/test_train_da_gate_resume.py -q` → 8 passed |
| 5 | contract + frozen guard | `PASS` | `pytest tests/test_training_contract.py tests/test_frozen_guard.py -q` → 41 passed |
| 6 | dry run | `PASS` | `--seed 11 --dry-run` prints `--exposures 15 --early-stop-patience 3 --min-rounds 10 --max-rounds 15 --max-episodes 795 --run-id FORMAL_1b47e0ed_seed11`; no run directory created |
| 7 | full suite | `PASS_WITH_FAILURE` | 1429 passed, 1 failed, 53 deselected in 925.58s; the failure is the environment item below |
| 8 | v2 / ready / runs / snapshots unchanged | `PASS` | no file under `runs/`, `snapshots/`, `formal_training_v2/` or `formal_training_ready/` has an mtime on or after 2026-11-05 (2050 / 67 / 12 / 12 files) |
| 9 | hook denies v3 paths, allows ordinary ones | `PASS` | direct hook invocation with absolute paths: deny on 4 pinned sources, the pinned resume test, v3 and v2 configs, `runs/FORMAL_1b47e0ed_seed11`, `snapshots/FORMAL_1b47e0ed_seed11`, `runs/FORMAL_2148a577_seed11`; allow on `reports/cc02/README.md` and `.claude/settings.json` |

Round-boundary arithmetic, probed directly:

| case | result |
|---|---|
| `resume_position(28, 53)` | `resume_episode 28`, `rounds_completed 0`, `round_start 0`, `partial_round True` |
| `resume_position(53, 53)` | `resume_episode 53`, `rounds_completed 1`, `round_start 53`, `partial_round False` |
| `pair_index(28, 53)` | 28 — the resumed partial round continues at `order[28]` |
| `pair_index(53, 53)` | 0 — the next round starts at `order[0]` |
| `should_early_stop(9, …)` / `(10, …)` with 3 non-improving rounds | `False` / `True` |
| `round_cap_reached(14, 15)` / `(15, 15)` / `(1, 0)` | `False` / `True` / `False` |
| `episode_budget(53, 15)` | 795 |

## v2 source pins are superseded — stated, not hidden

A `src/` path can satisfy one hash. v3 therefore reads source pins only from
its own code manifest, and `run_formal_training.py --check --config-dir
.../formal_training_v2` now reports `code_sources_match: false` and `NO_GO`.
That is the expected state of a historical protocol, recorded as TD-08. The v2
config, data manifest and run evidence are unchanged and are still read with
`--config-dir` / `--contract-dir`.

## Not tested

- **A live resume.** The exact-episode restart, the rule-mismatch refusal and
  the evaluation filter are covered by unit tests only. No run was started or
  resumed, so a real mid-round resume has no runtime evidence yet.
- **A live round-cap stop.** The cap is proven at the function level and by the
  clamp arithmetic; no run has reached round 15 under v3.
- **The launcher's success path.** `--dry-run` stops before `subprocess.run`;
  the wrapper's happy path still launches training.

## Blocked

- **One full-suite test fails on the environment.**
  `tests/test_formal_ensemble.py::test_every_consumer_reads_the_declared_comparison_set`
  fails at `src/make_rot_comparison_figures.py:28` with
  `module 'matplotlib' has no attribute 'use'`. `import matplotlib` in
  `energy_env` returns `__file__ = None`, so the package is a stub. It
  reproduces in isolation, imports no file changed here, and installing a
  plotting dependency was not authorised. Recorded, not worked around.

## Hook activation

`.claude/settings.json` was restored to the original `PreToolUse` entry. The
running session did not pick the entry up: a probe write into the v3 directory
succeeded through the `Write` tool (the file was removed immediately; the
directory listing is unchanged). The hook script itself denies the same path
when invoked directly, and the test suite drives it through stdin. The guard
becomes active in the session that starts after this one. Until then, the
protection is the rule, not the mechanism.

## Not done

No seed was launched, resumed or rerun. No v2 or ready file was modified. No
commit or push.
