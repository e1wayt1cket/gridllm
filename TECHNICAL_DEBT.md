# Technical debt

Open defects that are known, deliberately not fixed, and recorded here so they
are not rediscovered as surprises. Each entry states what clears it. Nothing in
this file authorises editing a frozen file.

## TD-01 — `train.log` hash is taken before the completion line is written

**Status:** closed in protocol v4 (`7834d0aa`). Both the success and the pause
paths now write the log line and flush before `artifact_hashes(run_dir)` runs
(`src/train_da_gate.py`, marked `# TD-01`), so the recorded hash covers the
completion line. The v3 and older run manifests keep their `INFERRED` log
hashes as recorded history. The text below records the pre-v4 behaviour.

**Where:** `src/train_da_gate.py`, in the success path of `main()`. The order is
`log_fh.flush()` → `artifact_hashes(run_dir)` → `run_manifest.json` written →
`log("completed: ...")`. The final log line lands in `train.log` after its hash
was already recorded, so `artifacts_sha256["train.log"]` never matches the file
on disk.

**Evidence:** every completed run. In
`runs/FORMAL_2148a577_seed11/run_manifest.json`, the five data artifacts match
their recorded hashes byte for byte and `train.log` alone differs.

**Effect:** `src/audit_run.py` cannot assert the log hash, so it reports
`train_log_hash_matches` as `INFERRED` and the run verdict as
`PASS_WITH_NOT_TESTED` rather than `PASS`. Nothing else is affected: the log is
an append-only transcript and the checkpoints, replay buffer, episode records,
gate statistics and evaluation history all verify.

**Do not** hash-patch the manifest or rewrite the log to make the check pass.
Either would destroy the evidence the check exists to protect.

**Cleared by:** moving the `log("completed: ...")` call above
`artifact_hashes(...)`, or by hashing the log after the completion line and
writing the manifest last. Both are edits to a frozen file and require an
explicit re-freeze: a new manifest directory, a new config hash, and a new set
of runs. Until then the auditor keeps reporting `INFERRED`.

## TD-02 — one clearing failure aborts the whole seed

**Status:** closed in protocol v4 (`7834d0aa`). A failed block raises
`EpisodeClearingError` (`src/rl_da_gate.py`): the episode is atomic — no block
transition reaches the buffer before the day completes — so the training loop
catches it and skips the episode (`src/train_da_gate.py`, `except
mod.EpisodeClearingError`) instead of aborting the seed. The text below
records the pre-v4 behaviour.

**Where:** `src/rl_da_gate.py`, `train_day`, raising
`RuntimeError: <day>|<price-day> block 23: clearing failed; training stops`.

**Evidence:** `runs/FORMAL_2148a577_seed53` stopped at episode 28 of a minimum
of 530. Its verdict is `NO_GO_RUN_FAILED`, not a budget finding.

**Effect:** a single infeasible clearing discards the seed. There is no retry,
no per-day skip, and no partial-run evidence beyond the diagnostic checkpoint
saved by the failure path.

**Cleared by:** a protocol decision on what a failed clearing should do, made
outside the frozen budget contract. It is not a bug fix to apply while training
evidence is being collected.

## TD-03 — the write guard covers `Edit` and `Write` only

**Where:** `.claude/settings.json` matches `Edit|Write`. A shell redirect, a
Python script or `git checkout` can still reach a frozen file.

**Effect:** the guard stops the accidental edit path, which is the one that
loses evidence. It is not a sandbox.

**Cleared by:** nothing in this phase. The ruling was
`OPTION_1_PRETOOLUSE_WRITE_PROTECTION_ONLY` and adding shell interception was
explicitly out of scope. Revisit only if a frozen file is ever damaged through a
non-`Edit`/`Write` path.

## TD-04 — `run_tests.ps1 -Level Integration` names its files

**Where:** `scripts/run_tests.ps1`. The `Unit` and `Integration` levels list test
files explicitly, so a renamed or newly added test file silently drops out of
that level.

**Effect:** `Full` runs everything and is the level used for acceptance, so this
cannot hide a regression from a full run. The narrower levels are conveniences.

**Cleared by:** moving the selection to pytest markers if these levels start
being used as gates.

## TD-05 — resume re-reads the stop rule from the CLI and skips a partial round

**Status:** closed in protocol v3 (`1b47e0ed`). `--min-rounds` defaults to 10,
the resume checks `early_stop_patience` / `min_rounds` / `max_rounds` and
refuses a missing key, and a resume restarts at the checkpoint's exact episode
with the per-round order offset preserved. The text below records the v2
behaviour and the runs that carry it.

**Where:** `src/train_da_gate.py`. The resume `checks` dict (lines 585–602)
compares seed, data hashes, market/b1 hashes, split, exposures, budget, train
subset, action schema, quote map and source hashes against
`run_manifest.json`, but not `early_stop_patience` or `min_rounds`. Both are
taken from `args` (lines 891–893), and `--min-rounds` defaults to 0 (line 422).
Separately, `completed_rounds` (line 106) is a ceiling, so `resume_from
["resume_episode"] = completed_rounds(episodes, len(order)) * len(order)`
rounds a partial round up.

**Evidence:** seed 53's checkpoint records 28 episodes with `order = 53`, so
`completed_rounds(28, 53) == 1` and a resume would restart at episode 53. The
25 training days belonging to episodes 28–52 are never exposed in round 0, so
they would finish with 14 exposures instead of the declared 15. The runtime
probe and the guard matrix are in `reports/cc02/phase24_verdict.md`.

**Effect:** a resume is not guaranteed to run under the same stop rule as the
run it continues. Without `--min-rounds`, early stopping is permitted from the
first validation round. The exposure shortfall stays above the contract
minimum of 10, so it does not by itself break the budget rule, but the runs it
produces are not identical in form to a fresh run.

**Cleared by:** adding both fields to the resume `checks` dict and replacing
the ceiling with an explicit decision — either resume the partial round from
its exact episode, or refuse to resume and require a new run directory. Both
edit a pinned file and need a new candidate id, a new manifest and a re-freeze.

## TD-06 — `audit_phase22_resume.py` writes into the superseded manifest directory

**Where:** `src/audit_phase22_resume.py:20`, `OUT` resolves to
`reports/v2_cy_jn_2025/formal_training_ready/phase22_resume.json`. That
directory is `SUPERSEDED_READ_ONLY` and is part of the frozen set, and
`line 110` creates the parent directory before writing.

**Effect:** running the script as written writes into a frozen directory,
which the training contract forbids. It also emits a report about the
superseded config (`b21a97c1`), so its result would not describe the live
protocol.

**Cleared by:** repointing `OUT` at a new path under the live config, or
deleting the script once no audit needs the superseded run. It was not run in
CC-02.

## TD-07 — the formal launcher is not pinned and the recorded launches omit `--max-episodes`

**Status:** closed in protocol v3. `run_formal_training.py`, the contract, the
auditor and the manifest generator are pinned, the launcher reuses the
generator's `code_payload()` instead of its own copy, and it passes
`--min-rounds 10 --max-rounds 15 --exposures 15 --max-episodes 795`. The v2
launch logs remain as recorded history and are not re-launched.

**Where:** `src/run_formal_training.py` builds the training command and is not
one of the 15 `source_sha256` entries in the live code manifest. The frozen
rule (`.claude/rules/training_contract.md`, "Early stopping") requires the
launcher to pass `--min-rounds`, `--exposures` and `--max-episodes`, and says a
command omitting any of them "is not a formal launch".

**Evidence:** `formal_training_v2/seed11_launch.log` and
`formal_training_v2/remaining_seeds_launch.log` record commands with
`--exposures 15 --early-stop-patience 3 --min-rounds 10` and no
`--max-episodes`. The current `command()` does pass `--max-episodes 795`, as
`--seed 71 --dry-run` prints. Full record in
`reports/cc02/launcher_verification.md`.

**Effect:** the minimum-round floor that the contract relies on is supplied by
an unpinned file, so the guard's floor is only as stable as that file, and the
recorded runs satisfy the budget while not matching the written launch form.
Numerically the omission was inert (53 × 15 = 795 = the frozen maximum), so the
three completed runs are within budget.

**Cleared by:** adding `run_formal_training.py` to the pinned set with a new
manifest, and deciding explicitly whether the recorded launches need to be
re-launched under the letter of the rule. That is a protocol decision, not a
code fix.

## TD-08 — v2 source pins are superseded by v3

**Where:** `reports/v2_cy_jn_2025/formal_training_v2/formal_training_code_manifest.json`.

**Effect:** one `src/` path can satisfy only one hash, so v2's `source_sha256`
no longer matches the working tree. `run_formal_training.py --check
--config-dir .../formal_training_v2` therefore reports `all_pass: false` and
`NO_GO`, and `src/audit_run.py --contract-dir .../formal_training_v2` still
validates the budget because the config and data manifests are unchanged.
This is the expected state of a historical protocol, not drift.

**Cleared by:** nothing. Read v2 run evidence with its own config; add a new
version directory for any future protocol change.

## TD-09 — a launch killed before the first episode leaves an unresumable run directory

**Where:** `src/train_da_gate.py` creates `runs/<run_id>/`, `eval_history.json`
and `train.log` before the first episode. A checkpoint is written only at the
end of a round or by the failure handler, so a process killed earlier leaves a
directory with no `manifest.json`.

**Evidence:** 2026-11-05, `runs/FORMAL_1b47e0ed_seed11/` was created at 21:51
and the process was killed by the harness before episode 1. The directory holds
`eval_history.json` (`[]`) and an empty `train.log`; it has no checkpoint, so
`--resume` raises `no checkpoint manifest to resume` and a fresh launch refuses
because the directory is not empty.

**Effect:** neither the "new run directory" rule nor the resume path can clear
this state. The only recovery is removing the directory, which the contract
reserves as a refusal rather than an instruction.

**Cleared by:** a protocol decision on empty-launch directories — either an
explicit `--discard-empty-run` that only deletes a directory with no checkpoint
and no episodes, or a documented ruling that a directory with zero episodes may
be removed. Do not delete by hand while the rule stands.

## TD-10 — the interpreter is not pinned or recorded

**Status:** half closed in protocol v4 (`7834d0aa`). `run_manifest.json` now
records `interpreter.executable` and `interpreter.version` (marked `# TD-10`),
so two seeds under different runtimes are visible in the evidence. The resume
refusal on an interpreter mismatch is **not** implemented — the stop-rule
check still covers only patience and the round bounds.

**Where:** `src/run_formal_training.py::command` uses `sys.executable`, so the
launch runs under whichever `python` resolves on `PATH`; nothing records the
interpreter or the package versions in `run_manifest.json`.

**Evidence:** the v3 seed 11 launch printed
`C:\ProgramData\anaconda3\python.exe` (Python 3.12.7, torch 2.5.1), while the
project's other interpreter `C:\Users\26036\.conda\envs\energy_env` is Python
3.10.20 with torch 2.11.0. Both import torch and gurobipy.

**Effect:** two seeds launched from different shells could train under
different runtimes and nothing would notice. The manifests pin source hashes,
not the environment.

**Cleared by:** recording `sys.version`, `torch.__version__` and
`gurobipy.gurobi.version()` in `run_manifest.json`, and refusing a resume whose
interpreter differs. Both are edits to pinned files and need the next version
directory.

## TD-11 — the formal launcher has no resume path

**Status:** closed in protocol v4 (`7834d0aa`). The launcher takes `--resume`
and re-emits the frozen command with `--resume` appended
(`src/run_formal_training.py`), so a stopped seed continues under the pinned
launcher rather than a hand-typed trainer command. The text below records the
pre-v4 behaviour.

**Where:** `src/run_formal_training.py` refuses a seed whose run directory
exists, and passes no `--resume`. Continuing a stopped run therefore has to
call `src/train_da_gate.py` by hand, outside the pinned launcher that supplies
the frozen stop rule.

**Evidence:** 2026-11-06, `runs/FORMAL_1b47e0ed_seed11` held a round-4
checkpoint and 221 episodes after the process was killed for memory. The only
continue path is the raw trainer command with `--resume`, typed by whoever
resumes it.

**Effect:** nothing enforces that a manual resume passes `--min-rounds`,
`--max-rounds`, `--exposures` and `--max-episodes`. The trainer refuses a
resume whose stop rule differs from `run_manifest.json`, which catches a wrong
value, but not an operator who reads the values from the right place by hand.

**Cleared by:** a `--resume` flag on the launcher that re-emits the frozen
command with `--resume` appended. It is an edit to a pinned file and needs the
next version directory.

## TD-12 — the corrected radial tree invalidates the frozen price reference

**Status:** open. Blocks any v5 protocol that keeps `OBS_V6` unchanged.

**Where:** `src/dispatch_core._select_radial_lines`, now restricted to
in-service lines; `src/rl_env.py`, `PRICE_REF_CNY_MWH = 288.19757259048833`
and the `[-1, 3]` clip on the price-history features.

**Evidence:** before the correction the selector admitted the four open tie
lines of `case33bw` (32, 33, 34, 36) and dropped four main segments (6, 9, 13,
26). On the in-service tree, `baseline` clears at SOCP price mean 1544.0
against 726.7 before (LMP spatial spread 700.3 against 45.5). Both
`last_lmp_norm` and `avg_lmp_hist_norm` are `lmp / 288.197...` clipped at 3.0,
so the whole day now saturates and the observation carries no price level.
`tests/test_rl_spec.py::test_a_blocks_realized_lmp_reaches_only_the_next_observation`
is marked `xfail(strict=True)` for this reason.

**Effect:** every clearing number produced before the correction — the
`JN-Base` benchmark tables, `b1_terminal_neutral_results.csv`, the training
readiness headroom — describes a network that is not the case's. The
correction is right and the numbers have to be recomputed. The price reference
cannot be recomputed in place: it is part of the frozen observation
definition and needs a new candidate id and a new manifest.

**Cleared by:** re-deriving `PRICE_REF_CNY_MWH` (and the clip) for the v5
candidate, regenerating the benchmark and baseline tables on the corrected
tree, and removing the `xfail` marker.
