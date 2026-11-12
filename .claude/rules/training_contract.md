# Training contract

Applies to every formal-training launch, resume, re-freeze and audit.

## Live manifest

- `reports/v2_cy_jn_2025/formal_training_v3/` is the only live formal manifest
  set (config hash `1b47e0ed`, `protocol_version: 3`).
- `reports/v2_cy_jn_2025/formal_training_v2/` (`2148a577`) is historical and
  read-only. Its runs remain frozen evidence. Its source pins are superseded:
  a `src/` path can satisfy only the live manifest's hash, so `--config-dir`
  against v2 reports `code_sources_match: false` by design.
- `reports/v2_cy_jn_2025/formal_training_ready/` is `SUPERSEDED_READ_ONLY`
  (`b21a97c1`). It may be read for historical audit with an explicit
  `--config-dir`; it is never the default and never modified.

## Budget invariants

With `train_days = data_manifest.counts.train`:

- `minimum_rounds >= 1`
- `minimum_rounds == exposures_per_day_min`
- `minimum_rounds <= exposures_per_day_max`
- `episodes_per_seed_min == minimum_rounds * train_days`
- `episodes_per_seed_max == exposures_per_day_max * train_days`
- `block_steps_per_seed_min == episodes_per_seed_min * episode.blocks`
- `block_steps_per_seed_max == episodes_per_seed_max * episode.blocks`
- `early_stop_patience >= 1`
- `seeds` is non-empty and has no duplicates
- terminal SOC starts and ends at the same value and is applied only in the
  day-end window

The frozen run for the current protocol is 10–15 rounds, 530–795 episodes,
12,720–19,080 block steps per seed.

## Early stopping

Early stopping is allowed only after `minimum_rounds` completed validation
rounds; a partial round does not satisfy the floor. No run may exceed
`exposures_per_day_max` rounds. The launcher must pass `--min-rounds`,
`--max-rounds`, `--exposures` and `--max-episodes` from the frozen config; a
command that omits any of them is not a formal launch. `--min-rounds` defaults
to 10 and `--max-rounds` defaults to unlimited, so both must be passed
explicitly to be relied on.

A resume restores the stop rule rather than re-reading it: `early_stop_patience`,
`min_rounds` and `max_rounds` are compared against `run_manifest.json` and a
missing or different value refuses the resume. It restarts at the checkpoint's
exact episode, so a partial round keeps its remaining days.

## Seeds and checkpoints

- The v2 runs `FORMAL_2148a577_seed{11,23,37,53}` are read-only: never rerun,
  resume over, rename, migrate or overwrite. Seed 53 failed at episode 28; a
  failed run is still evidence.
- v3 runs get new run ids (`FORMAL_1b47e0ed_seed<N>`) and do not overwrite
  them.
- A new seed requires a new run directory. A populated run directory is a
  refusal, not a prompt to delete.
- Best and final checkpoints are distinct artifacts. `best` is the highest
  validation mean fleet delta G; `final` is the end of budget or early stop.
- `checkpoint`, replay buffer, `run_manifest.json` and `train.log` are
  evidence. They are never regenerated to make a check pass.

## Enforcement

- `src/training_contract.py` implements the invariants above.
- `src/run_formal_training.py --check` is the gate; it must return
  `all_pass: true` and `budget_contract_ok: true`.
- `.claude/hooks/protect_frozen.py` denies Edit/Write on the frozen set.
- On any violation: stop, report the named violation, and either create a new
  version directory or request an explicit re-freeze. Never edit the frozen
  file to silence the check.
