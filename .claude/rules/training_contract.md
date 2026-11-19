# Training contract

Applies to every formal-training launch, resume, re-freeze and audit.

## Live manifest

- `reports/v2_cy_jn_2025/formal_training_v7/` is the only live formal manifest
  set (config hash `8db03638`, `protocol_version: 7`,
  `V7_candidate_002`; re-freeze records in
  `reports/audit_remediation_2026-11-15/REFREEZE_V7.md` and
  `REFREEZE_V7_DISJUNCTION.md`, the latter superseding the former).
- `reports/v2_cy_jn_2025/formal_training_v6/` (`ac0f78d2`) is historical and
  read-only, superseded on the same terms as v5 and older below.
- `reports/v2_cy_jn_2025/formal_training_v5/` (`6e32398b`) is historical and
  read-only, superseded on the same terms as v4 and v3 below.
- `reports/v2_cy_jn_2025/formal_training_v4/` (`7834d0aa`) and
  `reports/v2_cy_jn_2025/formal_training_v3/` (`1b47e0ed`) are historical and
  read-only. Their runs remain frozen evidence. Their source pins are
  superseded: a `src/` path can satisfy only the live manifest's hash, so
  `--config-dir` against them reports `code_sources_match: false` by design.
- `reports/v2_cy_jn_2025/formal_training_v2/` (`2148a577`) is historical and
  read-only, superseded on the same terms.
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
- v6: terminal SOC starts and ends at the same value and is applied only in the
  day-end window
- v7: `terminal_soc.mode: free` with a declared initial value; the endpoint is
  valued by the settlement's inventory term and never pinned

The frozen run for v6 is 10–15 rounds, 530–795 episodes, 12,720–19,080 block
steps per seed. The frozen run for v7 is 10–15 rounds, 900–1350 episodes,
21,600–32,400 block steps per seed over the 90-day training block.

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
- v6 runs get new run ids (`FORMAL_ac0f78d2_seed<N>`) and do not overwrite
  any earlier protocol's runs.
- v7 runs get new run ids (`FORMAL_<config>_seed<N>` under its own config
  hash) on the same terms; no v7 run id may be a v6 one.
- A new seed requires a new run directory. A populated run directory is a
  refusal, not a prompt to delete.
- Best and final checkpoints are distinct artifacts. `best` is the highest
  validation mean fleet delta G; `final` is the end of budget or early stop.
- `checkpoint`, replay buffer, `run_manifest.json` and `train.log` are
  evidence. They are never regenerated to make a check pass.

## Enforcement

- `src/training_contract.py` implements the invariants above for v6 and
  `src/training_contract_v7.py` for v7; `src/run_formal_training.py` selects
  the one the manifest's `protocol_version` names.
- `src/run_formal_training.py --check` is the gate; it must return
  `all_pass: true` and `budget_contract_ok: true`.
- `.claude/hooks/protect_frozen.py` denies Edit/Write on the frozen set.
- On any violation: stop, report the named violation, and either create a new
  version directory or request an explicit re-freeze. Never edit the frozen
  file to silence the check.
