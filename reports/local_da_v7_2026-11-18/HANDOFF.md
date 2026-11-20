# v7 formal launch: GO / NO-GO handoff

Protocol `v7`, candidate `V7_candidate_004`, config hash `0f35fed8`,
manifest `reports/v2_cy_jn_2025/formal_training_v7/`, re-freeze record
`reports/audit_remediation_2026-11-15/REFREEZE_V7_CONE_TOL.md`, which
supersedes `REFREEZE_V7_BUNDLE.md`, `REFREEZE_V7_DISJUNCTION.md` and
`REFREEZE_V7.md` in that order.

Nothing below authorises a launch. `run_formal_training.py --seed` refuses
while `status` is not `GO_FORMAL_TRAINING`, and it is `null`. Recorded
2026-11-19, updated 2026-11-20; no formal run exists under any v7 hash.

## Gate state

`PYTHONPATH=src python -B src/run_formal_training.py --config-dir
reports/v2_cy_jn_2025/formal_training_v7 --check`:

```
data_payload_matches true     budget_contract_ok true
code_sources_match true       library_counts true
code_tests_match true         all_disjoint true
config_hash_matches true      legacy_label_declared true
data_manifest_hash_matches true   h2_declared true
code_manifest_hash_matches true   all_pass true
status: null    contract_violations: []
```

The tree and its manifest agree. `training_readiness_status` is written only
by `--finalize`, from a `pretraining_validation_results.json` produced on the
v7 sources; that file does not exist, so no launch is authorised.

## Acceptance 1-8

Verdicts are the ones recorded in `reports/local_da_v7_2026-11-18/RESULTS.md`;
each cites the artifact it was read from there.

| # | Item | Verdict |
| --- | --- | --- |
| 1 | Payoffs finite, no missing clear, SOC in band, declared point honoured, one inventory term | PASS (three days, `smoke.json`) |
| 2 | Latency table at both windows | PASS as a measurement; the per-seed projection is INFERRED |
| 3 | Smoke flags on three days | PASS |
| 4 | Training-block price against the whole-day price | PASS (sign agreement 1.000, median gap 0.028-0.048) |
| 5 | Block-vs-day price gap | recorded, no criterion |
| 6 | Deviation weight | PASS (`gradient_share` 0.0429 at 0.05, criterion <= 0.50) |
| 7 | Feasibility selection | recorded: 90 train / 18 validation / 18 test / 10 holiday, no day dropped |
| 8 | Launcher end to end | `NOT_TESTED` (see below) |

## Price qualification, counted apart

Rows below are the **superseded** bilinear measurement. They are kept because
the re-measurement is what found the defect, not because they describe the tree.

`reports/local_da_v7_2026-11-18/qualification/summary.json`, 136 days:
`SMOOTH_PASS` 22, `SUBGRADIENT_PASS` 55, `NONSMOOTH_OR_FAILED` 59. Label-qualified
77; of those, 62 settle and 15 pass the label but fail the physical settlement
audit. The two labels are never summed into one "qualified" claim: a corner
price is a subgradient, not a unique derivative.

Before the remediation the count was 0 of 136 (108 suboptimal conditional
prices, 24 unsmooth probes).

**Re-measured on the disjunction tree, 2026-11-20**
(`reports/local_da_v7_2026-11-20/qualification/`): `SMOOTH_PASS` 19,
`SUBGRADIENT_PASS` 62, `NONSMOOTH_OR_FAILED` 55, label-qualified 81, and only
**38** settle. Twenty days that settled before stopped settling, all of them on
`cone_tightness_max_mw2` alone, with `ac_replay` passing to `1e-10` pu. The
cause was the barrier tolerance the disjunction refreeze moved from `1e-10` to
`1e-8`; `REFREEZE_V7_CONE_TOL.md` gives the cone check its own `1e-5` bound.
The 136-day count under that bound is `NOT_TESTED`.

## Cost per seed

Block steps per seed 21,600-32,400 (900-1,350 episodes of 24 blocks).

| Window | s/block | Hours per seed |
| --- | --- | --- |
| `ROLL_HORIZON=20` | 1.991 | 11.95 - 17.92 |
| 12-period fallback | 1.328 | 7.97 - 11.95 |

Recorded in the manifest as `wall_clock` with `status: recorded for the reader;
not a gate`; `training_contract_v7` has no latency criterion.

A block that reaches the 300 s solver limit spends all of it. A day whose 24
blocks all bind costs two hours, which no average above bounds.

## Memory and concurrency

The machine reports 15.6 GB with ~3.5 GB free. The launcher smoke was killed
by the harness for low system memory while the session was idle, so one seed
at a time, foreground, in day-sized pieces. Two seeds in parallel is not
advisable on this machine.

## What the disjunction changed

The exact model now carries a binary charge/discharge disjunction instead of a
bilinear complementarity equality, and a training window stops at the declared
1% gap instead of closing it to 1e-6. The record is
`reports/audit_remediation_2026-11-15/REFREEZE_V7_DISJUNCTION.md`.

Measured on the days that used to abort (`2021-06-13`, `2021-06-04`, seeded
policy, 24 blocks each): both episodes finished in 48 s and 59 s, no block
reached the time limit, and every block cleared. The same day on the bilinear
encoding aborted at block 5 after 300 s with no incumbent.

`gap_solved_blocks` therefore reads 24 on a cleared training day by design: it
counts the blocks whose payoff is known only to the declared 1%, not a rare
fallback.

## What is not established

| item | verdict | reason |
| --- | --- | --- |
| the launcher on a full round | `PASS` for the trainer path, `NOT_TESTED` for the gate | `runs/SMOKE_local_da_v7_bundle` cleared two training episodes, measured one validation round over two days, and wrote `eval_history.json`, `snapshots/SMOKE_local_da_v7_bundle/round_001.json`, the best checkpoint and the final bundle; `load_checkpoint` reads that bundle back. The first attempt, `runs/SMOKE_local_da_v7_round`, completed the same round but died writing the bundle on a schema mismatch (`REFREEZE_V7_BUNDLE.md`). `run_formal_training.py --seed` itself is still unexercised: it refuses at the `status` gate before reaching the trainer. |
| the launcher smoke that was killed | `FAIL` | `runs/SMOKE_local_da_v7_gap` cleared both of its training episodes (48 s and 59 s) and then entered the first validation pass, where the process was killed for low memory. The launcher logs a round only once it is measured, so the run ends at episode 2 with `eval_history.json`, the round snapshot, the best checkpoint and the final bundle absent and `run_manifest.json` still at `stopped: null`, `best.value: -inf`. That the kill fell inside validation is read from that absence plus the harness notification, not from a log line. |
| a cleared episode | `PASS` | `runs/SMOKE_local_da_v7_gap/episodes.csv` has two rows with status `cleared` over the two training days, finite per-agent rewards and `gap_solved_blocks = 24` on each; `total_steps` is cumulative (24, then 48), so each episode is 24 blocks. |
| the wall-clock projection | `INFERRED` | Extrapolated from the latency table; no formal run has measured it. |
| `pretraining_validation_results.json` | `BLOCKED` | It describes the sources it was produced on; the v6 file describes v6. Producing one is the only thing that sets `status`. |

## What a GO would require

1. The launcher completing one full round on a reduced day set, under
   `--seconds 300`, writing `eval_history.json`, a round snapshot, a best
   checkpoint and a final bundle. **Met on 2026-11-20** by
   `runs/SMOKE_local_da_v7_bundle`; see `REFREEZE_V7_BUNDLE.md` for the schema
   mismatch the first attempt hit and the re-freeze that cleared it.
2. `pretraining_validation_results.json` produced on the v7 sources, so
   `--finalize` can set `status: GO_FORMAL_TRAINING`. No producer exists for
   it: `tools/make_pretraining_validation.py` reads the retired Phase 20-22
   artifacts under `reports/v2_cy_jn_2025/p21_candidates/` and the
   `formal_training_ready` directory, and its output describes that stack.
   A v7 producer has to assemble this directory's acceptance evidence
   (`RESULTS.md`, `qualification/summary.json`, `smoke.json`, the audit
   summary) into one record whose `all_pass` is computed from those items,
   and `make_formal_training_manifests.py --finalize` then reads it.
3. The full suite green apart from the known pre-existing failure. The last
   completed full run was on the disjunction tree before two of its tests were
   reconciled: `python -m pytest tests/ -q` gave `3 failed, 1577 passed, 53
   deselected` in 1051 s. One of those three is
   `test_da_market_system::test_standard_cone_matches_rotated_form_and_ac`,
   which fails at HEAD and touches no file this protocol pins. The other two
   asserted the retired encoding — no integer variables in the dispatch, and a
   1e-10 barrier on the exact model — and were rewritten, together with the
   re-freeze record, in the commit named below. The four files those tests live
   in then ran `40 passed, 1 failed` (the pre-existing failure).
   **Measured on the current tree, 2026-11-20**:
   `python -m pytest tests/ -q` gave `2 failed, 1578 passed, 53 deselected` in
   1076 s. One failure is the known one above. The other,
   `test_formal_ensemble::test_every_consumer_reads_the_declared_comparison_set`,
   is the interpreter and not the tree: `energy_env`'s `matplotlib` is a
   namespace package with no `use` attribute, and the same test passes under
   `C:/ProgramData/anaconda3/python.exe` (`1 passed`). A full run under that
   interpreter has not been done, so "green apart from the known failure" holds
   for every test the `energy_env` run could execute.

## Where the tree stands at this stop

Committed:

- the disjunction and the gap hand-off (`local_da_clearing.py`, `rl_da_env.py`),
  `4cfba8a`;
- the regenerated v7 manifests and the re-freeze record, `a707e2d`;
- this handoff's corrected rows, `8c5d721`;
- the two reconciled tests and the record paragraph naming them, `8d825cd`.

Re-verified after the test edits: the gate still reports `all_pass: true`,
`config_sha256: 8db03638…`, `status: null`; the pinned file set was not
touched by them.

The bundle-slot re-freeze (`train_local_da_v7.py`,
`make_formal_training_manifests.py`), the regenerated manifests under
`b8b61d8d` and `REFREEZE_V7_BUNDLE.md` are committed; the gate performs all
eight of its checks on that tree with `status: null`.

Stopped, not done:

- the full-suite run that would confirm the reconciled tree (killed on
  request, not by memory pressure);
- any round beyond the two-day smoke; the formal launch itself;
- `tools/run_da_market.py` and `tools/run_candidate_q050.py` remain untracked
  pending a check that they write no frozen directory;
- `tmp/manifest_superseded/` holds five abandoned manifest directories that
  await an owner's decision to delete.

Four tracked files were already modified when this work started and were left
untouched by it: `docs/2026-09-18-matd3-smoke.md`,
`docs/2026-09-20-external-wholesale-price.md`, `tools/check_checkpoint_obs.py`
and `tools/demo_snapshot.py`. They carry no part of this protocol.

Recommendation: **NO-GO**. Item 1 is met: the trainer carried a day-sized
round and wrote every artifact, including a bundle that loads. What still
blocks a launch is item 2 — no v7 producer for
`pretraining_validation_results.json` exists, so `status` stays `null`.
