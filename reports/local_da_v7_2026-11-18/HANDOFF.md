# v7 formal launch: GO / NO-GO handoff

Protocol `v7`, candidate `V7_candidate_002`, config hash `8db03638`,
manifest `reports/v2_cy_jn_2025/formal_training_v7/`, re-freeze record
`reports/audit_remediation_2026-11-15/REFREEZE_V7_DISJUNCTION.md`, which
supersedes `REFREEZE_V7.md`.

Nothing below authorises a launch. `run_formal_training.py --seed` refuses
while `status` is not `GO_FORMAL_TRAINING`, and it is `null`.

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

`reports/local_da_v7_2026-11-18/qualification/summary.json`, 136 days:
`SMOOTH_PASS` 22, `SUBGRADIENT_PASS` 55, `NONSMOOTH_OR_FAILED` 59. Label-qualified
77; of those, 62 settle and 15 pass the label but fail the physical settlement
audit. The two labels are never summed into one "qualified" claim: a corner
price is a subgradient, not a unique derivative.

Before the remediation the count was 0 of 136 (108 suboptimal conditional
prices, 24 unsmooth probes).

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
| the launcher on a full round | `NOT_TESTED` | `runs/SMOKE_local_da_v7_gap` cleared both of its training episodes (48 s and 59 s) and then entered the first validation pass, where the process was killed for low memory. The launcher logs a round only once it is measured, so the run ends at episode 2 with `eval_history.json`, the round snapshot, the best checkpoint and the final bundle absent and `run_manifest.json` still at `stopped: null`, `best.value: -inf`. That the kill fell inside validation is read from that absence plus the harness notification, not from a log line. |
| a cleared episode | `PASS` | `runs/SMOKE_local_da_v7_gap/episodes.csv` has two rows with status `cleared` over the two training days, finite per-agent rewards and `gap_solved_blocks = 24` on each; `total_steps` is cumulative (24, then 48), so each episode is 24 blocks. |
| the wall-clock projection | `INFERRED` | Extrapolated from the latency table; no formal run has measured it. |
| `pretraining_validation_results.json` | `BLOCKED` | It describes the sources it was produced on; the v6 file describes v6. Producing one is the only thing that sets `status`. |

## What a GO would require

1. The launcher completing one full round on a reduced day set, under
   `--seconds 300`, writing `eval_history.json`, a round snapshot, a best
   checkpoint and a final bundle.
2. `pretraining_validation_results.json` produced on the v7 sources, so
   `--finalize` can set `status: GO_FORMAL_TRAINING`. No producer exists for
   it: `tools/make_pretraining_validation.py` reads the retired Phase 20-22
   artifacts under `reports/v2_cy_jn_2025/p21_candidates/` and the
   `formal_training_ready` directory, and its output describes that stack.
   A v7 producer has to assemble this directory's acceptance evidence
   (`RESULTS.md`, `qualification/summary.json`, `smoke.json`, the audit
   summary) into one record whose `all_pass` is computed from those items,
   and `make_formal_training_manifests.py --finalize` then reads it.
3. The full suite green apart from the known pre-existing failure, which is
   where the tree stands: `python -m pytest tests/ -q` on the committed tree
   gives `1 failed, 1578 passed, 53 deselected` in 1162 s, and the failure is
   `test_da_market_system::test_standard_cone_matches_rotated_form_and_ac`,
   which fails at HEAD and touches no file this protocol pins.

Recommendation: **NO-GO** until item 1 exists. The environment and the market
are frozen and gated; the launcher has not yet shown it can carry a day.
