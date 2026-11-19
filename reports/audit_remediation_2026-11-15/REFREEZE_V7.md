# Re-freeze v7

Protocol `v7`, candidate `V7_candidate_001`, config hash `ea92b0af`.
Manifest directory `reports/v2_cy_jn_2025/formal_training_v7/`.
Run ids are `FORMAL_<config-prefix>_seed{11,23}`.

v6 (`ac0f78d2`) and everything older are untouched and remain frozen evidence.
They now report `code_sources_match: false` against this tree, which is the
designed reading: a source path can satisfy only the live manifest's hash.

## Why a re-freeze

v7 does not train the same market. v6 trains the aggregate prosumer model in
`v2_cy_jn_env` on `data/processed/v2_cy_jn_2025_v6`; v7 trains the local
day-ahead clearing on the v3 candidate-day library: 136 days of Jiangsu bus
load paired with 2025 day-ahead price, split 90 train / 18 validation / 18 test
/ 10 holiday, cleared by the exact MIQCP dispatch and the fixed-direction
conditional price solve in `local_da_clearing`.

That changes every interface the protocol pins:

- The observation is `v7_12d`; the action is `storage_quote_v5`, four entries
  per unit (charge price, charge quantity, discharge price, discharge quantity)
  read through `da_quote_map_v5`. Neither is a rescaling of v6's.
- The clearing market is the local one, with its declared `0.93-1.07` voltage
  band and the zero-active reactive asset at buses 17, 31, 32.
- Terminal SOC is free: the day ends on whatever state it reached, valued by
  the settlement's inventory term, never pinned.
- The data footprint is the v3 library, not the v6 aggregate inputs, so the
  data manifest hashes a different input set.

A v6 checkpoint cannot initialise a v7 run. The observation name, the quote
box and the clearing market all differ, so the checkpoint describes a policy
for a market that no longer exists.

## Source changes

33 pins against v6's 22. Fifteen are shared, and two of those changed; the
rest of v6's 22 are dropped as superseded.

Changed shared pins:

| file | change |
|---|---|
| `make_formal_training_manifests.py` | `OUT` -> v7, `CANDIDATE` -> `V7_candidate_001`, `protocol_version` -> 7, `INPUTS` -> the v3 library and the price directory, `SOURCE_FILES` -> 33, `TESTS` -> 11, a `terminal_soc` block declaring `mode: free`, a `training_gap_tolerance` that governs training windows only, and the H2 declaration in `boundaries` and `claims`. |
| `run_formal_training.py` | `OUT` -> v7; the contract and the data checks are selected by `protocol_version`, so a v6 directory still reports v6's `common_split_counts` / `h2_sealed` while a v7 directory reports `library_counts` / `h2_declared`; `command()` reads the trainer path from `formal_training.trainer`. |

Pins added: `rl_da_env.py`, `rl_da_gate_v7.py`, `rl_spec_v7.py`,
`da_quote_map_v5.py`, `rl_td3.py`, `price_forecaster.py`,
`participant_payoff.py`, `money.py`, `local_da_days.py`,
`local_da_contracts.py`, `local_da_clearing.py`, `local_da_operating.py`,
`local_da_accounting.py`, `local_da_audit.py`, `run_local_da.py`,
`validate_checkpoint_schema.py`, `train_local_da_v7.py`,
`training_contract_v7.py`.

Pins dropped: `audit_run.py`, `b1_daily_baseline.py`, `calibrate_anchor_v4.py`,
`calibrate_hinge_v4.py`, `dispatch_socp.py`, `train_da_gate.py`,
`v2_cy_jn_env.py`. Their sources stay in the tree; the v6 manifest still names
them.

`src/training_contract.py` is pinned and unchanged. v7's contract lives in
`src/training_contract_v7.py`: it accepts `terminal_soc.mode: free`, requires
the v3 library counts, refuses a run that does not declare the price dates
drawn on or that claims joint source-load generalisation, and records the
wall-clock projection as a manifest field with no gate. The two modules cover
different protocols and are selected by `protocol_version`; neither is a
generalisation of the other.

## Tests

Eleven pinned tests against v6's seven. Added:
`test_local_da_env_v7.py`, `test_local_da_env_v7_probe.py`,
`test_local_da_env_v7_reward.py`, `test_local_da_v7_days.py`,
`test_local_da_audit_v7.py`, `test_rl_da_gate_v7.py`, `test_rl_spec_v7.py`,
`test_local_da_market.py`, `test_smoke_local_da_v7.py`,
`test_training_contract_v7.py`. Dropped: `audit_phase20_action_routing.py`,
`test_da_quote_mapping.py`, `test_train_da_gate_guards.py`,
`test_train_da_gate_resume.py`, `test_training_contract.py`,
`test_v4_mechanics.py`.

## Data

| | v6 | v7 |
|---|---|---|
| inputs | 10, in `data/processed/v2_cy_jn_2025_v6` and `_ready` | 6, in `reports/candidate_days_v3_2026-11-17/evidence` and `data/processed/v2_cy_jn_2025` |
| counts | train 53 / validation 11 / test 11 / holiday 5 | train 90 / validation 18 / test 18 / holiday 10 |
| split seed | `common_split` | 20261117 |
| H2 | sealed | not sealed; 40 H2 price dates used |

The H2 declaration is mechanical, not a judgement. `holdout_manifest.json`
enumerates `h2_pair_ids`; their price dates number 40, all 40 appear among the
v3 library's 136 price dates, 96 library price dates are outside H2 and no H2
price date is outside the library. The library therefore trained and validated
on every price date the H2 definition names. `data.h2_opened`,
`data.h2_price_dates_used: 40` and `boundaries.h2_opened: true` record it, and
`claims.independent_generalisation_claim: false` bounds what any v7 result may
support: limited price-date adaptation on those 40 dates with historical load
scenarios, never a joint source-load generalisation claim.

## Training-window gap tolerance and solver limit

`formal_training.training_gap_tolerance: 0.01` declares the one acceptance the
training path makes and the settlement path does not. The exact model demands a
1e-6 relative and 0.001 CNY absolute gap; some shaded-quote windows in the
training loop cannot certify that inside the solver's limit while holding a
feasible incumbent (measured: `solver_status 9`, three solutions, `MIPGap`
5.47e-3 on `2021-06-13` block 0). `rl_da_env.GAP_TOLERANCE` accepts such a
dispatch when its own reported relative gap is inside 0.01, prices it with the
same bound separating the conditional objective from the exact one, and counts
the block in `env.gap_solved_blocks`, which every episode row carries as
`gap_solved_blocks`. `local_da_clearing.clear_day`, the settlement ledger and
every audit pass `0.0` and still demand the certified optimum. No settlement
payment and no audit price is read off an uncertified incumbent.

`formal_training.training_window_seconds: 300` raises the per-window solver
limit from 120 s, and `run_formal_training.command` passes it to the launcher
so a launch cannot silently use another value. The limit is a search budget,
not a criterion. The measured basis: on the exploration policy
`2021-06-13` block 5 held an incumbent whose gap exceeded 1 % at 120 s, and
`2021-06-04` block 1 — a day that clears in seconds at the truthful quote —
found no incumbent at all at 120 s. A block that reaches the limit spends it,
so a day whose blocks all bind can cost 24 x 300 s; the declared wall-clock
projection is the typical-block rate and does not bound that.

## Gate proof

`PYTHONPATH=src python -B src/run_formal_training.py --config-dir
reports/v2_cy_jn_2025/formal_training_v7 --check` returns:

```
data_payload_matches true          budget_contract_ok true
code_sources_match true            library_counts true
code_tests_match true              all_disjoint true
config_hash_matches true           legacy_label_declared true
data_manifest_hash_matches true    h2_declared true
code_manifest_hash_matches true    all_pass true
status: null    accepted_candidate: V7_candidate_001
config_sha256: ea92b0af    contract_violations: []
```

## What v7 does not carry

`pretraining_validation_results.json` and `training_readiness.md`. They describe
earlier protocols' sources. The generator writes `training_readiness_status:
null`, and only `--finalize` sets it, from a validation result produced on the
v7 sources. A protocol change invalidates the previous protocol's validation,
so no launch is authorized until that file exists.

No `runs/FORMAL_*` directory was created for v7.

## Not validated

| item | verdict | reason |
|---|---|---|
| a full formal run | `NOT_TESTED` | No v7 run has been launched. |
| the launcher's full path | `NOT_TESTED` | `runs/SMOKE_local_da_v7_launcher3` was the attempt: two train days, two validation days, 120 s windows. Both episodes aborted inside the training block (`CLEAR_UNPROVEN` on `2021-06-13` block 5, `CLEAR_INFEASIBLE` on `2021-06-04` block 1), so no validation round, no checkpoint and no final bundle was written; the run directory holds only `episodes.csv` (2 aborted rows), `train.log` and `run_manifest.json`. The harness then killed the process for low system memory, so the run is incomplete evidence and not a passing smoke. The 300 s limit and the 1 % gap were chosen from those two failures and are untested end to end. |
| the wall-clock projection | `INFERRED` | 12-18 h per seed at roll horizon 20 is extrapolated from the latency table in `reports/local_da_v7_2026-11-18/RESULTS.md`; no formal run has measured it, and the projection assumes typical blocks. It gates nothing. |
| a gap-solved episode | `NOT_TESTED` | The acceptance is tested on a stub solve; the two aborted episodes each recorded `gap_solved_blocks = 1` before failing, so one block was read off an incumbent, but no episode has completed with the column populated. |

## Rollback

`git checkout -- src/ .claude/ CLAUDE.md` restores the pre-re-freeze tree, then
remove `reports/v2_cy_jn_2025/formal_training_v7/`; the v3 library is an input
in `reports/candidate_days_v3_2026-11-17/evidence` that no deleted manifest
owned and is kept. No frozen directory of an earlier protocol was written to.
