# Re-freeze v5

Protocol `v5`, candidate `V5_candidate_001`, config hash
`6e32398b19406d647aa9ebc7751b74d677c58ce9ebdc023a5472b21cbbc0b68c`.
Manifest directory `reports/v2_cy_jn_2025/formal_training_v5/`.

v4 (`7834d0aa`) and everything older are untouched and remain frozen evidence.
They now report `code_sources_match: false` against this tree, which is the
designed reading: a source path can satisfy only the live manifest's hash.

## Why a re-freeze

The audit's fix list included six defects that live inside the 22 sources
pinned by the v4 code manifest. Editing them in place would have made v4's
`source_sha256` describe code that no longer exists, so the fixes were staged
as `tmp/refreeze_v5/frozen_patches.diff` and applied here, followed by a new
manifest set whose hashes describe the patched sources.

## Source changes

Eight of the 22 pins changed. Five carry the staged patch:

| file | audit item | change |
|---|---|---|
| `dispatch_socp.py` | F6 | The terminal-SOC value term is zero when the endpoint is pinned by an equality. A constant the optimizer cannot move is not a term it traded off; the earlier code valued it at the day mean and added that constant to the objective. |
| `dispatch_socp.py` | F4 | The nodal re-solve objective carries `penalty_carbon_slack` and `penalty_re_slack`, as the first-pass objective does. Without them the second pass could violate a cap at no cost, so the prices it produced were for a cheaper problem than the one solved. |
| `rl_b_c2_gate.py` | F9 | The anchor is the sum over an agent's legs of squared box-normalised deviations, averaged over the batch. It was a mean over the legs as well, which halves a two-leg box and makes the calibrated weight mean something else. |
| `rl_da_gate.py` | F9 | `anchor_weight` 0.01 → 0.10, the weight `rl_td3.CALIBRATED_DEV_PENALTY` selected for the sum form. |
| `rl_env.py` | 5.3 | Comment only. The differential counterfactual is one clear with every RL agent's committed quotes truthful; the code already did that, the comment claimed a per-agent hold. |
| `rl_env.py` | F10 | `DayCapture` starts unrun periods at the fleet's truthful declaration instead of zero. Zero is truthful for a battery and a giveaway for a load or prosumer, so a settlement over 96 periods priced the unrun part of a partial day at zero. |
| `checkpoint_writer_v2.py` | 5.10 | A checkpoint is written into `<step>.tmp/` and renamed when complete; `manifest.json` is written through a `.tmp` and `os.replace`. An existing checkpoint directory now refuses rather than being refilled. |

Three more changed without a staged patch:

| file | audit item | change |
|---|---|---|
| `make_formal_training_manifests.py` | payload truth | `CANDIDATE` → `V5_candidate_001`, `protocol_version` → 5, anchor weight and form restated to the calibrated sum, `supersedes` → v4, `checkpoints` note. Without this the new manifest would have declared 0.01 while the code ran 0.10. |
| `run_formal_training.py` | live pointer | `OUT` → v5. |
| `training_contract.py` | live pointer | `DEFAULT_CONFIG_DIR` → v5. |

## Tests

| file | pinned | change |
|---|---|---|
| `tests/test_v4_mechanics.py` | yes | `test_v4_config_defaults` asserted `anchor_weight == 0.01`. F9 moved the live default to 0.10, so the assertion reads 0.10 and says why. v4's own record of 0.01 stays in `formal_training_v4/formal_training_config.yaml`; the test guards the live protocol, not a superseded value. This is the only pinned test whose content changed, and its new hash is in the v5 code manifest. |
| `tests/test_day_capture.py` | no | The fake env in `test_from_env_adopts_the_envs_geometry_and_agents` gained `_default_actions`, because `from_env` now installs the fleet's truthful declaration for unrun periods. A `getattr` fallback in the source would have restored the F10 behaviour silently for any env lacking the method, which is the bug F10 removed. |

The first full-suite pass after the re-freeze failed on exactly these two, and
on nothing else: `2 failed, 1483 passed, 53 deselected`. They are recorded here
rather than left out because a re-freeze that quietly rewrote a pinned
assertion, or quietly relaxed a source check to satisfy a fake, would be the
failure mode this guard exists to prevent.

## Non-pinned changes in the same tree

| file | audit item | change |
|---|---|---|
| `models.py` | 3.8 / 8.2 | `opf_tolerance` and `opf_max_iter` deleted. Neither had a reader, so both were knobs that silently did nothing. |
| `llm.py` | 3.8 | The two deleted names removed from the natural-language config exclusion set. |
| `dispatch.py` | 3.8 | `_GUROBI_MISSING_MSG` deleted. It was never referenced, and each engine's own `RuntimeError` already names the mode; a license failure surfaces Gurobi's own message, which names the license. |
| `.gitignore` | 10.3 | `!data/processed/industry_meter_bus_map_v2.csv` and `!reports/multi_user_load_2026-11-13/evidence/*.csv`. `run_historical_da.py`'s reproduction manifest lists the first of these verbatim, so a checkout could not rebuild that run. |

`opf_tolerance` / `opf_max_iter` were also two fields of `asdict(MarketConfig)`,
which is what `rl_b_c2_gate.config_hash` hashes into `market_config_hash`. A v4
run can therefore no longer be resumed, by construction. That is the correct
outcome for a protocol change and is stated here rather than worked around,
because a resume would otherwise mix two protocols' rewards under one manifest.

## Deliberately not done

| item | reason |
|---|---|
| 3.8 solver-parameter single source | The six `setParam` calls are not divergent: `Method` 2 (barrier) vs 1 (dual simplex) and `QCPDual` / `BarHomogeneous` belong to different engines on purpose. Consolidating them is cosmetic and touches every engine; the one measured defect in that row was the two dead knobs, which are gone. |
| 4.2 settlement declaration | The payload still says `two_settlement_ledger` while the formal path settles through `participant_payoff`. Which one is right is an open decision (plan D-F), so the declaration was left as it was rather than changed to match code that may itself change. |
| 5.9 thread count, 10.2 source list | Not prepared in the staged patch and not measured in this round. Both need their own evidence before they can be claimed. |

## What v5 does not carry

`pretraining_validation_results.json` and `training_readiness.md` are v4
artifacts describing v4 code. They were not copied. The generator writes
`training_readiness_status: null`, and only `--finalize` sets it, from a
validation result produced on the current sources.

`PYTHONPATH=src python src/run_formal_training.py --check --config-dir reports/v2_cy_jn_2025/formal_training_v5`
returns, in full:

```
data_payload_matches true          common_split_counts true
code_sources_match true            all_disjoint true
code_tests_match true              legacy_label_matches true
config_hash_matches true           h2_sealed true
data_manifest_hash_matches true    budget_contract_ok true
code_manifest_hash_matches true    all_pass true
status: null    accepted_candidate: V5_candidate_001
```

So the tree and its manifest agree, and the protocol is not validated. No
formal launch is authorized: `run_formal_training.py --seed` refuses unless the
status is `GO_FORMAL_TRAINING`.

## What the patch did to the numbers

Method: `tmp/v4_src_tree/` holds a copy of `src/` with every tracked source
reverted to `HEAD`, i.e. the pre-re-freeze tree. `tmp/diag_v5_before_after.py`
clears one fixed day under whichever tree is on `PYTHONPATH` and writes the
arrays to `.npz`; `tmp/verify_v5_targeted.py` does F10 and 5.10 the same way.

### F6 -- formal day, no constraint multi-objective

Day `2022-10-20` on price curve `2025-10-20`, `typical_day_industry_subset`,
four ESS units at truthful quotes, terminal SOC pinned at 0.5.

| quantity | before | after |
|---|---|---|
| `lmp` (96x33) | -- | bitwise identical |
| every unit's `p_ch`, `p_dis` | -- | bitwise identical |
| fleet payoff | 656.771154 | identical |
| `clearing_objective` | 66432.444952 | 65914.221619 |

Delta = -518.223333, which is exactly `mean(wholesale) 161.944791667 x 3.200
MWh`. 3.200 MWh is the pinned end energy: four units x 1.6 MWh x SOC 0.5, and
`storage.discount_factor` is 1.0 so `disc_T` is 1. The 518.22 CNY was 0.78% of
a reported objective the optimizer could not have changed, because the endpoint
it prices was an equality.

### F4 -- same day with the constraint method forced on

Cap 60 tCO2 (the day emits 66.21), `re_min_rate` 0.35 (the day clears 28.98%).

| quantity | before | after |
|---|---|---|
| `re_consumption_rate` | 28.980154 | 80.737786 |
| slack shadow prices reported | `carbon_slack_tco2`, `re_slack_mwh` | none |
| `lmp` mean / max | 168.901309 / 409.151525 | 421.245816 / 669.881298 |
| throughput | 21.207500 MWh | 11.896224 MWh |
| fleet payoff | 656.773069 | 182.976772 |

The pre-patch re-solve dropped both caps and returned the *unconstrained*
schedule: its objective (66432.444952) and throughput are the no-constraint
day's to the digit, while it reported nonzero slacks for the violation. The
post-patch re-solve holds both caps. The two clearing objectives are different
functions and are not comparable to each other.

Scope: `use_constraint_multi_obj` is false on the formal day, so F4 moves the
constraint-based method (`compare_methods.py` and any constraint-based run),
not formal training. F6 moves every whole-day clear that pins the endpoint,
which includes the formal path, but only the reported objective, not the prices
or the schedules.

### F10 -- a partial day

`DayCapture` fed one block (4 periods) of a 96-period day:

| quantity | before | after |
|---|---|---|
| unrun periods reading a declared quote of zero | 92 of 92 | 0 of 92 |
| what the 92 unrun periods then read | 0.0 | 1.0 (the fleet's truthful multiplier) |

### 5.10 -- an interrupted checkpoint save

`torch.save` made to fail on its second call:

| quantity | before | after |
|---|---|---|
| final `checkpoints/step_00000048` after the failure | present | absent |
| staging directory left behind | none | `step_00000048.tmp` |
| `manifest.json` written | no | no |
| re-saving an existing step | silently allowed | `FileExistsError` |

The pre-patch behaviour is the dangerous half: a directory with the final
name, a partial set of files inside it, and no manifest, so a later reader has
a checkpoint that looks complete and is not.

Known consequence of the post-patch behaviour: after an interruption the
staging directory stays and a save at the same step refuses with "staging
directory ... already exists". A resume normally restores the last *complete*
checkpoint's counters and so writes a different step, but a retry at the same
step needs that `.tmp` removed by hand. Nothing cleans it up automatically.

## What is still not validated

`F9` changes what the actor is regularised toward. It is a training-time
change with no day-level measurement available: the only way to observe it is a
run under the v5 protocol, which `status: null` forbids. `NOT_TESTED`.

The `baseline` scenario was not re-cleared for the F6 table. `NOT_TESTED`; the
measured day pins the mechanism to an exact CNY figure, but that is an
inference for `baseline`, not a measurement of it.

Until a baseline recompute exists, no number in this repository may be quoted
as a v5 result. `JN-Base`, the three-day settlement record and the
training-readiness headroom all cite v4 numbers.

## Verification run

| check | command | result |
|---|---|---|
| guard still denies v5, v4 and the pinned sources, and allows unpinned ones | `protect_frozen.py` called with each path on stdin | `deny` for v5 config, v5 manifest, v4 config, `src/dispatch_socp.py`; `allow` for `src/models.py` |
| guard contracts | `pytest tests/test_frozen_guard.py -q` | `31 passed` |
| patched modules | `pytest` on the checkpoint, gate, env, terminal-SOC, clearing-schema, resume and guard suites | `74 passed` |
| full suite | `PYTHONPATH=src python -m pytest tests/ -q` | see `REPORT.md`; recorded there, not here, so this file cannot claim a count it did not see |

## Rollback

`git checkout -- src/ .claude/` restores the pre-re-freeze tree, then delete
`reports/v2_cy_jn_2025/formal_training_v5/` and restore
`.claude/settings.json` with `git checkout -- .claude/settings.json` to put the
PreToolUse registration back. v4's directory was never written to and does not
need restoring.
