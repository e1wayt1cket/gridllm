# Re-freeze v6

Protocol `v6`, candidate `V6_candidate_001`, config hash
`ac0f78d2bc3aa3349597ef4f500910263524e65609d122ee8475c5ef2556ed8f`.
Manifest directory `reports/v2_cy_jn_2025/formal_training_v6/`.
Run ids are `FORMAL_ac0f78d2_seed{11,23}`.

v5 (`6e32398b`) and everything older are untouched and remain frozen evidence.
They now report `code_sources_match: false` against this tree, which is the
designed reading: a source path can satisfy only the live manifest's hash.

## Why a re-freeze

Two reasons, and the second is the one that forces a version bump.

The audit's fix list had three items left inside pinned sources: the anchor's
magnitude was computed but never reported (`5.2`), a non-finite clearing result
could reach settlement and the reward without raising (`6.3`), and the B1
baseline path was a bare constant no manifest recorded.

Separately, `data/processed/v2_cy_jn_2025/b1_terminal_neutral_results.csv` is a
hashed input of the v5 data manifest, and it was cleared before the radial-tree
correction: it describes a network that admitted the four open tie lines of
`case33bw`. The live `rl_b_c2_gate.B1_RESULTS` pointed at it, so every gate under
the live protocol was measured against a baseline on the wrong tree. Rewriting
that file in place would make the v5 manifest describe a dataset that no longer
exists. The recompute therefore lands in a new directory, which is what makes
this a new protocol rather than a patch.

## Source changes

Seven of the 22 pins changed.

| file | item | change |
|---|---|---|
| `rl_b_c2_gate.py` | B1 baseline | `B1_RESULTS`, `B1_BLOCKS`, `B1_MANIFEST` → `data/processed/v2_cy_jn_2025_v6/`, on the recomputed artifact. |
| `rl_b_c2_gate.py` | 5.2 | `train_day` accumulates the anchor term and the actor loss per agent from every update it ran, and returns `anchor_by_agent` (episode mean, `None` when no update ran) and `anchor_weight`. The anchor is a regulariser whose size relative to the loss is what says whether an episode's updates were learning or regularisation; before this it was computed and discarded. |
| `rl_b_c2_gate.py` | 6.3 | `_day_rewards` raises `FloatingPointError` on a non-finite per-block series, and `update` raises on a non-finite critic or actor loss, naming the update index and the offending values. A NaN reaching the reward reads as an enormous or a zero profit rather than as a failed solve, so it is refused at the boundary. |
| `rl_da_gate.py` | 5.2 | The anchor accumulators and `anchor_weight` are passed through the same path as the hinge statistics. |
| `train_da_gate.py` | 5.2 | Two columns on the per-episode row: `anchor_<agent>` and `anchor_weight`. |
| `dispatch_socp.py` | 6.3 | `_assert_finite` runs over `lmp` and each unit's `p_buy`, `p_sell`, `served`, `unserved`, `pv_used`, `wind_used`, `p_ch`, `p_dis`, `soc` before the result dict is returned. A non-finite value raises instead of being settled. |
| `make_formal_training_manifests.py` | payload truth | `OUT` → v6, `CANDIDATE` → `V6_candidate_001`, `protocol_version` → 6, `INPUTS` as `(name, root)` pairs so the three B1 inputs hash from the v6 directory while the load and price inputs stay in the v5 one, and a `compatibility` block that names the two reasons a v5 checkpoint cannot initialise a v6 run: the source hashes and the B1 baseline. |
| `run_formal_training.py` | live pointer | `OUT` → v6. |
| `training_contract.py` | live pointer | `DEFAULT_CONFIG_DIR` → v6. |

## Tests

No pinned test changed: the v6 code manifest carries the same seven test hashes
as v5. `PYTHONPATH=src python -m pytest tests/test_frozen_guard.py -q` →
`31 passed`.

Every other suite is reported in `REPORT.md`.

## Non-pinned changes in the same tree

| file | item | change |
|---|---|---|
| `run_artifacts.py` | 4.4 | The three settled-money columns move out of `metrics.csv` into `economics.csv`, keyed by `episode`; no run that recorded no money gets the file. A reward is the differential signal an actor learns from, a payoff is money, and one row holding both invites reading the signal as the money. |
| `data_aggregator.py` | 3.5 | `run_summary` carries `opf_mode`, read from the run's own `config.json`. `dc` has no voltage variables and does not model the storage fleet, so the engine is not a detail of a consolidated table. |
| `make_training_summary.py` | 3.5 | The digest's run table gains an engine column; a run written before the field existed reads `—`. |
| `run_terminal_neutral_b1.py` | B1 producer | `--out-dir` (already added) defaults to the v6 directory, so a no-argument rerun cannot overwrite a hashed v5 input. |
| `.claude/hooks/protect_frozen.py` | live pointer | `MANIFEST_DIRS[0]` → `formal_training_v6`; v5 stays listed below it. |

## What the B1 recompute did to the numbers

Same 96 days, same pairing, same order. `96/96` feasible, `0` fallbacks,
`0` non-optimal, `0` voltage-violation cells, `0` line-overload cells,
`max |terminal_err|` `0.0` — before and after.

| quantity (mean over 96 days) | v5 artifact | v6 artifact | change |
|---|---|---|---|
| `G_ESS9` | 39.1439 | 44.4498 | +13.6% |
| `G_ESS14` | 37.9859 | 42.5424 | +12.0% |
| `G_ESS17` | 36.8206 | 41.0271 | +11.4% |
| `G_ESS27` | 51.4353 | 51.6378 | +0.4% |
| `G_total` | 165.3858 | 179.6571 | +8.6% |
| `losses_mwh` | 2.8103 | 4.2113 | +49.9% |
| `cycles_ESS*` | 0.3667–0.4022 | 0.3712–0.3934 | within ±2.2% |

All 96 days changed. The pattern is the correction itself: the pre-correction
tree used the open tie lines in place of four main segments, which understates
the losses a radial feeder carries (2.81 → 4.21 MWh/day) and prices the four
units' arbitrage against a cheaper network, so the honest tree pays them more.
Throughput is essentially unchanged, so this is a price difference, not a
quantity difference.

`near` subset only (80 of 96 days): mean `G_total` 147.4516 → 161.4853.

## What v6 does not carry

`pretraining_validation_results.json` and `training_readiness.md`. They are v4
artifacts describing v4 code. The generator writes
`training_readiness_status: null`, and only `--finalize` sets it, from a
validation result produced on the current sources. A protocol change
invalidates the previous protocol's validation, so no launch is authorized.

`PYTHONPATH=src python src/run_formal_training.py --check` returns, in full:

```
data_payload_matches true          common_split_counts true
code_sources_match true            all_disjoint true
code_tests_match true              legacy_label_matches true
config_hash_matches true           h2_sealed true
data_manifest_hash_matches true    budget_contract_ok true
code_manifest_hash_matches true    all_pass true
status: null    accepted_candidate: V6_candidate_001
config_sha256: ac0f78d2...    contract_violations: []
```

So the tree and its manifest agree, and the protocol is not validated.
`run_formal_training.py --seed` refuses unless the status is
`GO_FORMAL_TRAINING`.

## What is still not validated

| item | verdict | reason |
|---|---|---|
| 5.2 anchor observability | `PASS` (code path) / `NOT_TESTED` (values) | The columns exist and the accumulator is exercised by the gate suites, but no run has written an episode row yet, so no `anchor_<agent>` value has been read. Its magnitude is a training-time quantity with no day-level measurement available. |
| 6.3 non-finite guards | `NOT_TESTED` | No test injects a NaN into a solved clearing or into a loss vector. The guards are structural (a raise at the boundary) and compile, but nothing has made them fire. |
| 4.4 `economics.csv` | `PASS` | `tests/test_run_artifacts.py` asserts the header, the absence of the columns from `metrics.csv`, and the load round-trip. |
| 3.5 engine column | `PASS` | `build_digest` rendered the column from real run artifacts; the value read was `socp`. |
| B1 baseline on old diagnostics | `NOT_TESTED` | `audit_holdout_exposure.py`, `diagnose_profit_opportunity.py` and `audit_winter_15min_quote_acceptance.py` still read the pre-correction artifact; see `TECHNICAL_DEBT.md` TD-13. |
| an evaluation CSV's engine | `NOT_TESTED` | Not recorded in any generation of `eval.csv`; see TD-16. |

Until a baseline recompute exists under v6, no number in this repository may be
quoted as a v6 result. `JN-Base`, the three-day settlement record and the
training-readiness headroom cite earlier protocols.

## Rollback

`git checkout -- src/ .claude/` restores the pre-re-freeze tree, then delete
`reports/v2_cy_jn_2025/formal_training_v6/`; the v6 data directory
`data/processed/v2_cy_jn_2025_v6/` is an input the deleted manifest no longer
names and may be kept. Restore `.claude/settings.json` with
`git checkout -- .claude/settings.json` to put the PreToolUse registration back.
No frozen directory of an earlier protocol was written to.
