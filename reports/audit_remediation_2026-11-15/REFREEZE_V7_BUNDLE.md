# Re-freeze v7: the bundle data slots

Protocol `v7`, candidate `V7_candidate_003`, config hash `b8b61d8d`. This
record supersedes `REFREEZE_V7_DISJUNCTION.md` (`V7_candidate_002`, config hash
`8db03638`) as the live manifest. No formal run and no formal round exists
under either hash, so nothing is invalidated retrospectively.

## Why

The first launcher run that reached the end of a round stopped at the final
bundle:

```
ValueError: checkpoint manifest failed schema validation: data.pairing: must be
a 64-hex sha256; data.bus_loads: must be a 64-hex sha256; data.price_curves:
must be a 64-hex sha256; data.common_split: must be a 64-hex sha256
```

`validate_checkpoint_schema` names four fixed data slots at bundle level, the
v6 stack's artifacts. `train_local_da_v7.py` passed
`{"library_split": sha(LIBRARY / "split.json")}` — a key that appears nowhere
else in `src/`, `tests/` or `tools/` and that nothing reads. It was written
when v7 was frozen on its own data stack and never checked against the schema.

This is not a writer-only defect: `checkpoint_loader_v2.load_checkpoint` runs
the same `validate`, so a bundle written past the check would have failed to
load as well. v7 could not produce a final bundle at all.

The round itself was unaffected. Before this record,
`runs/SMOKE_local_da_v7_round` held `episodes.csv` over two days, one
validation round in `eval_history.json`, the round snapshot and the resume and
best checkpoints; only the bundle was missing, and only because of the
mismatch.

## What changed

`src/train_local_da_v7.py` — one helper, `bundle_data()`, hashes the files this
protocol actually reads into the four schema slots:

| slot | file |
| --- | --- |
| `pairing` | `reports/candidate_days_v3_2026-11-17/evidence/split.json` |
| `bus_loads` | `reports/candidate_days_v3_2026-11-17/evidence/bus_active_load_mw.npz` |
| `price_curves` | `data/processed/v2_cy_jn_2025/price_curves.npz` |
| `common_split` | the same `split.json` |

`src/make_formal_training_manifests.py` — `CANDIDATE` is `V7_candidate_003`,
which moves the config hash because `accepted_candidate` is part of the config
payload.

Nothing in the environment, action space, reward, gate, quote map or trainer
changed.

## What the slots do not say

The slot names are the schema's, not this protocol's. A reader who resolves
`data.pairing` against the v6 pairing CSV, or `data.common_split` against
`common_split.json`, will get a mismatch by design: the values are v7 library
files. v7 has no separate common split — the block assignment lives in the
library's `split.json`, so `common_split` repeats `pairing`'s hash rather than
naming a file that does not exist.

`bus_loads` hashes the active-load array only. The reactive array is pinned by
the library's own `manifest.json` and re-checked by `local_da_days.load_pair`
on every day read, so a swapped reactive file fails there rather than in this
slot.

## Gate proof

`PYTHONPATH=src python -B src/run_formal_training.py --config-dir
reports/v2_cy_jn_2025/formal_training_v7 --check`:

```
data_payload_matches true     code_sources_match true
code_tests_match true         config_hash_matches true
data_manifest_hash_matches true   code_manifest_hash_matches true
budget_contract_ok true       library_counts true
all_disjoint true             legacy_label_declared true
h2_declared true              all_pass true
status: null    accepted_candidate: V7_candidate_003
config_sha256: b8b61d8d    contract_violations: []
```

`status` is null for the reason it was null before: the readiness status is
written only by `--finalize`, from a `pretraining_validation_results.json`
produced on these sources, and no producer for v7 exists.

## Bundle proof

`runs/SMOKE_local_da_v7_bundle`, `--train-days 2 --validation-days 2
--max-episodes 2 --min-rounds 1 --max-rounds 1 --seconds 300`, seed 99:
two training episodes, one validation round, `stop_reason: max_rounds`,
`best.value: -758.84` at round 1.

`load_checkpoint("runs/SMOKE_local_da_v7_bundle/final")` returns a bundle whose
`data` block carries the four slots, and each value equals `sha256sum` of the
file named above. `eval_history.json` holds one round;
`snapshots/SMOKE_local_da_v7_bundle/round_001.json` exists.

This is a smoke, not a formal launch: `formal_launch` is false for it, and it
authorises nothing.

## Cost

None measured. The change adds four file hashes to one save call.

## Rollback

The superseded manifest set is preserved at
`tmp/manifest_superseded/v7_8db03638/`. It is not tracked and describes a
source tree that no longer exists, so restoring it would require reverting this
record's commit as well.

## Not validated

- No formal run and no formal round exist under `b8b61d8d`.
- The price qualification and latency tables in
  `reports/local_da_v7_2026-11-18/RESULTS.md` were measured on the bilinear
  encoding; the disjunction record states they must be re-measured, and this
  record does not re-measure them either.
- `pretraining_validation_results.json` does not exist for v7. Until a v7
  producer assembles one, `status` stays null and no launch is authorised.
