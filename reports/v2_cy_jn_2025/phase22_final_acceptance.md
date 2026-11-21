# Phase 22 final acceptance

**training_readiness_status = GO_FORMAL_TRAINING**

This is the final acceptance table for the formal-training preparation. The
full decision, frozen protocol and residual risks are in
`formal_training_ready/training_readiness.md`; the machine-readable record is
`formal_training_ready/pretraining_validation_results.json`.

## Data

| condition | state | evidence |
|---|---|---|
| `common_split.json` near view is 53 / 11 / 11 / 5 | PASS | `formal_training_data_manifest.json` |
| train / validation / test / holiday are pairwise disjoint | PASS | same, `overlaps` empty |
| test + holiday = `LEGACY_SEEN_TEST` 16 pairs | PASS | same, equality with `holdout_status.json` |
| frozen input hashes match | PASS | same, `input_sha256` |
| H2 not opened | PASS | entry open audit, `phase22_entry_isolation.json` |
| test / holiday not evaluated | PASS | entry manifest (`test_evaluation: null`) |

## Environment

| condition | state | evidence |
|---|---|---|
| 24-block rolling clearing, no fallback | PASS | smoke hard checks |
| voltage and line limits | PASS | 0 cells in both smoke runs |
| SOC bounds and terminal SOC 0.5 | PASS | smoke and entry audit |
| B1 settlement identity | PASS | `G_policy − G_B1 = delta` ≤ 1e-6 |
| no test/H2 mechanism change | PASS | manifest hashes; new run ids |

## Algorithm

| condition | state | evidence |
|---|---|---|
| update and execution use the same v3 map | PASS | `phase20_action_routing.json`; `tests/test_da_quote_mapping.py` |
| v2 calls in one actor update | 0 | routing update audit |
| actor gradient finite | PASS | routing update; Phase 19 gradient chain |
| Bellman target reconstructed from code | PASS | `phase20_bellman_diagnostics.json` |
| critic action indices/order | PASS | routing audit |
| smoke has no persistent action degeneration | PASS | `phase21_smoke_control_vs_candidate_001.json` |
| candidate 001 accepted | PASS | non-inferiority rule, `phase21_decision.md` |
| critic ranking/calibration | **residual risk** | explained but not repaired; see readiness doc |

The critic residual risk is disclosed, not counted as a deterministic code
failure. No network, sampler or credit change was supported by the evidence.

## Engineering

| condition | state | evidence |
|---|---|---|
| checkpoint save/restore/continue | PASS | `phase22_checkpoint_roundtrip.json` |
| pause/resume, no duplicate counters or rows | PASS | `phase22_resume.json` |
| output in a new run directory; refuse reuse | PASS | entry guard; smoke run ids |
| per-episode date/device/block trace | PASS | `block_decisions.jsonl`, 96 rows/episode |
| validation not written to replay | PASS | `phase22_entry_isolation.json` |
| OBS future-price causality | PASS | same, with positive control |
| data/config/code manifests and hashes | PASS | `formal_training_manifest.json`, `--check` |
| historical artifacts not overwritten | PASS | new names and new run ids only |
| default test suite | PASS | 1365 passed, 0 failed; `full_test_suite.md` |

## Decision

All data, environment, algorithm and engineering groups pass. The residual
critic-ranking risk is recorded. The formal training entry is prepared but was
**not** started:

```bash
PYTHONPATH=src python src/run_formal_training.py --check
```

`formal_training_manifest.json` carries
`training_readiness_status = GO_FORMAL_TRAINING`. The old G10 NO-GO remains the
boundary for an independent-generalisation claim.
