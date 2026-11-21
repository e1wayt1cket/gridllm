# Manifest directory audit

Two frozen manifest directories exist for the same formal-training experiment.
Only one of them is live.

| Directory | Config hash | Role |
|---|---|---|
| `reports/v2_cy_jn_2025/formal_training_v2` | `2148a5777ea6f16a4f5f26730e04ec5cfa4e3d3476c0b50586fe203ba5e4f68f` | live default |
| `reports/v2_cy_jn_2025/formal_training_ready` | `b21a97c15d16aae5ed0f8bd599e54a2f8aafb6e687bca2b63b06bd5a8da8e1e7` | `SUPERSEDED_READ_ONLY` |

Both directories are protected by `.claude/hooks/protect_frozen.py`: writes to
either are denied regardless of which one is the default.

## Tree state

Digest is `sha256(json.dumps({path: sha256(file)}, sort_keys=True))`.

| Directory | Files | Tree digest |
|---|---|---|
| `formal_training_v2` | 12 | `3207d33629b83ceeaf8da11ff1ffa2ef8b0f615096017ecb01bffca6e79c124f` |
| `formal_training_ready` | 12 | `485a9e7353a183aa2ea43115a892579cd445ddd99e19c2b07c666239b89191c6` |

The two directories share three file names but not three contents:

| File | v2 | ready |
|---|---|---|
| `formal_training_code_manifest.json` | `4e5d6c2912e062da…` | `f06332409f170ba6…` |
| `formal_training_config.yaml` | `a60ab1ed4d26c14e…` | `eb6d0e376199e840…` |
| `formal_training_manifest.json` | `ceef65e2b3f7244b…` | `293973193c08904c…` |
| `formal_training_data_manifest.json` | `e5eaa358c3445b61…` | `e5eaa358c3445b61…` (identical) |

The remaining nine files are phase evidence and are unique to each directory.

## What actually differs

A field-by-field comparison of the two configs shows exactly three differences.
Every budget number other than `minimum_rounds` is unchanged.

| Field | v2 | ready | Meaning |
|---|---|---|---|
| `protocol_version` | `2` | absent | the guard is part of the protocol |
| `budget.minimum_rounds` | `10` | absent | the guard's threshold |
| `code.source_sha256.train_da_gate.py` | `a722b2f9…` | `c32b7fec…` | the source that carries the guard |

Unchanged across both: seeds `[11, 23, 37, 53, 71]`, `exposures_per_day_min = 10`,
`exposures_per_day_max = 15`, `episodes_per_seed_min = 530`,
`episodes_per_seed_max = 795`, `block_steps_per_seed_min = 12720`,
`block_steps_per_seed_max = 19080`, `early_stop_patience = 3`, and the whole data
manifest (`d88ea6d4…`).

So the superseded directory differs only by not having the minimum-rounds guard.
A launch driven from it can stop before round 10, which is what happened to
`runs/FORMAL_b21a97c1_seed11` (6 rounds, 318 episodes).

## Seed references

`budget.seeds` is `[11, 23, 37, 53, 71]` in the live config. Run directories are
named `FORMAL_<config_sha256[:8]>_seed<N>`, so a run can be matched to the config
that produced it by name:

| Run | Config | State |
|---|---|---|
| `runs/FORMAL_2148a577_seed11` | v2 | 10 rounds, 530 episodes, `completed_early_stop` |
| `runs/FORMAL_2148a577_seed23` | v2 | `completed_early_stop` |
| `runs/FORMAL_2148a577_seed37` | v2 | `completed_early_stop` |
| `runs/FORMAL_2148a577_seed53` | v2 | `failed`, clearing failure at episode 28 |
| `runs/FORMAL_b21a97c1_seed11` | ready | 6 rounds, 318 episodes, `completed_early_stop` |
| seed 71 | — | never launched |

There is no run directory that cannot be attributed to one of the two configs,
and no run directory named for a config hash that does not exist on disk.

## How the default is resolved

`src/run_formal_training.py` and `src/make_formal_training_manifests.py` both
default to `formal_training_v2`. `--config-dir` still accepts the superseded
directory for historical audit. There is no fallback: pointing at a directory
whose hashes do not verify returns `all_pass: false` and `NO_GO`, it does not
quietly retry v2.

`src/make_formal_training_manifests.py` refuses to write into a directory that
already holds any of its four manifest files, and refuses a second `--finalize`.
A new protocol version therefore has to be published under a new directory name.

## Why the old directory was kept

It is the evidence for the run that exposed the seed 11 incident, and it is
referenced by that run's manifest. Deleting or renaming it would break
reproduction of `runs/FORMAL_b21a97c1_seed11`. It is marked
`SUPERSEDED_READ_ONLY` and is never the default.
