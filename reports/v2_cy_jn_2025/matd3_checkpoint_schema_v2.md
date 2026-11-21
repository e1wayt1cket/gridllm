# MATD3 checkpoint schema v2 (V2-CY-JN-2025)

Schema and validation interface only. No checkpoint is loaded for training and
no MATD3 code is changed. Sample and validation artifacts:
`data/processed/v2_cy_jn_2025/pilot/checkpoint_schema_v2.sample.json`,
`checkpoint_schema_validation.json`; validator: `src/validate_checkpoint_schema.py`.

## Three levels

| level | contains | use |
|---|---|---|
| `inference` | actors + observation/action specs + config hash | evaluate a policy |
| `resume` | inference + targets + twin critics + optimizers + RNG + counters + buffer declaration | continue training |
| `bundle` | resume + data hashes + exposure log + eval provenance + reward definition | reproduce the experiment and audit it |

A directory that holds only actors is an **inference** checkpoint and must not
be called resumable. The fifth-phase pilot saved exactly that.

## Bundle layout

```
runs/<run_id>/                     # {dataset}_{seed}_{version}
  manifest.json                    # top-level schema v2 record
  config.json                      # full MarketConfig + support + cycle cost
  data_manifest.json               # pairing / loads / prices / split hashes
  reward_definition.json           # B or C2 + parameters
  exposure.csv                     # episode -> load_date, price_date, blocks
  train_log.csv                    # components, losses, Q, actions, SOC
  checkpoints/step_XXXXXXXX/
    actors/<agent>.pt
    actor_targets/<agent>.pt
    critics/<agent>.pt             # q1 and q2
    critic_targets/<agent>.pt
    optimizers.pt                  # actor + critic state_dicts, Adam moments
    rng.json
    counters.json
  buffer/replay.npz                # or declared absent with a reason
  eval/<split>.csv
  eval/provenance.json             # viewed days, tuning days, model version
  README.md
```

## manifest.json (schema v2)

| section | required fields |
|---|---|
| identity | `schema_version`, `level`, `run_id`, `dataset`, `seed`, `code_revision`, `config_hash` |
| specs | `obs_spec {name, version}`, `action_spec {name, version}` |
| networks | `actors`, `actor_targets`, `critics`, `critic_targets` — one entry per agent |
| optimizers | `actor`, `critic` state (learner, moments, step); scheduler state if any |
| counters | `total_steps`, `episode`, `updates`, `buffer_len` |
| rng | `python`, `numpy`, `torch_cpu`, `torch_cuda` (or null), `env` (or null) |
| buffer | `saved` (bool), `path`, `capacity`, `index`, `count`; if not saved, `reason` |
| reward | `definition`, `gamma`, `scale`, `potential`, `normalization`, `terminal_rule`, `b1_source` |
| data (bundle) | `pairing`, `bus_loads`, `price_curves`, `common_split` SHA-256 |
| evidence (bundle) | `exposure_log`, `eval_provenance` |
| hyperparameters | lr, gamma, tau, policy delay, noise schedule, batch, buffer, start steps, dev penalty |

`config_hash` and every data hash are 64-hex SHA-256. A mismatch on load is a
hard error.

## Resume contract `resume(step)`

1. Load the manifest, config, data manifest and reward definition; verify hashes
   against the files on disk.
2. Load actors, actor targets, twin critics, critic targets, both optimizers,
   RNG states and counters.
3. Restore the replay buffer if present; if not, the run is **not** exactly
   resumable (the reward scale and the sampled batches cannot be reproduced) and
   the reason must be recorded.
4. Continue the day schedule from the recorded episode with the restored RNG;
   the exposure log continues, not restarts.

## Validation results (this phase's schema checker)

| check | result |
|---|---|
| inference sample valid | pass |
| resume sample valid | pass |
| bundle sample valid | pass |
| missing critic in a resume checkpoint detected | pass |
| unsaved buffer without a reason detected | pass |
| malformed SHA-256 detected | pass |
| inference does not require a buffer | pass |

The validator is a pure function over a dictionary; it does not import torch or
the environment.

## What the next run must do

- Implement writers for all three levels before training, not after.
- Store reward components separately: `R_i`, the shaping term, the final
  return, and the normalization applied.
- Save the B1 table and its hash with the reward definition.
- Record the days viewed for tuning in `eval/provenance.json`; the current
  11-day benchmark is no longer a blind holdout.

Without these, option D (a critic-based reward) remains unauditable and the
pilot's critic gap repeats.
