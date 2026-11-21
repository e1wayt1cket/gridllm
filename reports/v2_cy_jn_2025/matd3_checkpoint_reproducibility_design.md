# MATD3 checkpoint and reproducibility design (V2-CY-JN-2025)

Design only. The pilot saved actors (`save_policy`) and nothing else, so the
trained critics, targets, optimizers and replay buffer are gone. That is why
the sixth phase could measure the penalty gradient exactly but only a
**fresh-critic scale reference** for the critic term.

## Three levels

### 1. Inference checkpoint

Everything needed to reproduce the policy's actions:

- one actor per RL agent, with the policy metadata header (obs spec name and
  version, action spec name and version, obs dim, actor type);
- the action bounds and truthful point per agent;
- the exact `MarketConfig` the actions were trained against, or its hash;
- the day/scenario the evaluation used.

This is what `save_policy` writes today (plus the header). It is enough for
evaluation and not enough for anything else.

### 2. Resumable training checkpoint

Everything needed to continue training **bit-identically**:

| group | fields |
|---|---|
| networks | actors, actor targets, twin critics per agent, critic targets |
| optimizers | actor optimizer state, critic optimizer state (Adam moments, step) |
| counters | `total_steps`, `episode`, `updates`, `start_steps` |
| RNG | `torch.get_rng_state()`, `numpy.random` state, Python `random` state, any env RNG |
| replay buffer | observations, actions, rewards, next observations, done masks **in insertion order**, or an explicit statement that it is not restored and what that breaks |
| reward scale | the buffer-derived scale used at the next update (recomputable only if the buffer is restored) |
| specs | obs/action spec names and versions, action bounds, truthful points |
| config | full MarketConfig, network support assumption, cycle cost, reward definition id |
| data | pairing/bus_loads/price_curves/common_split hashes, dataset id, day pool, exposure log |
| run | seed, hyperparameters, code revision |

Without the buffer, the reward standardization (`buffer.reward_std()`) cannot be
reproduced, so even with all networks and optimizers the next update differs.
A run that does not save the buffer must say so and cannot claim exact
resumption.

### 3. Reproducible evidence bundle

An immutable directory per run:

```
runs/<dataset>_seed<seed>/
  manifest.json            # ids, hashes, code revision, command
  config.json              # full MarketConfig + reward definition + support
  data_manifest.json       # pairing / loads / prices / split hashes
  exposure.csv             # episode -> load_date, price_date, blocks
  train_log.csv            # reward components, losses, Q stats, actions, SOC
  checkpoints/step_XXXX/   # level-2 state
  buffer/                  # optional; declared present or absent
  eval/                    # per-day settlement rows and policy provenance
  README.md                # what was viewed, what was tuned on
```

The bundle must state which days were viewed for tuning and which were used for
the final evaluation; after the pilot, the 11-day test is a viewed benchmark,
not a blind holdout.

## Validation checklist for a future run

1. **Spec match:** `load_policy` on every actor with OBS_V6 and
   `storage_quote_v2` succeeds; a mismatched file raises.
2. **Action identity:** same observation → identical action from the loaded
   actor (exact float equality).
3. **Update identity:** restore a checkpoint, draw one batch from the restored
   buffer with the restored RNG, run one update, and compare critic/actor
   losses and parameter checksums with the pre-save run.
4. **Critic inspectability:** the loaded critic reproduces the actor-gradient
   norm recorded at save time.
5. **Data identity:** hashes in `data_manifest.json` match the files on disk;
   exposure matches the logged schedule.

## Required changes before the next training

- Save critics and critics' targets, the actor targets, both optimizers, the
  RNG states and the counters at every checkpoint.
- Either save the replay buffer or declare it irretrievable and accept that
  exact resumption is impossible.
- Record the reward definition id and its parameters (A/B/C/D, γ, scale,
  terminal treatment) in the manifest, because the same actor under a
  different reward is not the same experiment.
- Record the exposure log in the bundle, not only a summary in a report.
