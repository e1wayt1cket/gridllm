# Formal training protocol (frozen draft)

**Status: frozen for an exploratory / development run only.** G10 (a new
independent holdout) is not met, so this protocol may not be used to start the
formal multi-seed experiment or to support an independent-generalisation
claim.

## Data

| item | value |
|---|---|
| main dataset | `JN-Base-NEAR`, 80 pairs (subset of FULL96) |
| split seed | 20261031 (`data/processed/v2_cy_jn_2025/near/split.json`) |
| train / validation / test | 53 / 11 / 11 |
| holiday block | 5, held out whole and reported separately |
| data-pool ablation | FULL96, separate protocol; its 16 extra days are training-only |
| holdout | none; the 2025 days are viewed |
| cross-year warning | 2022 metered load priced by a 2025 Jiangnan curve; not same-period settlement |

### Frozen input hashes

| input | sha256 |
|---|---|
| `data/processed/v2_cy_jn_2025/load_price_pairing.csv` | `9cb980dc4cb4be4397634c48eb98be000b7834fc155c7f670fb49ff203e0e2f1` |
| `data/processed/v2_cy_jn_2025/near/load_price_pairing.csv` | `e41b8992cea469b2f7bb05896742837322451ddfca54ddc7ac538036a0555363` |
| `data/processed/v2_cy_jn_2025/bus_loads_96x33.npz` | `754fbc6bb83d057e63b3396062670a00d803d0657a3114a658a53642e90d8078` |
| `data/processed/v2_cy_jn_2025/price_curves.npz` | `316595de0e5e4a01fcf0f1637d6bd12f97dbc7d435b1c12377362bdd5a9b73e1` |
| `data/processed/v2_cy_jn_2025/near/split.json` | `19697cb4f8a008ed725497610fbb56c5a0866d774d3f727e89353c11524b96c9` |
| `data/processed/v2_cy_jn_2025/split.json` (FULL96) | `39b1e9a1679a7b94f3c5b0e1e87f4defd063772335a10ea205fa7290291d2d6d` |
| `data/processed/v2_cy_jn_2025/common_split.json` | `fef4842872d3496a273aa2dd337950296ac7ad303c60dee72325ec05bd25d335` |
| `docs/data.csv` | `1bf06f8d537a5e06911353e4a7348bec8efcedee8bdc76ba84a4bc4d9de8b00c` |
| terminal-neutral B1 manifest | `79a823e5e72c51f82bfb6128b4a31ae80ec1bb344846608b8b137c80f35f0011` |
| terminal-neutral B1 mechanism hash | `835d1a3a298483c491e83287844e5864de7fbbe9878478c323c82c42f6879141` |
| terminal-neutral B1 results | `886ec54e6d553af3c5181bda6351ddef805437946c92a35c823a9a2300be50aa` |
| terminal-neutral B1 block payoffs | `87cb7e9ddc2493b40479b2354ec4fcb266508c730582bb999b03c74f91483fed` |
| terminal SOC config | `abf4f910b87e8c0cf1a5985c6e044758a56b726cd434f2905e3bf88bd461b1c3` |
| trainer configs | `b3d4b084ce9586a6708abc9097aff5d83f1ccba51f2593f6405a11914a34d5a9` |

## Branches

| id | data | reward | gate |
|---|---|---|---|
| `NEAR_B_C2_GATE` | NEAR train 53 | B + C2 | per-leg truthful gate, straight-through |
| `NEAR_B_ONLY_GATE` | NEAR train 53 | B only | same |
| FULL96 pool ablation | FULL96, own split | B + C2 | same, reported separately |

The two NEAR branches differ only in the reward mode. FULL96 is never pooled
with NEAR in one conclusion. Fixed-budget and fixed-common-exposure FULL96
protocols are separate experiments and are not merged.

## Seeds and budget

Five fixed seeds: `11, 23, 37, 53, 71`. The same seed controls the date
sequence, the gate Bernoulli draws and the replay sampling.

| quantity | value |
|---|---|
| minimum full-day exposures per training day | 10 |
| maximum, frozen before start | 15 |
| episodes per seed | 530 minimum, 795 maximum |
| block interactions per seed | 12,720 minimum, 19,080 maximum |
| critic updates | one per block after batch warm-up |
| actor updates | one every 2 blocks |
| update / interaction ratio | 1.0 critic, 0.5 actor |
| batch size | 128 |
| replay capacity | 100,000 transitions |
| start steps | 0 |

Evaluation pauses count as zero exposures. The date sequence is a fixed
permutation over the 53 training days, repeated; a seed never visits a
validation, test or holiday day.

## Validation and model selection

- Validation set: 11 NEAR validation days.
- Policy evaluation is deterministic: hard gate, no price noise.
- Selection metric: mean fleet ΔG over the 11 validation days, where
  ΔG_i = G_i^π − G_i^B1 from actual settlement. Training reward is never the
  selection metric.
- Evaluate after each complete pass over the 53 training days and after the
  final episode.
- Save the best full checkpoint by validation ΔG; save a last checkpoint at
  the end; save a resume checkpoint at every evaluation.
- Early stop if validation ΔG does not improve for 3 consecutive evaluations.
  Early stopping is allowed only within the frozen 15-exposure/day budget.

## Frozen trainer configuration

| knob | value |
|---|---|
| terminal SOC | every ESS SOC0 = SOC24 = 0.5, absolute, day-end windows only |
| gamma | 0.99 per block |
| reward scale | 1000.0, fixed; no running standardisation |
| C2 potential | `(24-t)/24 · 288.19757259048833 · 1.6 · (soc-0.5)` |
| gate | 2 per ESS, 8 total; straight-through; Bernoulli exploration |
| initial gate probability | 0.5 (near-zero logits) |
| gate temperature | 1.0 |
| price exploration noise | 0.5 logit std |
| target policy noise | 0.2 logit std, clip 0.5 |
| actor / critic LR | 3e-4 |
| tau | 0.005 |
| policy delay | 2 |
| networks | actor 256-128-4; centralized twin critic 256-256 per twin |
| observation | OBS_V6, 12 dimensions, unchanged |
| action | executed quote, 2 dimensions per ESS |

`data/processed/v2_cy_jn_2025/trainer_configs.json` carries the two branch
configs and their hashes. Neither the B1 lookup key nor the trainer config may
change after the seed list is opened.

## Abort conditions

Stop the seed and keep the diagnostic checkpoint on any of:

- a clearing failure or fallback;
- any voltage or line violation;
- a B1 manifest or input-hash mismatch;
- an SOC outside its band or an end SOC different from 0.5;
- NaN/Inf in an actor or critic loss;
- a gate/Critic/replay action-semantics mismatch;
- a checkpoint that cannot be restored;
- a reward component that does not close;
- a validation or holdout date inside the training sequence.

No anomalous day is deleted and no constraint is relaxed to keep a run alive.

## What this protocol does not authorise

- Starting formal training while G10 fails.
- Reporting any 2025-test result as an independent holdout result.
- Selecting data, reward mode, potential, gate estimator or hyperparameters
  using the holdout (there is none) or the viewed common test.
