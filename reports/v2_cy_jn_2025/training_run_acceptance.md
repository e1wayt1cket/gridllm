# Development training acceptance

Run: `EXPLORATORY_NEAR_B_C2_GATE_seed11_v2`. **Exploratory/development only**: H1 joint holdout is 0; the legacy test was not run; the H2 price-date holdout was not read.

## 1. Training completeness

| item | value |
|---|---:|
| rounds completed | 10 / 10 |
| episodes | 530 |
| unique episode ids | 530 (1..530) |
| block interactions | 12720 |
| update calls | 12593 |
| critic updates | 50372 |
| actor update calls | 6297 |
| actor updates | 25188 |
| replay buffer length | 12720 |
| snapshots | 10 |
| checkpoints | 10 |

Two external stops occurred. The run resumed from the last complete round checkpoint (steps 5088 and 6360); uncheckpointed partial-round episodes were re-run, and the episode CSV was truncated to the checkpoint. The final episode ids are 1..530, unique and contiguous. `train.log` retains the interrupted attempts; the CSV is authoritative. All ten round checkpoints were reloaded and re-evaluated; every ΔG reproduced.

An earlier attempt, `EXPLORATORY_NEAR_B_C2_GATE_seed11`, stopped at episode 45 because the C2 check rejected a 6e-7 SOC numerical overshoot with a 1e-12 tolerance. The tolerance was changed to 1e-6 with clipping and a regression test, and the run restarted as `_v2`. The failed run is retained at `runs/EXPLORATORY_NEAR_B_C2_GATE_seed11/` with its diagnostic checkpoint `step_00001056`; it is not part of the 530 completed episodes.

Each round contains 53 distinct dates; the date set is identical in all ten rounds, and every training date has exactly 10 exposures. No date repeats within a round. The 530-row episode CSV, the 530 gate-stat rows and the 10 round snapshots are consistent with the 10 checkpoints (each step equals 24 × its episode count).

## 2. Validation performance

| round | episode | mean ΔG | median ΔG | positive | ESS9 | ESS14 | ESS17 | ESS27 | modes | v/l |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 53 | -35.5 | -21.2 | 4/11 | -24.6 | -3.3 | -0.3 | -7.3 | 1 | 0/0 |
| 2 | 106 | -30.7 | -26.6 | 5/11 | -21.4 | -5.9 | 1.0 | -4.4 | 1 | 0/0 |
| 3 | 159 | -33.2 | 0.0 | 6/11 | 2.4 | -7.5 | -3.1 | -25.0 | 1 | 0/0 |
| 4 | 212 | -78.8 | -86.8 | 4/11 | 3.1 | -55.3 | 2.1 | -28.8 | 1 | 0/0 |
| 5 | 265 | -38.4 | 0.0 | 6/11 | 1.4 | -14.9 | 2.2 | -27.1 | 1 | 0/0 |
| 6 | 318 | -41.0 | 0.0 | 6/11 | 1.4 | -16.9 | 2.7 | -28.3 | 1 | 0/0 |
| 7 | 371 | -64.8 | -27.6 | 4/11 | 0.9 | -31.2 | 2.4 | -36.9 | 2 | 0/0 |
| 8 | 424 | -73.1 | -27.6 | 4/11 | 3.0 | -41.0 | 2.4 | -37.5 | 3 | 0/0 |
| 9 | 477 | -82.8 | -21.6 | 4/11 | -18.2 | -30.3 | 3.6 | -37.9 | 3 | 0/0 |
| 10 | 530 | -77.7 | -4.6 | 4/11 | -4.6 | -25.8 | 3.3 | -50.6 | 3 | 0/0 |

Best validation round: **2** (mean fleet ΔG -30.7, 5/11 positive); checkpoint `C:\Projects\00_WorkSpace\01_Projects\MakerB\runs\EXPLORATORY_NEAR_B_C2_GATE_seed11_v2\checkpoints\step_00002544`. The legacy test was not used for selection.

### Per-ESS energy and cycles (validation means)

| round | ESS9 ch/dis/cyc | ESS14 ch/dis/cyc | ESS17 ch/dis/cyc | ESS27 ch/dis/cyc |
|---:|---:|---:|---:|---:|
| 1 | 0.000/0.000/0.000 | 0.565/0.478/0.326 | 0.620/0.525/0.358 | 0.515/0.436/0.297 |
| 2 | 0.040/0.034/0.023 | 0.555/0.470/0.320 | 0.649/0.549/0.374 | 0.484/0.409/0.279 |
| 3 | 0.724/0.613/0.418 | 0.537/0.455/0.310 | 0.415/0.351/0.240 | 0.165/0.140/0.095 |
| 4 | 0.691/0.585/0.398 | 0.341/0.289/0.197 | 0.715/0.605/0.413 | 0.117/0.099/0.068 |
| 5 | 0.661/0.560/0.382 | 0.335/0.284/0.193 | 0.706/0.598/0.407 | 0.171/0.144/0.098 |
| 6 | 0.599/0.507/0.346 | 0.183/0.155/0.106 | 0.711/0.602/0.410 | 0.220/0.186/0.127 |
| 7 | 0.648/0.548/0.374 | 0.322/0.273/0.186 | 0.714/0.604/0.412 | 0.011/0.009/0.006 |
| 8 | 0.672/0.569/0.388 | 0.404/0.342/0.233 | 0.707/0.598/0.408 | 0.004/0.003/0.002 |
| 9 | 0.407/0.345/0.235 | 0.347/0.293/0.200 | 0.707/0.598/0.408 | 0.000/0.000/0.000 |
| 10 | 0.420/0.356/0.242 | 0.338/0.286/0.195 | 0.707/0.598/0.408 | 0.395/0.334/0.228 |

Terminal SOC is 0.5 for every ESS in every round (max error 0.0e+00); voltage and line violation cells are 0 in every round. The complete 10-round table is `data/processed/v2_cy_jn_2025/training_validation_rounds.csv`.

## 3. Gate learning

Open frequency / mean probability per gate, deterministic validation:

| round | ESS9.charge | ESS9.discharge | ESS14.charge | ESS14.discharge | ESS17.charge | ESS17.discharge | ESS27.charge | ESS27.discharge | distinct modes | sat U/L | permanent open / closed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.00/1.00 | 1.00/1.00 | 1.00/0.98 | 0.00/0.30 | 1.00/0.95 | 0.00/0.33 | 1.00/0.96 | 1.00/0.93 | 1 | 528/0 | 6/2 |
| 2 | 1.00/0.97 | 1.00/0.97 | 1.00/0.98 | 0.00/0.08 | 1.00/0.94 | 0.00/0.23 | 1.00/0.96 | 1.00/0.95 | 1 | 0/0 | 6/2 |
| 3 | 0.00/0.01 | 0.00/0.32 | 1.00/0.98 | 0.00/0.11 | 1.00/0.98 | 1.00/0.99 | 1.00/0.94 | 1.00/0.96 | 1 | 0/0 | 5/3 |
| 4 | 0.00/0.01 | 1.00/0.92 | 1.00/1.00 | 0.00/0.01 | 0.00/0.41 | 1.00/0.97 | 1.00/0.96 | 1.00/0.97 | 1 | 205/172 | 5/3 |
| 5 | 0.00/0.01 | 1.00/0.95 | 1.00/0.96 | 1.00/0.99 | 0.00/0.38 | 1.00/0.98 | 1.00/0.96 | 1.00/0.97 | 1 | 0/2 | 6/2 |
| 6 | 0.00/0.02 | 1.00/0.96 | 1.00/0.97 | 1.00/0.93 | 0.00/0.11 | 1.00/0.97 | 1.00/0.96 | 1.00/0.97 | 1 | 0/166 | 6/2 |
| 7 | 0.00/0.03 | 1.00/0.93 | 0.81/0.66 | 1.00/0.93 | 0.00/0.08 | 1.00/0.83 | 1.00/0.99 | 1.00/0.98 | 2 | 0/264 | 5/2 |
| 8 | 0.00/0.03 | 1.00/0.81 | 0.19/0.25 | 1.00/0.97 | 0.00/0.08 | 1.00/0.78 | 1.00/1.00 | 1.00/0.99 | 3 | 52/262 | 4/2 |
| 9 | 0.00/0.04 | 1.00/0.96 | 0.28/0.30 | 1.00/0.97 | 0.00/0.09 | 1.00/0.81 | 1.00/0.99 | 1.00/1.00 | 3 | 79/248 | 4/2 |
| 10 | 0.00/0.05 | 1.00/0.95 | 0.39/0.40 | 0.99/0.96 | 0.00/0.12 | 1.00/0.91 | 0.00/0.08 | 1.00/1.00 | 3 | 79/264 | 3/3 |

### Continuous quote distribution, best and final rounds

| round | gate | n | mean | std | min | p50 | max |
|---:|---|---:|---:|---:|---:|---:|---:|
| 2 | ESS9.charge | 264 | 140.74 | 2.66 | 129.86 | 141.28 | 143.99 |
| 2 | ESS9.discharge | 264 | 132.66 | 3.69 | 124.48 | 133.34 | 143.93 |
| 2 | ESS14.charge | 264 | 51.84 | 3.17 | 42.10 | 51.82 | 62.07 |
| 2 | ESS14.discharge | 0 | closed | - | - | - | - |
| 2 | ESS17.charge | 264 | 14.77 | 5.78 | 5.09 | 13.86 | 29.51 |
| 2 | ESS17.discharge | 0 | closed | - | - | - | - |
| 2 | ESS27.charge | 264 | 25.32 | 2.34 | 23.17 | 24.61 | 36.26 |
| 2 | ESS27.discharge | 264 | 50.15 | 8.73 | 30.68 | 51.71 | 75.99 |
| 10 | ESS9.charge | 0 | closed | - | - | - | - |
| 10 | ESS9.discharge | 264 | 101.86 | 7.57 | 85.17 | 103.29 | 121.50 |
| 10 | ESS14.charge | 104 | 48.94 | 20.91 | 21.93 | 43.30 | 110.93 |
| 10 | ESS14.discharge | 262 | 194.21 | 118.14 | 0.49 | 268.75 | 299.91 |
| 10 | ESS17.charge | 0 | closed | - | - | - | - |
| 10 | ESS17.discharge | 264 | 1.19 | 0.87 | 0.38 | 0.94 | 6.75 |
| 10 | ESS27.charge | 0 | closed | - | - | - | - |
| 10 | ESS27.discharge | 264 | 222.67 | 55.29 | 57.56 | 237.46 | 288.47 |

**Assessment.** The deterministic policy does **not** form a meaningful differentiated gate strategy. The joint fleet mode is almost static: 1 distinct mode in rounds 1–6, only 2–3 of 256 in the last rounds. Several gates are permanently open or permanently closed over all 264 validation decisions, and the open price branch reaches the lower boundary (264 saturated legs at the final round). The gate has learned a fixed mask plus near-boundary quotes, not a state-dependent declaration of truthful versus premium quotes. The training-time Bernoulli exploration does not rescue the deterministic policy. Full per-round gate data: `gate_behavior_by_round.json`.

## 4. C2 and SOC audit

| check | result |
|---|---|
| C2 identity day | `2022-07-23|2025-07-13` |
| max raw vs clipped SOC deviation | 0.000e+00 |
| max raw SOC band overshoot (audited day) | 0.000e+00 |
| max raw SOC band overshoot (all validation rounds) | 6.028e-07 |
| terminal SOC max error | 0.000e+00 |
| voltage / line violation cells | 0 / 0 |
| B vs C2 discounted identity, max error | 2.984e-15 |
| shaping sum vs boundary, max error | 3.553e-15 |
| B paid only at block 23 | True |
| B at block 23 equals delta | True |
| C2 shaping present | True |

The C2 tolerance is 1e-6, matching solver feasibility. The largest raw SOC overshoot observed across all validation rounds is 6.03e-07, which is numerical and was clipped; a larger excursion would raise. Raw and clipped SOC paths are recorded in `c2_soc_training_audit.json`. The identity audit day had no raw overshoot. Terminal SOC is exactly 0.5 and no voltage or line violation was observed.

## 5. Checkpoints and recovery

- Best validation: `C:\Projects\00_WorkSpace\01_Projects\MakerB\runs\EXPLORATORY_NEAR_B_C2_GATE_seed11_v2\checkpoints\step_00002544` (round 2, step 2544).
- Final: `C:\Projects\00_WorkSpace\01_Projects\MakerB\runs\EXPLORATORY_NEAR_B_C2_GATE_seed11_v2\checkpoints\step_00012720` (step 12720).
- Best root: `runs/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2_best/`.
- All 10 round snapshots: `snapshots/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2/round_001..010.json`.
- Full checkpoint hashes: `data/processed/v2_cy_jn_2025/training_checkpoint_index.json`.
- All 10 checkpoints were reconstructed and re-evaluated; every mean fleet ΔG reproduced the snapshot. See `checkpoint_recovery_audit.md`.

## 6. Holdout boundary

- H2 was **not read**: `h2_holdout_read=false` in the run manifest.
- The legacy test was **not run**: `test_evaluation=null`.
- The 40 frozen H2 price dates remain untouched for a later one-shot evaluation after the model and five-seed protocol are frozen.

## 7. Final conclusion

- **Budget:** completed 10/10 rounds, 530 episodes, 12,720 block steps, 12593 update calls, 50372 critic and 25188 actor updates.
- **Stable improvement:** no. Validation mean fleet ΔG goes from −35.5 (R1) to a best of −30.7 (R2), then worsens to −77.7 (R10). Only 4–6 of 11 validation days are positive.
- **Better than B1:** no. Mean fleet ΔG is negative in every round; B1 is the zero reference by construction.
- **Gate degeneration:** yes. The deterministic gate collapses to 1–3 joint modes, with permanent open/closed gates and lower-bound quote saturation; it is not a meaningful differentiated quote strategy.
- **Other seeds:** not recommended until the gate collapse and the worsening validation curve are diagnosed. A five-seed matrix would likely reproduce the degeneration and is not a fix.
- **H2 unseen:** yes. The 40 price dates were not read.
- **Old code/data:** old MATD3, the clearing engines and the frozen data were not modified in this training run. The only numerical change is the C2 1e-6 SOC tolerance described above.

Do not report this single-seed development run as a formal generalisation result.
