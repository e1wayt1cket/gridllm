# Checkpoint recovery audit

Run: `EXPLORATORY_NEAR_B_C2_GATE_seed11_v2`. All ten round checkpoints were reconstructed into loadable roots, loaded, and used to re-evaluate the validation split. The re-evaluated mean fleet ΔG matches the training-time snapshot for every round.

| round | episode | step | checkpoint tree sha256 (first 16) | reproduced |
|---:|---:|---:|---|---|
| 1 | 53 | 1272 | `7d0be9b09c2b29d4` | True |
| 2 | 106 | 2544 | `55f79ac1f82eb35e` | True |
| 3 | 159 | 3816 | `d74d006387ff0ffd` | True |
| 4 | 212 | 5088 | `7cfd5732f26c46bd` | True |
| 5 | 265 | 6360 | `f90cba81e3ea73d6` | True |
| 6 | 318 | 7632 | `beb34792b1ed4665` | True |
| 7 | 371 | 8904 | `3903b7c23cdef49e` | True |
| 8 | 424 | 10176 | `3e3c91eac6f977a6` | True |
| 9 | 477 | 11448 | `40f77670fa09ef3c` | True |
| 10 | 530 | 12720 | `c96329b499f2243b` | True |

- Best validation checkpoint: `C:\Projects\00_WorkSpace\01_Projects\MakerB\runs\EXPLORATORY_NEAR_B_C2_GATE_seed11_v2\checkpoints\step_00002544` (round 2, step 2544).
- Final checkpoint: `C:\Projects\00_WorkSpace\01_Projects\MakerB\runs\EXPLORATORY_NEAR_B_C2_GATE_seed11_v2\checkpoints\step_00012720` (step 12720).
- Best root: `runs/EXPLORATORY_NEAR_B_C2_GATE_seed11_v2_best/` (manifest step 2544).
- Full hash index: `data/processed/v2_cy_jn_2025/training_checkpoint_index.json`.
- The reconstructed roots use a manifest whose `step` and `counters` are set from the step's own `counters.json`; the network, optimizer, RNG and buffer files are hard-linked from the original step.
- No test or H2 evaluation was used in this recovery audit.
