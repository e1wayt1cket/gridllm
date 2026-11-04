# Phase 24 decision

**PASS_SEED11_RERUN_AND_CONTINUE**

Seed 11 rerun `FORMAL_2148a577_seed11` completed 10 rounds with 530 episodes, 12720 block interactions and 12593 optimizer updates.

- early stop: round 10
- best: round 3, value -20.775
- best checkpoint: `C:\Projects\00_WorkSpace\01_Projects\MakerB\runs\FORMAL_2148a577_seed11_best\checkpoints\step_00003816`
- final checkpoint: `C:\Projects\00_WorkSpace\01_Projects\MakerB\runs\FORMAL_2148a577_seed11\checkpoints\step_00012720`
- Test / holiday / H2: not run / not run / not read

The guard held at or beyond round 10 and the run stayed within 15 rounds. The remaining seeds were started in order with individual `--seed` invocations (never `--all`), each into its own run directory under the same version-2 config and code:

- `runs/FORMAL_2148a577_seed23`
- `runs/FORMAL_2148a577_seed37`
- `runs/FORMAL_2148a577_seed53`
- `runs/FORMAL_2148a577_seed71`

Seed 23 was training when this decision was written; the remaining seeds follow sequentially and stop the chain on the first failure. Their outcomes are recorded in each run directory rather than in this decision.
