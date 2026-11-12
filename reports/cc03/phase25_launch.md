# Phase 25 — seed 11 launch record

## First attempt, killed before episode 1

`scripts/launch_seed.ps1 -Seed 11` at 2026-11-05 21:51 passed preflight
(`GO_FORMAL_TRAINING`, `1b47e0ed`) and started:

```
C:\ProgramData\anaconda3\python.exe src/train_da_gate.py --split common --seed 11
  --exposures 15 --early-stop-patience 3 --min-rounds 10 --max-rounds 15
  --max-episodes 795 --run-id FORMAL_1b47e0ed_seed11 --trace-blocks
```

The harness killed the background process because the machine was low on
memory. No episode ran, no checkpoint was written.

State on disk after the kill, recorded before removal at the user's ruling:

| path | bytes | sha256 | content |
|---|---|---|---|
| `runs/FORMAL_1b47e0ed_seed11/eval_history.json` | 2 | `4f53cda1…` | `[]` |
| `runs/FORMAL_1b47e0ed_seed11/train.log` | 0 | `e3b0c442…` | empty |

`runs/FORMAL_1b47e0ed_seed11_best/` and `snapshots/FORMAL_1b47e0ed_seed11/`
existed as empty directories with no files. Nothing was removed that held a
measurement: zero episodes, zero checkpoints, zero log lines.

Disposition: the user ruled the empty launch directory may be removed and the
seed relaunched under the same run id. Recorded as TD-09.

## Runtime

The command above shows the launcher resolved `python` to
`C:\ProgramData\anaconda3\python.exe` (3.12.7, torch 2.5.1). The relaunch pins
the interpreter explicitly to `C:\Users\26036\.conda\envs\energy_env\python.exe`
(3.10.20, torch 2.11.0), the environment this project's other work uses, by
invoking `run_formal_training.py` with that interpreter so `sys.executable`
propagates to the child. The five v3 seeds must share one runtime; this records
which one. `run_manifest.json` does not yet record the interpreter (TD-10).
