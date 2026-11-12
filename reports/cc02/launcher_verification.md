# CC-02 launcher verification

Interpreter: `C:\Users\26036\.conda\envs\energy_env\python.exe` (Python 3.10.20,
Gurobi 13.0.1, academic licence). It was placed first on `PATH` for every
PowerShell wrapper so the wrapper's `python` is the verified interpreter.

## Gate and dry run

| command | exit | result |
|---|---|---|
| `scripts/preflight.ps1` | 0 | Gurobi smoke ok; `training_contract.py --check` `ok: true`, no violations; `run_formal_training.py --check` `all_pass: true`, `budget_contract_ok: true`, `status: GO_FORMAL_TRAINING`, `config_sha256: 2148a577…` |
| `python src/run_formal_training.py --check` | 0 | same state, saved as `run_formal_check.json` |
| `python src/run_formal_training.py --seed 71 --dry-run` | 0 | printed `--split common --seed 71 --exposures 15 --early-stop-patience 3 --min-rounds 10 --max-episodes 795 --run-id FORMAL_2148a577_seed71 --trace-blocks`; `runs/FORMAL_2148a577_seed71` still absent |

The dry run is the only seed-71 command executed. Training was never started.

## Exit-code propagation

| test | command | expected | observed |
|---|---|---|---|
| missing seed | `powershell -NoProfile -NonInteractive -File scripts/launch_seed.ps1` | 1 | 1 (`MissingMandatoryParameter`) |
| populated run | `... -File scripts/launch_seed.ps1 -Seed 53` | 3 | 3, stderr `refused: …FORMAL_2148a577_seed53 already exists` |
| child process | `Start-Process powershell … -Wait -PassThru` on the same seed 53 command | 3 | `CHILD_EXIT=3` |
| preflight failure | same command with `GRB_LICENSE_FILE` pointing at a missing file | 10 or 3 | 10, `Python, dependency or Gurobi licence check failed`, `Preflight refused; seed 53 was not started` |
| contract failure | `scripts/preflight.ps1 -ConfigDir <temp copy with minimum_rounds: 0>` | 11 | 11, violation `minimum episodes must equal minimum rounds times train days` |

The populated-run refusal happens inside `run_formal_training.py` before any
child process starts, so seed 53's directory was read only. The temp config
copy lived outside the repository; `formal_training_v2` was not modified.

After every test, `runs/FORMAL_2148a577_seed53` re-hashed identical and
`runs/FORMAL_2148a577_seed71` did not exist (`guardrail_after.json`).

## Observations

- `launch_seed.ps1` has no log tee. Output goes to the console only; the
  earlier formal launches wrote their own logs beside the manifest. Observation,
  not a failure.
- The wrapper returns the child's exit code through `exit $LASTEXITCODE` inside
  `try`/`finally`; the `Start-Process` test confirms the code survives.

## Protocol finding: recorded launches omit `--max-episodes`

`.claude/rules/training_contract.md` says the launcher must pass
`--min-rounds`, `--exposures` and `--max-episodes`, and that a command omitting
any of them "is not a formal launch".

The recorded commands are:

- `formal_training_v2/seed11_launch.log` line 1: seed 11, `--exposures 15
  --early-stop-patience 3 --min-rounds 10`, no `--max-episodes`;
- `formal_training_v2/remaining_seeds_launch.log` lines 1 and the seed 23/37/53
  headers: the same shape, no `--max-episodes`.

The current `src/run_formal_training.py` `command()` does pass
`--max-episodes 795`; the dry run above shows it.

Numerically the omission was inert: 53 training days × 15 exposures = 795,
which equals `episodes_per_seed_max`, so the default `--max-episodes 0` produced
the same cap. All three completed seeds stopped at the 10-round floor, and seed
53 failed before any validation. It is still a literal deviation from the frozen
launch clause and is reported as such, not silently accepted.

Verdict: `INFERRED` — the recorded runs satisfy the budget but not the written
launch-form requirement. Whether that invalidates them is a protocol decision
outside CC-02.
