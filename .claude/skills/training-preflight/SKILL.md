---
name: training-preflight
description: Use before starting, resuming or re-freezing a formal training seed. Runs the executable gate (environment, live manifests, budget contract, seed directory, data isolation) and reports GO or NO_GO with the evidence path.
allowed-tools: Bash, Read, Grep, Glob
---

# Training preflight

Run the gate; do not re-derive it.

## Steps

1. Run the wrapper:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/preflight.ps1
   ```

2. Read the JSON it prints. The gate passes only when all three hold:

   - `checks.all_pass` is `true`
   - `checks.budget_contract_ok` is `true`
   - `status` is `GO_FORMAL_TRAINING`

   The live manifest is `reports/v2_cy_jn_2025/formal_training_v3`
   (config hash `1b47e0ed`, `protocol_version: 3`). `formal_training_v2` and
   `formal_training_ready` are historical and are never selected by default.

3. If any check is false, report `NO_GO` with:

   - the failing check name;
   - `contract_violations` as-is;
   - the exact command that produced it.

   Do not repair the config, delete a run directory or pass `--config-dir`
   to a different version to make the gate green.

## Manual equivalent

When the wrapper is unavailable:

```bash
PYTHONPATH=src python src/training_contract.py --check
PYTHONPATH=src python src/run_formal_training.py --check
```

Both must exit 0. The second also verifies data, config and source hashes.

## Boundaries

- This skill never starts training. `scripts/launch_seed.ps1 -Seed N` does.
- This skill never modifies a frozen file, manifest, checkpoint or log.
- A missing Gurobi licence is `BLOCKED`, not a reason to skip the gate.
