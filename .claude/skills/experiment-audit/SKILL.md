---
name: experiment-audit
description: Use when auditing a completed training run directory for budget compliance, early-stop correctness, checkpoint integrity, data isolation and report claims. Read-only; produces a verdict using PASS, FAIL, NOT_TESTED, INFERRED or BLOCKED.
allowed-tools: Bash, Read, Grep, Glob
---

# Experiment audit

Separate evidence collection from judgment: scripts collect, the read-only
experiment-auditor judges.

## Steps

1. Collect evidence without touching the run:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/audit_run.ps1 -RunDir <run-dir>
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/finalize_run.ps1 -RunDir <run-dir>
   ```

   `audit_run.ps1` prints the checks. `finalize_run.ps1` runs the same checks
   and writes the evidence JSON to `reports/cc01/audit/<run-id>.json`; it
   refuses a run that has not terminated and never writes inside the run.

2. Delegate the judgment to the `experiment-auditor` agent, passing the run
   directory and the evidence file path. That agent has no write tools.

3. Report one verdict per item, then one overall verdict.

## Checks

- run status and termination reason;
- rounds, episodes and block steps against the frozen budget
  (`minimum_rounds`, `episodes_per_seed_min/max`, patience), with optimizer
  updates reported alongside them;
- early stop did not fire before the round floor;
- best and final checkpoints exist, differ and match their manifests;
- validation history matches the reported best round and value;
- voltage, line and LMP fallback counts;
- terminal SOC error;
- test, holiday and H2 were not read;
- every number in the run's own report is traceable to an artifact.

## Verdict rules

- A budget shortfall is `FAIL`, even when every checkpoint is intact.
- An intact run with a protocol violation is still `FAIL` for the protocol.
- Missing artifacts are `NOT_TESTED` or `BLOCKED`, never `PASS`.
- The auditor never edits, renames or deletes anything under the run
  directory, and never re-runs training.
