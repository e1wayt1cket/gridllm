# Storage cycle-cost snapshot

Created 2026-10-27T11:39:58.647747.

**Diagnostic only.** The load day and the price day are different calendar dates; this is not a same-day market replay and it does not lift the V2 price-data gate.

## Results

| case | cost CNY/MWh/leg | required margin | charge MWh | discharge MWh | net CNY |
|---|---:|---:|---:|---:|---:|
| C80 | 80.0 | 160.56 | 0.1282 | 0.1085 | 0.0899 |
| C90 | 90.0 | 180.63 | 0.0000 | 0.0000 | -0.0016 |
| C100 | 100.0 | 200.70 | 0.0000 | 0.0000 | -0.0001 |
| NoESS | - | - | 0.0000 | 0.0000 | 18271.90 system |

## Files

- `config.json` - effective configuration, inputs and software.
- `results.json` - raw key results for experiments A, B and C.
- `results.csv` - one tidy row per case/arm/agent and fleet.
- `dispatch_trace.csv` - per-period price, power, SOC and money.
- `run.log` - command, start/end, warnings and solver metadata.
- `hashes.json` - input and output SHA-256, git state.

The formal report is `reports\storage_cycle_cost_sensitivity.md`.

## Caveats

- One frozen cross-date diagnostic day, truthful quotes for experiment A.
- The MATD3 block, when present, is one smoke checkpoint evaluated at three costs; it is not three trained policies.
- The LP bound uses the No-ESS nodal prices as exogenous inputs and ignores price feedback from storage.
