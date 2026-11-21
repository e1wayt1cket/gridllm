# The single-day diagnostic clearing

One real clearing run on the single-day case: the real metered industry
load from the node-load ledger, against the real `261009` price curve,
through `single_day_data`. This report is the output of
`src/run_single_day_diagnostic.py`; every number in it is a solver
output and none has been adjusted.

> **This is not a historical replay.** The load is dated **2022-11-30** and the price **2026-10-09**
> — different days, paired by decision as a `cross_date_diagnostic`.
> Nothing here is what either day actually met in its own market, and no
> revenue, profit or welfare figure computed from this run describes
> anything that happened.

## The verdict, first

**The network cannot carry this load.** It served **354.8 MWh** of the **10,995.6 MWh** demanded — **3.23%** — and shed **10,640.8 MWh**.

This is reported as the result, not treated as a failure of the
run. The load was not scaled, no line was widened, no band was
relaxed and no storage parameter was moved to change it.

| | |
|---|---|
| case type | `cross_date_diagnostic` |
| load date | 2022-11-30 |
| price date | 2026-10-09 |
| same day | **False** |
| load | `data/processed/node_loads/typical_day_industry/bus_loads_96x33_audit.csv` |
| price | sheet `261009` from `C:\Projects\00_WorkSpace\01_Projects\MakerB\docs\日前价格261009.xlsx` |
| unit basis | **user_confirmed_project_ruling** — `kW`, evidence `ruling_only` |
| externally verified | **False** |

## What the solver returned

| | |
|---|---|
| status | `OPTIMAL` (optimal: True) |
| fell back to a single-period engine | False |
| nodal-price fallbacks | 0 |
| price field | `period index to period index, no shift, no resampling` |
| clearing objective | -26,504,339.90 |
| renewable consumption rate | 99.07% |
| carbon | 207.51 tCO2 |

## Network state

| | |
|---|---|
| bus voltage band | 0.93 .. 1.05 pu |
| voltage min | **0.9300** at bus 22, period 82 |
| voltage max | 1.0000 |
| voltage sitting on the lower bound | **True** |
| bus-periods outside the band | 0 |
| worst line utilization | **1.0000** on line 1, period 66 |
| line sitting at its limit | **True** |
| lines at or over their limit | 45 (of 0) |
| PCC import | max 18.74 MW, mean 14.91 MW |
| PCC export | max 0.00 MW, over 0 periods (limit 2.5 MW) |

### What is binding

The clearing is held by the lower voltage bound, at 0 bus-periods outside the band and by a line's thermal limit, with 45 line-periods at or over it. A shed figure of this size is therefore a statement about the **network's
capacity at this load**, not about the price or about the fleet:
the market shed load it could not deliver, and it could not deliver
it because the feeder's voltage and thermal limits are already
reached.

## What this run does not establish

- It is **not** a historical replay, and no revenue, profit or
  arbitrage figure from it describes anything that happened.
- The load rests on a **user-confirmed project ruling** for the reading
  unit (`kW`), not on an external document. `is_externally_verified` is
  `false` and no figure here may be presented as externally certified.
- The price curve is a **spatial mean** read as a single day **by
  project ruling**; it represents no bus or zone.
- It is **not** a formal physical network case: the formal ledger is
  still refused, and this run is a diagnostic on the audit ledger.

## Reproduce

```
PYTHONPATH=src python src/run_single_day_diagnostic.py
```
