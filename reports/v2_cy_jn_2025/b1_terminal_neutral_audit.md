# Terminal-neutral B1 and feasibility audit

Every RL storage unit starts and must end the day at SOC 0.5. The constraint is absolute and is applied only to the rolling windows whose own end is the day's last period (`horizon_type=day_end`); earlier windows remain unpinned. The truthful arm uses the same 24-block interface, initial SOC, load, price, network, cycle cost and block settlement as the policy arm.

| item | value |
|---|---|
| terminal target | 0.5 |
| days run | 96 |
| feasible (24 clears, 0 fallbacks, end SOC within 1e-5) | 96 |
| infeasible | 0 |
| max per-unit terminal error | 0.000e+00 |
| days with a voltage violation | 0 |
| days with a line overload | 0 |
| total network losses (MWh) | 269.785 |
| NEAR records found by pair_id | 80 / 80 |
| NEAR pair ids absent from FULL96 | [] |

The earlier rolling B1 (`b1_daily_results.csv`) did not pin terminal SOC and is kept unchanged as `CASHFLOW_NO_TERMINAL`. It is not the day-level benchmark for the terminal-neutral experiment.

## Infeasible days

None. Every day completed 24 clears with 0 fallbacks and ended each unit on the target.
