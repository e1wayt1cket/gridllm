# Storage cost sensitivity

## Verdict

This is a **diagnostic on one frozen load/price day with truthful bidding**, not a training result. No OBS_V6 checkpoint exists in the repository, so learned reward is not reported and the differential reward is zero by construction.

C0 and C1 do not cycle beyond solver noise; C2 begins to cycle but settles only +0.09 CNY, effectively break-even. On this day a lower cycle cost is the first case that produces measurable cycling, not evidence of a meaningful arbitrage strategy.

## What the cost is

- Name: `storage.cycle_cost`.
- Value: **100 CNY/MWh per leg** in `config/defaults.yaml`.
- Units and timing: CNY per MWh moved, charged once on charge and once on discharge, and multiplied by the 0.25 h period.
- Decision path: subtracted from the SOCP/LDF dispatch objective.
- Settlement path: `participant_payoff._ledger` sets `degradation_cost = cycle_cost * (p_ch + p_dis) * dt` and subtracts it from the participant's total.
- Break-even: `required_margin = cycle_cost * (1/eta_ch + eta_dis)`; the three values are in the case table below.
- There is no capex, fixed O&M or separate efficiency charge in the code. Efficiency enters through the SOC transition and the margin.
- Fleet: 4 x 1.6 MWh / 0.4 MW at buses 9/14/17/27, eta 0.92/0.92, SOC band 0.10-0.90, endpoints pinned by settlement.

## Cases

| case | multiplier | cycle_cost CNY/MWh | required margin CNY/MWh |
|---|---:|---:|---:|
| C0 | 1.00 | 100.00 | 200.70 |
| C1 | 0.90 | 90.00 | 180.63 |
| C2 | 0.80 | 80.00 | 160.56 |

Only the cycle cost changes; the load, price, fleet, efficiency, SOC band and market rules are identical.

## Fleet results

| case | charge MWh | discharge MWh | equivalent cycles | discharge utilisation | gross margin CNY | profit CNY | degradation CNY | SOC min | SOC max | SOC terminal | clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | -0.00 | 0.00 | 0.500 | 0.500 | 0.500 | yes |
| C1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | -0.00 | 0.01 | 0.500 | 0.500 | 0.500 | yes |
| C2 | 0.128 | 0.108 | 0.018 | 0.003 | 19.02 | 0.09 | 18.93 | 0.500 | 0.523 | 0.500 | yes |

## Per-agent results

| case | agent | charge MWh | discharge MWh | cycles | utilisation | profit CNY | degradation CNY | SOC range |
|---|---|---:|---:|---:|---:|---:|---:|---|
| C0 | ESS9 | 0.000 | 0.000 | 0.000 | 0.000 | -0.00 | 0.00 | 0.500–0.500 |
| C0 | ESS14 | 0.000 | 0.000 | 0.000 | 0.000 | -0.00 | 0.00 | 0.500–0.500 |
| C0 | ESS17 | 0.000 | 0.000 | 0.000 | 0.000 | -0.00 | 0.00 | 0.500–0.500 |
| C0 | ESS27 | 0.000 | 0.000 | 0.000 | 0.000 | -0.00 | 0.00 | 0.500–0.500 |
| C1 | ESS9 | 0.000 | 0.000 | 0.000 | 0.000 | -0.00 | 0.00 | 0.500–0.500 |
| C1 | ESS14 | 0.000 | 0.000 | 0.000 | 0.000 | -0.00 | 0.00 | 0.500–0.500 |
| C1 | ESS17 | 0.000 | 0.000 | 0.000 | 0.000 | -0.00 | 0.00 | 0.500–0.500 |
| C1 | ESS27 | 0.000 | 0.000 | 0.000 | 0.000 | -0.00 | 0.00 | 0.500–0.500 |
| C2 | ESS9 | 0.037 | 0.032 | 0.022 | 0.003 | -0.00 | 5.53 | 0.500–0.522 |
| C2 | ESS14 | 0.027 | 0.023 | 0.016 | 0.002 | -0.00 | 3.99 | 0.500–0.516 |
| C2 | ESS17 | 0.024 | 0.021 | 0.014 | 0.002 | -0.00 | 3.60 | 0.500–0.514 |
| C2 | ESS27 | 0.039 | 0.033 | 0.023 | 0.003 | 0.09 | 5.81 | 0.500–0.523 |

## Network

| case | max line util. | periods ≥0.99 | min V | max V | curtailment | RE rate | objective |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.1889 | 0 | 0.9480 | 1.0166 | 2.3351 | 85.9235 | 19451.7365 |
| C1 | 0.1889 | 0 | 0.9480 | 1.0166 | 2.3352 | 85.9233 | 19451.7225 |
| C2 | 0.1889 | 0 | 0.9481 | 1.0166 | 2.3352 | 85.9229 | 19452.2009 |

## Price and limits

- Day-ahead wholesale price: min 225.90, max 445.00, mean 386.83 CNY/MWh.
- Raw min-max spread: 219.10 CNY/MWh.
- Required margin C0: 200.70 CNY/MWh.
- Required margin C1: 180.63 CNY/MWh.
- Required margin C2: 160.56 CNY/MWh.
- The raw spread is not the reachable spread: a 0.4 MW / 1.6 MWh unit cannot pair the day's minimum and maximum periods through its SOC band, and efficiency and the two-leg cost sit between them.

## Reading

- C0 discharge: 0.0000 MWh (solver-noise scale), gross margin 0.00 CNY, net -0.00 CNY.
- C1 discharge: 0.0000 MWh, gross margin 0.00 CNY, net -0.00 CNY.
- C2 discharge: 0.1085 MWh, gross margin 19.02 CNY, degradation 18.93 CNY, net 0.09 CNY.
- C2 is the first case with measurable cycling, but its settled profit is essentially zero: the gross margin is almost entirely consumed by degradation. This day does not support lowering the cost further without first checking the reachable price spread, efficiency, SOC limits and market constraints.
- C0 and C1 do not cycle beyond solver noise on this day. Their required margins (200.70 and 180.63 CNY/MWh) are not cleared by a reachable cycle.
- No line reaches 99% utilisation and the SOC trajectory stays inside its band; neither the network nor the SOC limits bind on this day.

## Falsification

- C1 discharge throughput: no measurable cycling (both below 0.001 MWh).
- C1 settled profit: -0.00 CNY, within 1 CNY of C0 (-0.00 CNY).
- C2 discharge throughput: C0 is below 0.001 MWh; C2 reaches 0.1085 MWh, the first measurable cycling.
- C2 settled profit: +0.09 CNY, within 1 CNY of C0 (-0.00 CNY).

## Limitations

- One diagnostic day, one price curve, truthful quotes: this does not estimate a distribution over days or a learned policy.
- The price curve is dated 2026-10-09 and the load 2022-11-30; the pair is a cross-date diagnostic, not a historical replay.
- A learned (training) reward is not reported: with no OBS_V6 checkpoint, the truthful policy's differential reward is 0 by construction.
- The frozen subset's load varies by date; this day is the 3.653 MW diagnostic day, not a typical day.

## Artifacts

- `reports\storage_cost_sensitivity.csv`
- `reports\storage_cost_network.csv`

## Command

`src/storage_cost_sensitivity.py`
