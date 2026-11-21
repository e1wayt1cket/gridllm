# Storage cycle-cost sensitivity

## Verdict

**Diagnostic only.** The load day and the price day are different calendar dates, and the storage quotes in experiment A are truthful. This is not a same-day market replay, not a training result and not a basis for lifting the V2 price-data gate.

Measurable cycling appears only at C80. Fleet net profit is C80 +0.0899 CNY, C90 -0.0016 CNY, C100 -0.0001 CNY. The gross margin is almost entirely consumed by degradation; this is not a meaningful arbitrage opportunity.
Higher cycle cost never raised the optimal net profit by more than 0.01 CNY: **yes**.
The exact comparison flags C90->C100 by +0.001576 CNY, which is solver-noise scale (below the 0.01 CNY tolerance). Both cases are idle and their profits are zero within the solver's precision.

**Sufficiency for a MATD3 cost-sensitivity study: not yet.** At C90 and C100 the executable LP bound is zero, so there is no arbitrage signal to learn. At C80 the truthful clearing earns 0.09 CNY on 0.1085 MWh. A formal training study needs a day with a material executable spread, and it needs the dated price gate to be met first.

## Cost definition and audit

- Config: `storage.cycle_cost`, CNY/MWh **per leg**.
- Charged on both charge and discharge: `cycle_cost * (p_ch + p_dis) * DT_HOURS`, with `DT_HOURS = 0.25`.
- The same expression is used in the clearing objective and in `participant_payoff._ledger`; there is no separate settlement formula.
- `eta_ch` and `eta_dis` enter only through the SOC transition. They are not a money term, so efficiency is not charged twice.
- Break-even spread per MWh stored: `required_margin = cycle_cost * (1/eta_ch + eta_dis)`. It is not `2 * cycle_cost` unless the round trip is lossless.

| case | cycle cost | required margin CNY/MWh |
|---|---:|---:|
| C80 | 80.00 | 160.56 |
| C90 | 90.00 | 180.63 |
| C100 | 100.00 | 200.70 |

### Breakeven-ratio audit

`diagnose_typical_day._window` previously divided the valley term by `eta_ch * eta_dis` and then divided the whole numerator by `eta_dis * valley`, which is equivalent to an extra `eta_dis`. The formula is corrected in this run; the old and corrected values are reported below. `round_trip_margin_cny_mwh` was already correct, so the CNY/MWh siting criterion did not move.

| case | agent | old breakeven ratio | corrected ratio | old CNY/MWh | corrected CNY/MWh | round-trip CNY/MWh |
|---|---|---|---|---|---|---|
| C80 | ESS9 | 2.0417 | 1.9389 | -11.12 | 10.65 | 10.65 |
| C80 | ESS14 | 2.0400 | 1.9372 | -9.79 | 12.04 | 12.04 |
| C80 | ESS17 | 2.0386 | 1.9359 | -8.68 | 13.18 | 13.18 |
| C80 | ESS27 | 2.0253 | 1.9226 | -14.81 | 7.45 | 7.45 |
| C90 | ESS9 | 2.1364 | 2.0336 | -31.19 | -9.42 | -9.42 |
| C90 | ESS14 | 2.1345 | 2.0317 | -29.86 | -8.03 | -8.03 |
| C90 | ESS17 | 2.1329 | 2.0302 | -28.75 | -6.89 | -6.89 |
| C90 | ESS27 | 2.1180 | 2.0152 | -34.88 | -12.62 | -12.62 |
| C100 | ESS9 | 2.2310 | 2.1283 | -51.26 | -29.49 | -29.49 |
| C100 | ESS14 | 2.2289 | 2.1262 | -49.93 | -28.10 | -28.10 |
| C100 | ESS17 | 2.2272 | 2.1245 | -48.82 | -26.96 | -26.96 |
| C100 | ESS27 | 2.2106 | 2.1079 | -54.95 | -32.69 | -32.69 |

At C80 the sign flips: the old ratio reported a negative price-only margin while the corrected ratio and `round_trip_margin_cny_mwh` report positive. The executable LP bound and the actual clearing are still small, so this does not create a material opportunity; it does change the ratio-space `by_bus` viability list. The CNY/MWh siting criterion is unchanged.

## Inputs and date match

- Scenario: `typical_day_industry_subset`.
- Load date: **2022-11-30**.
- Price date: **2026-10-09**.
- Case type: `cross_date_diagnostic`; same day: **False**.
- Ledger kind: `formal_ledger`; unit status: `unit_basis_ruling`.
- The pair is a cross-date diagnostic. It is not a historical replay, and this experiment does not modify the frozen V2 manifest or release the V2 price gate.

## Experiment A: Gurobi optimal dispatch

Only `storage.cycle_cost` changes; load, price, fleet, efficiency, SOC band, initial/terminal SOC, power limits, network and solver settings are identical. Quotes are truthful in every arm.

| case | cost | required margin | charge MWh | discharge MWh | cycles | net CNY | delta system CNY | idle share | solver | clean |
|---|---|---|---|---|---|---|---|---|---|---|
| C80 | 80.0 | 160.56 | 0.1282 | 0.1085 | 0.0185 | 0.0899 | 6.80 | 0.896 | OPTIMAL | yes |
| C90 | 90.0 | 180.63 | 0.0000 | 0.0000 | 0.0000 | -0.0016 | 5.09 | 1.000 | OPTIMAL | yes |
| C100 | 100.0 | 200.70 | 0.0000 | 0.0000 | 0.0000 | -0.0001 | 5.02 | 1.000 | OPTIMAL | yes |
| NoESS | - | - | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | - | OPTIMAL | yes |

No-ESS system settlement total: **18271.90 CNY**. This is the project's participant-payoff total, not cash welfare; prosumer totals include load valuation. The per-case system delta is not the fleet's profit: even an idle fleet changes the clearing's constraint set, so the system total moves while the fleet's own settlement stays near zero.

Because No-ESS carries no storage, the fleet's own net profit in the table below is already its increment over No-ESS. The system delta is a separate total and must not be read as storage revenue.

### Fleet economics

| case | charge purchase CNY | discharge revenue CNY | cycle cost CNY | other cost CNY | net CNY |
|---|---|---|---|---|---|
| C80 | 30.41 | 49.43 | 18.93 | 0.00 | 0.0899 |
| C90 | 0.01 | 0.01 | 0.01 | 0.00 | -0.0016 |
| C100 | 0.00 | 0.00 | 0.00 | 0.00 | -0.0001 |

### Per-agent dispatch

| case | agent | charge MWh | discharge MWh | ch periods | dis periods | cycles | net CNY | SOC init | SOC final | SOC min | SOC max | max ch MW | max dis MW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C80 | ESS9 | 0.0375 | 0.0317 | 1 | 3 | 0.0216 | -0.0000 | 0.500 | 0.500 | 0.500 | 0.522 | 0.1498 | 0.0533 |
| C80 | ESS14 | 0.0270 | 0.0229 | 1 | 3 | 0.0156 | -0.0001 | 0.500 | 0.500 | 0.500 | 0.516 | 0.1080 | 0.0531 |
| C80 | ESS17 | 0.0244 | 0.0206 | 2 | 4 | 0.0141 | -0.0001 | 0.500 | 0.500 | 0.500 | 0.514 | 0.0660 | 0.0277 |
| C80 | ESS27 | 0.0393 | 0.0333 | 1 | 5 | 0.0227 | 0.0901 | 0.500 | 0.500 | 0.500 | 0.523 | 0.1574 | 0.0607 |
| C90 | ESS9 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | -0.0004 | 0.500 | 0.500 | 0.500 | 0.500 | 0.0000 | 0.0000 |
| C90 | ESS14 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | -0.0004 | 0.500 | 0.500 | 0.500 | 0.500 | 0.0000 | 0.0000 |
| C90 | ESS17 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | -0.0005 | 0.500 | 0.500 | 0.500 | 0.500 | 0.0000 | 0.0000 |
| C90 | ESS27 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | -0.0003 | 0.500 | 0.500 | 0.500 | 0.500 | 0.0000 | 0.0000 |
| C100 | ESS9 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | -0.0000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.0000 | 0.0000 |
| C100 | ESS14 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | -0.0000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.0000 | 0.0000 |
| C100 | ESS17 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | -0.0000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.0000 | 0.0000 |
| C100 | ESS27 | 0.0000 | 0.0000 | 0 | 0 | 0.0000 | -0.0000 | 0.500 | 0.500 | 0.500 | 0.500 | 0.0000 | 0.0000 |

Generation, unserved-load and other operating costs are zero for an independent battery, so the whole operating cost is the two-leg cycle cost. The terminal state is pinned at the initial 0.50 in every case.


### Solver and physical checks

| case | solver | gap | time s | terminal SOC | SOC band | power | voltage | line | no churn | nodal LMP |
|---|---|---|---|---|---|---|---|---|---|---|
| C80 | OPTIMAL | 0.0000 | 1.45 | yes | yes | yes | yes | yes | yes | yes |
| C90 | OPTIMAL | 0.0000 | 1.31 | yes | yes | yes | yes | yes | yes | yes |
| C100 | OPTIMAL | 0.0000 | 1.26 | yes | yes | yes | yes | yes | yes | yes |
| NoESS | OPTIMAL | 0.0000 | 0.82 | - | - | - | yes | yes | - | - |

The SOCP is continuous, so Gurobi's MIP gap is not defined. An OPTIMAL final status means the reported objective is the optimum; the gap column records 0 for that case and is not a MIP-gap claim.

## Experiment B: theory versus executable bound

The theoretical spread is evaluated at each ESS bus against the **No-ESS nodal LMP**. The price-only candidate pairs ignore SOC, power and the terminal pin; the LP bound imposes all three and is therefore the executable opportunity. The fleet LP is the sum of four independent per-unit bounds and is not a joint fleet optimum.

| case | agent | LMP min/max/mean | required | price-only margin | LP bound CNY | LP charge | LP discharge | actual charge | actual discharge |
|---|---|---|---|---|---|---|---|---|---|
| C80 | ESS9 | 230.40/458.31/397.61 | 160.56 | 10.65 | 1.23 | 0.2000 | 0.1693 | 0.0375 | 0.0317 |
| C80 | ESS14 | 230.91/460.42/398.68 | 160.56 | 12.04 | 1.47 | 0.2000 | 0.1693 | 0.0270 | 0.0229 |
| C80 | ESS17 | 231.34/462.17/399.56 | 160.56 | 13.18 | 1.67 | 0.2000 | 0.1693 | 0.0244 | 0.0206 |
| C80 | ESS27 | 235.48/460.82/397.61 | 160.56 | 7.45 | 0.69 | 0.1000 | 0.0846 | 0.0393 | 0.0333 |
| C90 | ESS9 | 230.40/458.31/397.61 | 180.63 | -9.42 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C90 | ESS14 | 230.91/460.42/398.68 | 180.63 | -8.03 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C90 | ESS17 | 231.34/462.17/399.56 | 180.63 | -6.89 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C90 | ESS27 | 235.48/460.82/397.61 | 180.63 | -12.62 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C100 | ESS9 | 230.40/458.31/397.61 | 200.70 | -29.49 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C100 | ESS14 | 230.91/460.42/398.68 | 200.70 | -28.10 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C100 | ESS17 | 231.34/462.17/399.56 | 200.70 | -26.96 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C100 | ESS27 | 235.48/460.82/397.61 | 200.70 | -32.69 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Fleet bound versus actual

| case | LP bound CNY | actual net CNY | LP discharge | actual discharge | LP cycles | actual cycles | actual - LP |
|---|---|---|---|---|---|---|---|
| C80 | 5.06 | 0.09 | 0.5925 | 0.1085 | 0.1010 | 0.0185 | -4.97 |
| C90 | 0.00 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.00 |
| C100 | 0.00 | -0.00 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.00 |

The C80 gap between the 5.06 CNY LP bound and the 0.09 CNY actual net is not a solver error. The LP is a price-taker bound: one unit, perfect foresight, and the No-ESS nodal prices as exogenous inputs. The actual figure is a joint truthful clearing in which the fleet's own quotes and the nodal re-solve determine dispatch, and the four units share the same spread. The bound discharges 0.5925 MWh; the clearing dispatches 0.1085 MWh. A perfect forecaster outside the market can see more than the market itself captures.

Candidate price pairs (price-only, no energy limits):

| case | agent | pairs clearing the floor | best pair (margin) |
|---|---|---|---|
| C80 | ESS9 | 15 | 45->74 (10.65 CNY/MWh) |
| C80 | ESS14 | 17 | 45->74 (12.04 CNY/MWh) |
| C80 | ESS17 | 21 | 45->74 (13.18 CNY/MWh) |
| C80 | ESS27 | 10 | 45->88 (7.45 CNY/MWh) |
| C90 | ESS9 | 0 | - |
| C90 | ESS14 | 0 | - |
| C90 | ESS17 | 0 | - |
| C90 | ESS27 | 0 | - |
| C100 | ESS9 | 0 | - |
| C100 | ESS14 | 0 | - |
| C100 | ESS17 | 0 | - |
| C100 | ESS27 | 0 | - |

## Why the fleet is idle or cycling

- **C80**: the LP bound is +5.06 CNY and actual net profit is +0.09 CNY. Gross margin is almost entirely consumed by degradation; the opportunity is real but negligible.
- **C90**: the executable LP bound is +0.00 CNY, so a perfect forecaster cannot clear the degradation floor on this price field. Idleness is market-driven, not a policy failure.
- **C100**: the executable LP bound is +0.00 CNY, so a perfect forecaster cannot clear the degradation floor on this price field. Idleness is market-driven, not a policy failure.

The raw min-max spread is not the reachable spread: 0.4 MW and 1.6 MWh with a 0.10-0.90 SOC band cannot pair the day's extremes, and the terminal pin forbids ending with banked energy.

## Experiment C: the MATD3 smoke checkpoint

**One policy, three costs, not three trained policies.** The checkpoint is a smoke artifact: 25 episodes, 5 gradient updates, seed 42, trained at C100 on a non-formal scenario. It does not demonstrate convergence. C80 and C90 are out-of-distribution costs; the policy is not retrained.

The checkpoint's training recipe used `use_nodal_price=false`; this diagnostic market uses the scenario YAML value `true`. The comparison below therefore describes this checkpoint in this market, not a reproduction of its training environment.

| case | declared charge mean | declared discharge mean | charge MWh | discharge MWh | net CNY | delta vs truthful | delta vs LP | status |
|---|---|---|---|---|---|---|---|---|
| C80 | 39.76 | 56.90 | 0.0000 | 0.0000 | -0.0001 | -0.0900 | -5.0629 | ok |
| C90 | 39.72 | 56.89 | 0.0000 | 0.0000 | -0.0001 | 0.0016 | -0.0001 | ok |
| C100 | 39.72 | 56.89 | 0.0000 | 0.0000 | -0.0001 | 0.0000 | -0.0001 | ok |

Settlement status: C80 OPTIMAL (6.01 s rollout + settlement), C90 OPTIMAL (6.10 s rollout + settlement), C100 OPTIMAL (6.15 s rollout + settlement). Every case is clean; none fell back.

If the policy is idle where the truthful Gurobi arm cycles, the difference is a policy gap to locate. It is not evidence that the algorithm failed, and it is not evidence that MATD3 learned or failed to learn arbitrage.

At C80 the truthful Gurobi arm discharges 0.1085 MWh for +0.09 CNY, while the checkpoint discharges 0.0000 MWh for -0.00 CNY. Its mean declared charge price is 39.76 CNY/MWh and its mean discharge price is 56.90 CNY/MWh. At most 0.0% of the declared values sit at the lower box bound and at most 0.0% at the upper bound. The checkpoint was trained at C100, on a day whose executable bound is zero, so the training signal contained no arbitrage to learn. The idle C80 behaviour is therefore a policy gap to locate, not a demonstration that MATD3 cannot arbitrage.

## Prior record and reconciliation

The earlier C0/C1/C2 record is preserved at `reports/storage_cost_sensitivity.md` and its CSV is not overwritten. That run used `run_arm`'s rolling loop followed by an endpoint-pinned settlement; this run clears the whole day directly with truthful quotes. C100 is the cross-check.

| prior case | cost | charge MWh | discharge MWh | gross CNY | degradation CNY | net CNY |
|---|---|---|---|---|---|---|
| C0 | 100.0 | 0.0000 | 0.0000 | 0.00 | 0.00 | -0.00 |
| C1 | 90.0 | 0.0000 | 0.0000 | 0.00 | 0.01 | -0.00 |
| C2 | 80.0 | 0.1282 | 0.1085 | 19.02 | 18.93 | 0.09 |

The new C100/C90/C80 net profits (-0.000072, -0.001648, +0.089879 CNY) reproduce the prior C0/C1/C2 (-0.000072, -0.001648, +0.089879 CNY) to the shown precision. The direct full-day truthful clear and the old rolling-then-settled path therefore agree on this day.

## Limitations

- One frozen cross-date diagnostic day; no claim about a multi-day distribution or a historical market.
- Experiment A is truthful bidding, not learned bidding.
- Experiment C is one smoke checkpoint, not a trained comparison.
- The LP bound uses No-ESS nodal prices as exogenous inputs and ignores the price feedback the fleet itself creates; it is an upper bound for one unit at a time, not a joint fleet optimum.
- The No-ESS arm removes the independent fleet; prosumer and non-prosumer storage are disabled in `defaults.yaml`, so no other storage remains.

## Next experiments

- Repeat on the V2 multi-day library only after the dated price gate is met; do not reuse this cross-date day as evidence.
- A same-day load/price replay when a matching pair exists.
- Retrain at C80/C90/C100 before comparing learned policies across costs; the current checkpoint was trained at C100 only.
- Sensitivity to efficiency, the SOC band and the action box, and a joint fleet LP that prices the fleet's own effect on LMP.

## Artifacts and command

- Snapshot: `reports\snapshots\storage_cycle_cost_20261027_113958`.
- Report: `reports\storage_cycle_cost_sensitivity.md`.
- Command: `C:\ProgramData\anaconda3\python.exe src/storage_cycle_cost_sensitivity.py`.
