# Local differential reward vs full-day settlement (V2-CY-JN-2025)

Diagnostic on **training and validation days only**; the common test set was not used. The saved actors are replayed deterministically and each day is then settled with its own 96-period quotes.

## The identity

For every day the following is exact by construction:

    day_diff_vs_B1 = sum_local_diff + mechanism_term + state_path_term

where `sum_local_diff` is the sum of the block rewards, `mechanism_term = settled - sum_raw` (rolling clears vs the one-day settlement) and `state_path_term = sum_local_baseline - B1` (the truthful arm starting from the policy's own SOC path versus B1 starting at 0.5).

| run | days | mean sum_local_diff | mean day_diff | mean mechanism | mean state_path | corr(local, day) | positive-block share | days above B1 | max SOC divergence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL96 seed 11 | 12 | -45.62 | -60.58 | 114.15 | -129.11 | 0.42 | 575/1152 | 0 | 0.2717 |
| FULL96 seed 22 | 12 | 58.42 | -97.34 | 8.07 | -163.83 | -0.39 | 645/1152 | 0 | 0.2717 |
| NEAR seed 11 | 12 | 89.01 | -21.08 | 81.91 | -192.00 | -0.87 | 768/1152 | 0 | 0.2716 |
| NEAR seed 22 | 12 | 42.55 | -96.13 | 26.12 | -164.80 | -0.66 | 698/1152 | 0 | 0.2717 |

The identity holds to 1.14e-13 CNY across every day, so the arithmetic is consistent. Across all sampled run-days the correlation between the summed local reward and the day-level difference is **-0.32**.

## Examples: local positive while the day is below B1

| run | load_date | sum_local_diff | day_diff_vs_B1 | mechanism | state_path |
|---|---|---:|---:|---:|---:|
| FULL96/22 | 2022-10-20 | 2.76 | -505.81 | -42.75 | -465.82 |
| NEAR/22 | 2022-10-20 | 126.73 | -463.28 | -113.56 | -476.45 |
| NEAR/22 | 2022-12-12 | 426.14 | -259.69 | 168.92 | -854.75 |
| NEAR/22 | 2022-11-05 | 158.66 | -254.25 | 163.59 | -576.49 |
| FULL96/22 | 2022-11-05 | 144.23 | -251.44 | 123.85 | -519.53 |
| FULL96/22 | 2022-12-12 | 447.13 | -240.95 | 194.67 | -882.76 |
| NEAR/11 | 2022-10-20 | 725.49 | -179.75 | -77.79 | -827.45 |
| FULL96/11 | 2022-12-12 | 140.80 | -159.86 | 476.29 | -776.95 |

## Which cause is it?

The three candidates are separated as follows.

**Computation error: ruled out.** The state-matching gate passes 9/9 (same window, current SOC, same wholesale and config, truthful committed block, opponents unchanged), each agent's baseline comes from the one shared counterfactual clear at its own bus LMP, and the identity above closes to machine precision. Nothing here shows a mismatched state or a mis-signed term.

**Definition mismatch: present.** The reward differences one block, and its truthful arm is *teleported into the policy's current state*. The policy is therefore credited for a block's profit without paying for the SOC it consumed to get there, while B1 must earn its whole path from 0.5. The `state_path_term` is exactly the part the local reward omits: it is the truthful arm's profit on the policy's SOC path minus its profit on its own. When a policy discharges early for a modest local gain, the local differential can be positive while B1's full-day schedule earns more.

**Real cross-period opportunity cost: present and measurable.** The maximum committed SOC divergence between the policy and its counterfactual is in the table; it is the policy's own doing, not an artefact. The `mechanism_term` is the second gap: the training arm is a sequence of 16-period rolling clears that commit four periods, while the evaluation and B1 are one whole-day clear. That difference is a property of the evaluation design and is reported rather than attributed to the reward.

**Conclusion.** A positive cumulative local reward coexisting with a lower day-level payoff is expected under this definition. It is not a bug and not only an implementation artefact; it is the local reward measuring a state-matched block advantage while the day benchmark measures a whole path from the day-start state. The two are different quantities and must not be substituted for each other.

## Limits

- Twelve sampled days per run (six train, six validation), not every training day.
- The `state_path_term` bundles the SOC path with the rolling-vs-full-day mechanism; the separate `mechanism_term` isolates the policy arm's share of the latter, but the two cannot be fully separated without a same-state whole-day truthful clear.
