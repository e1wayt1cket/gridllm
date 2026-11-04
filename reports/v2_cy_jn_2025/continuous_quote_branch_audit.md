# Continuous quote branch and boundary saturation (P1)

Only validation/training states are used. No test or H2.

## Final-round quote distributions (deterministic validation)

| gate | open freq | mean prob | open quote mean | open quote min | open quote max | lower-sat | upper-sat |
|---|---:|---:|---:|---:|---:|---:|---:|
| ESS9.charge | 0.00 | 0.05 | closed | - | - | - | - |
| ESS9.discharge | 1.00 | 0.95 | 101.86 | 85.17 | 121.50 | - | - |
| ESS14.charge | 0.39 | 0.40 | 48.94 | 21.93 | 110.93 | - | - |
| ESS14.discharge | 0.99 | 0.96 | 194.21 | 0.49 | 299.91 | - | - |
| ESS17.charge | 0.00 | 0.12 | closed | - | - | - | - |
| ESS17.discharge | 1.00 | 0.91 | 1.19 | 0.38 | 6.75 | - | - |
| ESS27.charge | 0.00 | 0.08 | closed | - | - | - | - |
| ESS27.discharge | 1.00 | 1.00 | 222.67 | 57.56 | 288.47 | - | - |

Validation lower/upper gate saturation: 264/79.

### Per-gate saturation and price-branch gradient (R10)

| gate | open decisions | lower-sat fraction | upper-sat fraction | mean quote | price-logit grad norm |
|---|---:|---:|---:|---:|---:|
| ESS9.charge | 0 | 0.00 | 0.00 | nan | 2.36e-07 |
| ESS9.discharge | 264 | 0.00 | 0.00 | 101.86 | 5.09e-05 |
| ESS14.charge | 104 | 0.00 | 0.00 | 48.94 | 1.16e-04 |
| ESS14.discharge | 262 | 0.05 | 0.30 | 194.21 | 8.25e-05 |
| ESS17.charge | 0 | 0.00 | 0.00 | nan | 1.50e-06 |
| ESS17.discharge | 264 | 0.95 | 0.00 | 1.19 | 4.10e-06 |
| ESS27.charge | 0 | 0.00 | 0.00 | nan | 4.43e-07 |
| ESS27.discharge | 264 | 0.00 | 0.00 | 222.67 | 1.68e-04 |

## ESS17 discharge and ESS14 branches

| day | R2 ESS17 | R10 ESS17 | R2 ESS14 | R10 ESS14 | R2 ESS27 | R10 ESS27 |
|---|---:|---:|---:|---:|---:|---:|
| 2022-12-30|2025-12-31 | 4.1 | 7.2 | 11.8 | -179.2 | 4.2 | -239.1 |
| 2022-12-31|2025-12-28 | 5.0 | 9.0 | -3.2 | -41.4 | 0.6 | -205.0 |
| 2022-12-13|2025-12-11 | 12.7 | 11.3 | 4.3 | -56.7 | -29.7 | -115.9 |
| 2022-10-26|2025-10-24 | -17.7 | 3.4 | -81.5 | -15.1 | -55.1 | -25.6 |
| 2022-07-23|2025-07-13 | 0.1 | 0.5 | 0.1 | 20.1 | 19.1 | 19.1 |

## Critic grid and actual settlement at the fixed state

| ckpt | day | target | leg | quote | min Q | twin gap | dQ/dquote | target payoff | fleet payoff |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| best | val | ESS17 | discharge | 0.0 | -0.552 | 0.043 | -2.25e-04 | 0.00022 | 0.00052 |
| best | val | ESS17 | discharge | 0.0 | -0.552 | 0.043 | -2.25e-04 | 0.00022 | 0.00052 |
| best | val | ESS17 | discharge | 75.0 | -0.524 | 0.119 | 9.89e-05 | 0.00002 | 0.00032 |
| best | val | ESS17 | discharge | 150.0 | -0.508 | 0.087 | -1.76e-04 | -0.00000 | 0.00029 |
| best | val | ESS17 | discharge | 225.0 | -0.532 | 0.071 | -7.76e-04 | -0.00001 | 0.00029 |
| best | val | ESS17 | discharge | 300.0 | -0.599 | 0.077 | -1.10e-03 | -0.00001 | 0.00028 |
| final | val | ESS17 | discharge | 0.0 | -0.236 | 0.004 | 1.21e-05 | 0.00018 | 0.00017 |
| final | val | ESS17 | discharge | 0.0 | -0.236 | 0.004 | 1.21e-05 | 0.00018 | 0.00017 |
| final | val | ESS17 | discharge | 75.0 | -0.234 | 0.007 | 1.42e-05 | 0.00002 | 0.00001 |
| final | val | ESS17 | discharge | 150.0 | -0.233 | 0.008 | 2.24e-05 | -0.00000 | -0.00001 |
| final | val | ESS17 | discharge | 225.0 | -0.234 | 0.012 | -1.59e-05 | -0.00000 | -0.00001 |
| final | val | ESS17 | discharge | 300.0 | -0.235 | 0.016 | -1.61e-05 | -0.00000 | -0.00001 |
| best | val | ESS14 | charge | 0.0 | -0.687 | 0.010 | 6.20e-04 | 0.00031 | 0.00077 |
| best | val | ESS14 | charge | 0.0 | -0.687 | 0.010 | 6.20e-04 | 0.00031 | 0.00077 |
| best | val | ESS14 | charge | 37.5 | -0.689 | 0.049 | -1.27e-04 | 0.00021 | 0.00052 |
| best | val | ESS14 | charge | 75.0 | -0.660 | 0.039 | 6.71e-04 | 0.00021 | 0.00052 |
| best | val | ESS14 | charge | 112.5 | -0.677 | 0.027 | 2.91e-04 | 0.00015 | 0.00037 |
| best | val | ESS14 | charge | 150.0 | -0.674 | 0.002 | 1.38e-03 | 0.00041 | 0.00100 |
| final | val | ESS14 | charge | 0.0 | -0.202 | 0.004 | 2.10e-05 | 0.00020 | 0.00038 |
| final | val | ESS14 | charge | 0.0 | -0.202 | 0.004 | 2.10e-05 | 0.00020 | 0.00038 |
| final | val | ESS14 | charge | 37.5 | -0.201 | 0.004 | 2.85e-05 | 0.00020 | 0.00038 |
| final | val | ESS14 | charge | 75.0 | -0.200 | 0.004 | 2.33e-05 | 0.00021 | 0.00038 |
| final | val | ESS14 | charge | 112.5 | -0.199 | 0.004 | 1.89e-05 | 0.00021 | 0.00039 |
| final | val | ESS14 | charge | 150.0 | -0.199 | 0.003 | 1.15e-05 | 0.00021 | 0.00039 |
| final | val | ESS14 | discharge | 0.0 | -0.202 | 0.004 | 2.13e-05 | 0.00020 | 0.00038 |
| final | val | ESS14 | discharge | 0.0 | -0.202 | 0.004 | 2.13e-05 | 0.00020 | 0.00038 |
| final | val | ESS14 | discharge | 75.0 | -0.200 | 0.003 | 3.53e-05 | 0.00002 | 0.00018 |
| final | val | ESS14 | discharge | 150.0 | -0.199 | 0.001 | 1.69e-05 | -0.00000 | 0.00016 |
| final | val | ESS14 | discharge | 225.0 | -0.197 | 0.001 | 1.54e-05 | -0.00001 | 0.00015 |
| final | val | ESS14 | discharge | 300.0 | -0.196 | 0.002 | 1.96e-05 | -0.00001 | 0.00015 |

## Findings

- **ESS17 discharge at 1.19 CNY/MWh is not the fleet-loss source.** ESS17 is positive on the key validation days and in the final round mean. At the audited state, the target payoff is highest at the near-zero quote; raising the discharge quote reduces the accepted energy and the payoff. The 264 lower-bound saturations are a low-spread, near-idle behavior, not a proven loss. The critic Q is nearly flat there; it does not strongly reward the high quote.
- **ESS14 and ESS27 carry the tail losses.** On 2022-12-30 the final deltas are ESS14 −179.2 and ESS27 −239.1; on 2022-12-31 they are −41.4 and −205.0. The final discharge quotes are high (ESS14 mean 194.2, ESS27 mean 222.7) and the charge gates are partial or closed. The critic grid at the audited state is nearly flat in the quote and, where it ranks, slightly prefers higher discharge quotes while the actual settlement is lower. The critic does not strongly penalize the high-quote drift.
- **ESS14 charge gate 0.39:** the deterministic gate is open on only 39% of validation decisions, with a 48.9 mean charge quote. The same actor's discharge quote swings from 0.49 to 299.9. This is a mixed/partial gate, not a permanently closed one; it contributes to the ESS14 loss through the discharge branch rather than the charge gate alone.
- **Acceptance:** the saved validation audit records aggregate charge/discharge/cycles, not per-block acceptance. The fixed-state clearing grid shows the ESS17 quote is accepted only at near-zero energy at q=0 and loses the target payoff as q rises; the ESS14 discharge payoff is higher at low q. A full per-block acceptance table would require a new diagnostic clear, which this phase does not run.

The row-level grid is `critic_gate_action_grid.csv`.
