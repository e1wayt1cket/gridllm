# Critic long-horizon calibration (P0)

499 counterfactual rollouts, 88 state×ESS×leg groups, 22 fixed states. Dates: the three winter validation days and two normal control days. Each rollout changes one ESS/one quote leg at a fixed block, then continues the frozen deterministic policy to the day end; the future SOC is not forced. This is an actual clearing rollout, not a single-block payoff.

The critic Q is in the training reward units (B+C2 divided by fixed_scale = 1000). The rollout reports both the discounted training return in the same units and the actual settlement ΔG in CNY. The ranking comparison uses the actual full-day ΔG.

## Aggregate ranking

| slice | mean Spearman(Q, ΔG) | median | positive groups | groups |
|---|---:|---:|---:|---:|
| all | -0.008 | 0.057 | 47/88 | 88 |
| checkpoint R2 | 0.102 | 0.086 | 28/44 | 44 |
| checkpoint R10 | -0.118 | -0.057 | 19/44 | 44 |
| ESS14.discharge | -0.046 | 0.043 | 9/18 | 18 |
| ESS27.discharge | -0.302 | -0.200 | 5/18 | 18 |
| ESS14.charge | 0.027 | 0.029 | 9/18 | 18 |
| ESS27.charge | 0.206 | 0.200 | 13/18 | 18 |
| ESS9.charge | 0.050 | 0.193 | 3/4 | 4 |
| ESS17.discharge | 0.132 | 0.150 | 3/4 | 4 |

- The overall mean is essentially zero; R10 is slightly negative (-0.118); R2 is slightly positive (0.102).
- **72 of 88** groups have a less-than-1-CNY spread across the quote grid: for these the market value is genuinely flat, so a flat Q is not by itself evidence of a critic defect.
- **10 groups** have a material (>10 CNY) spread. Their ranking is the decisive evidence; the table below lists them.

## Material states (actual ΔG spread > 10 CNY)

| ckpt | day | block | ESS | leg | Q range | ΔG range | Spearman | actor quote | best quote | worst quote | best ΔG | worst ΔG |
|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2022-12-30|2025-12-31 | 11 | ESS14 | discharge | 0.312 | 19.5 | 1.00 | 0.0 | 0.0 | 300.0 | 11.8 | -7.7 |
| 2 | 2022-12-30|2025-12-31 | 11 | ESS27 | discharge | 0.311 | 17.8 | -0.71 | 51.7 | 0.0 | 225.0 | 5.5 | -12.3 |
| 10 | 2022-12-13|2025-12-11 | 2 | ESS27 | discharge | 0.006 | 16.2 | -0.77 | 285.1 | 0.0 | 285.1 | -99.7 | -115.9 |
| 10 | 2022-12-13|2025-12-11 | 11 | ESS14 | charge | 0.002 | 36.0 | -0.60 | 0.0 | 112.5 | 0.0 | -20.8 | -56.7 |
| 10 | 2022-12-13|2025-12-11 | 11 | ESS27 | charge | 0.000 | 17.8 | -0.90 | 0.0 | 112.5 | 37.5 | -98.1 | -115.9 |
| 10 | 2022-12-13|2025-12-11 | 11 | ESS27 | discharge | 0.005 | 30.6 | -0.14 | 211.3 | 0.0 | 300.0 | -85.3 | -115.9 |
| 10 | 2022-12-13|2025-12-11 | 20 | ESS14 | discharge | 0.062 | 24.4 | 0.54 | 1.4 | 150.0 | 300.0 | -56.7 | -81.2 |
| 10 | 2022-12-13|2025-12-11 | 20 | ESS27 | discharge | 0.010 | 27.1 | 0.03 | 87.1 | 0.0 | 300.0 | -115.9 | -143.0 |
| 10 | 2022-12-30|2025-12-31 | 11 | ESS14 | discharge | 0.007 | 56.1 | -0.94 | 299.2 | 0.0 | 299.2 | -123.1 | -179.2 |
| 10 | 2022-12-30|2025-12-31 | 11 | ESS27 | discharge | 0.007 | 62.7 | -0.77 | 282.0 | 0.0 | 300.0 | -176.4 | -239.1 |

## Q vs training return (absolute scale)

- Correlation between Q×1000 and the discounted training return: -0.591.
- Mean absolute gap |Q×1000 − training return|: 312.6 CNY-equivalent. The Q is not a usable absolute predictor of the discounted return either.

## Findings for the required questions

1. **Can the critic rank long-horizon quote value?** Not reliably. The overall Spearman is ~0; R10 is negative; in the material states the critic is often anti-ranked. The R2 best checkpoint is only slightly positive.
2. **Is the flat Q the model, the data, or the market?** Both. Most states (72/88) have an economically flat market, so a flat Q is appropriate there. In the material states the critic still fails, and the replay buffer does contain high quotes (the final buffer has upper-bound quotes in 64/255 sampled transitions), so the failure is not simply missing action coverage. It is a calibration/target problem in the states that matter.
3. **Are the high quotes causes or results?** In the material states the actor's quote is frequently at the worst end (ESS14/ESS27 discharge near 299/282) while the best quote is at/near 0; the high quote is the direct cause of the lost settlement. It is also the actor's chosen action, weakly preferred by the flat/anti-ranked critic; the winter price and cycle cost make the loss material.
4. **The winter blocks:** the per-block reconstruction identifies blocks 10/14/11 on 2022-12-30, 3/17 on 2022-12-13 and 17 on 2022-12-31 for ESS14/ESS27. The high quote plus cycle cost sits at or near the nodal price in the loss blocks (above it in several); the battery idles while B1, with an effective offer of 100, captures the spread.

The state snapshots (SOC, block, gate/quote, price window, other-agent actions, data/checkpoint hashes) are in `critic_long_horizon_states_cp*.json`; the row-level grid is `critic_long_horizon_cp*.csv`.
