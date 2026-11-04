# MATD3 action-mapping diagnostic (V2-CY-JN-2025)

## The mapping

`rl_td3.Actor.forward_logits` computes `a = mid + tanh(logit) * half`, i.e. tanh = 0 maps to the **centre** of the box. For a storage agent the boxes are charge 0-150 and discharge 0-300 CNY/MWh, so the centre is (75, 150) and the truthful point (0, 0) is the **lower corner**, reached only as tanh → −1. The mapping is monotone and correct; the truthful point is simply at a saturated edge, not at the centre.

That parameterisation predicts the pilot's first actions: the training log's first-five-episode charge mean is 63-76 CNY/MWh, near the 75 centre, and the discharge mean is near 150. With 20 episodes the actor barely leaves that region.

## Observed quotes

| run | unit | charge mean | charge sd | discharge mean | discharge sd | normalised distance | saturated share | corr(charge, SOC) | corr(charge, last LMP) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL96/11 | ESS14 | 149.91 | 0.07 | 0.60 | 0.34 | 1.00 | 288/288 | 0.05 | 0.25 |
| FULL96/11 | ESS17 | 0.79 | 0.38 | 1.62 | 0.76 | 0.01 | 148/288 | -0.24 | -0.43 |
| FULL96/11 | ESS27 | 16.70 | 2.27 | 64.61 | 3.95 | 0.33 | 0/288 | -0.09 | -0.38 |
| FULL96/11 | ESS9 | 10.19 | 1.76 | 149.92 | 7.45 | 0.57 | 0/288 | 0.06 | -0.15 |
| FULL96/22 | ESS14 | 53.70 | 1.94 | 120.92 | 2.69 | 0.76 | 0/288 | 0.08 | -0.34 |
| FULL96/22 | ESS17 | 149.74 | 0.20 | 106.54 | 8.54 | 1.35 | 278/288 | -0.01 | -0.02 |
| FULL96/22 | ESS27 | 148.78 | 0.51 | 0.15 | 0.13 | 0.99 | 288/288 | 0.05 | 0.34 |
| FULL96/22 | ESS9 | 48.28 | 2.48 | 59.32 | 4.00 | 0.52 | 0/288 | 0.00 | -0.56 |
| NEAR/11 | ESS14 | 98.56 | 8.27 | 0.59 | 0.35 | 0.66 | 279/288 | -0.35 | -0.72 |
| NEAR/11 | ESS17 | 62.91 | 10.28 | 0.13 | 0.11 | 0.42 | 288/288 | -0.36 | -0.80 |
| NEAR/11 | ESS27 | 0.30 | 0.19 | 11.73 | 2.85 | 0.04 | 279/288 | -0.21 | -0.42 |
| NEAR/11 | ESS9 | 50.23 | 1.73 | 41.06 | 7.25 | 0.47 | 0/288 | -0.04 | -0.52 |
| NEAR/22 | ESS14 | 126.61 | 2.32 | 44.81 | 6.17 | 0.99 | 0/288 | 0.27 | 0.50 |
| NEAR/22 | ESS17 | 12.11 | 3.00 | 0.05 | 0.05 | 0.08 | 288/288 | 0.23 | 0.43 |
| NEAR/22 | ESS27 | 149.75 | 0.16 | 92.62 | 5.14 | 1.31 | 283/288 | 0.17 | 0.31 |
| NEAR/22 | ESS9 | 105.87 | 1.53 | 104.35 | 2.78 | 1.05 | 0/288 | 0.20 | 0.33 |

`normalised distance` is `sum_i |a_i - truthful_i| / (high_i - low_i)` over the two prices; 0 would be truthful and 2 the opposite corner. The actors settle into **corners**, and which corner differs by unit and seed: the distance ranges from 0.01 (ESS17 in FULL96/11, almost truthful on both legs) to 1.35, and the saturated share ranges from 0/288 to 288/288. A corner quote near the top of the charge box (ESS14 in FULL96/11, charge 149.9) is a refusal to charge, not a mid-box compromise; a near-zero charge quote with a mid-box discharge (ESS9) is one-directional operation.

## ESS14

| run | settled mean | charge MWh/day | discharge MWh/day |
|---|---:|---:|---:|
| FULL96/11 | 1.16 | 0.01 | 0.01 |
| FULL96/22 | 0.01 | 0.00 | 0.00 |
| NEAR/11 | 19.28 | 0.19 | 0.16 |
| NEAR/22 | 0.00 | 0.00 | 0.00 |

Where the settled value is 0 and both energies are 0, ESS14 is simply not dispatched: its quotes were too high for the clearing at that bus. It is not a settlement cancellation (charge and discharge both non-zero offsetting each other), and it is not a sign error.

## Why the quotes stay away from truthful

1. **Action parameterisation.** Truthful is the lower corner at tanh = −1; the actor initialises at the centre. The final policies have travelled to a corner — but often the *other* one on one leg (charge ≈ 150 means refusing to charge), because the learned signal rewards the block, not the future SOC the charge would create. Reaching truthful is a saturated trip; reaching a refusal is just as saturated and the block reward can prefer it.
2. **Exploration and buffer.** Warmup actions are uniform over the box, whose mean is the centre, so the replay buffer is dominated by mid-box quotes before the actor has any signal.
3. **Training length and credit.** Twenty episodes, 417 updates. The policy has travelled to corners, so the budget is enough to saturate; what it lacks is a signal about the future value of the SOC those corners create.
4. **Critic Q bias: not measured.** The critics were not saved with the actors, so the learned Q surface cannot be inspected. This is a reproducibility gap, not evidence that the critic is wrong.

What it is **not**: a sign or scale bug. The truthful point maps to exactly (0, 0) at tanh = −1 (checked numerically in the tests), the action is within bounds every block, and the clearing receives the same numbers the actor produced.
