# Critic training-target and data-distribution audit (P1, Phase 15)

Read-only. The target is inspected in `rl_b_c2_gate.MATD3BC2Gate.update`; the
replay is the final checkpoint's buffer; the branch states are the Phase-15
validation/training states. No training, test or H2.

## Target calculation

For each agent `i`:

```
target_actions = [target_quotes(actor_target_j(obs_j)) for all j]
target_q1, target_q2 = critic_target_i(all_obs, target_actions)
target = rew_i + gamma * (1 - done) * min(target_q1, target_q2)
critic_loss = MSE(q1, target) + MSE(q2, target)
```

- `target_quotes` uses a deterministic **hard gate**; the target action is the
  executed quote, not the gate logit/probability.
- `rew_i` is the stored shaped reward divided by the fixed scale 1000:
  `(B_i + C2 shaping_i) / 1000`. The B reward is paid only at block 23; the C2
  shaping is per block; the C2 boundary is zero at the terminal.
- `done` is true at block 23 and false before it. There is no mid-episode
  truncation. `gamma = 0.99` per block.
- The critic input is the concatenation of all agents' observations and all
  **executed quotes**. The gate logit is not an input.
- The target critics are in eval mode; the target actor is hard-gated.

No target-construction bug was found. In particular, the day reward at block
23 is bootstrapped through the chain: at block 22 the target contains
`gamma * Q_target(23)`, and the value propagates to earlier blocks.

## Replay consistency

| check | result |
|---|---:|
| transitions / episodes | 12,720 / 530 |
| episode-end (`done`) count | 530 |
| sampled transitions with a closed gate and nonzero executed quote | 0 / 255 |
| sampled transitions with a quote at the upper bound | 64 / 255 (25%) |
| ESS14 quote max (charge/discharge) | 149.93 / 299.97 |
| ESS27 quote max | 148.98 / 299.36 |
| mean absolute last-block reward | 0.0187 |
| mean absolute non-last reward | 0.00517 |

The gate/quote and executed-quote/replay relationships are consistent: a
closed gate always stores an executed quote of zero. The high-quote actions
(up to the upper bound) are **present in the replay**, so the critic has seen
the action region; missing action coverage is not the explanation for the flat
high-quote value.

## Critic on the training batch vs the branch states

| round/agent | critic loss (training batch) | target Q | twin gap |
|---|---:|---:|---:|
| R2 ESS14 | 0.022 | −0.665 | 0.063 |
| R2 ESS27 | 0.010 | −0.418 | 0.029 |
| R10 ESS14 | 0.001 | −0.173 | 0.006 |
| R10 ESS27 | 0.000 | −0.091 | 0.003 |

On its training replay the critic fits the target (small loss, small twin gap).
On the long-horizon **validation** states, `Q×1000` has correlation −0.59 with
the discounted training return and a mean absolute gap of 312.6 CNY-equivalent;
in the material winter states it is often anti-ranked (Spearman −0.6 to −0.94).
The critic is therefore fit on the training distribution but not calibrated on
the validation winter states.

## State-distribution caveat

The branch states are validation winter days; the replay is the training days.
The action quotes are covered, but the state distribution is not. This is a
state-coverage/generalisation issue, not a missing action or a broken Bellman
target. It also explains why a flat Q on a training batch is not evidence that
the critic is calibrated on the states where the policy is evaluated.

## Why the day reward at block 23 is not the whole explanation

The B reward is paid only at block 23, but the critic target propagates it
through `gamma * Q_target`; the Q at block `k` is the discounted future
training reward. The long-horizon calibration compares exactly that quantity.
The Q is not systematically zero or truncated; it is small and misordered in
the material states. The critic's value learning, not the terminal timing, is
the failure point.

## Required answer

The flat/anti-ranked Q is partly the market (72/88 groups have <1 CNY spread)
and partly the critic on the material validation states. The target
calculation is correct; the replay covers the high-quote actions; the state
distribution is the coverage gap. Do not conclude network capacity from the
flat Q alone; conclude that the critic is not calibrated on the evaluation
states, and that value learning/state coverage must be part of the next
change.
