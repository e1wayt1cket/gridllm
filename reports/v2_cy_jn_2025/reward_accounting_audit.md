# Reward and cost accounting audit (V2-CY-JN-2025)

**Verdict: consistent. The clearing objective and the settlement ledger carry the cycle cost in different quantities, and the settlement charges it once.**

## One cycle cost, two places, no double charge

| place | form | what it does |
|---|---|---|
| clearing objective | `-100 * (ch + dis) * dt` | prices throughput so the market's dispatch reflects degradation; it is part of the optimization, not a participant's cash |
| `participant_payoff` | `degradation_cost = 100 * (ch + dis) * dt` | the only place the participant's cash is charged |

A synthetic leg (0.1 MW charge and 0.1 MW discharge for 96 periods at 400 CNY/MWh) settles to -480.0000 CNY; the hand computation `LMP*(sell-buy)*dt - 100*(ch+dis)*dt` gives -480.0000 CNY. Difference 0.00e+00. The degradation term is 480.00 CNY, charged once.

The clearing objective is **not** a payoff and is never subtracted again: `participant_payoff` builds the cash figure itself, and the env's raw profit is that function's total (confirmed).

## Reward decomposition

Per block, per RL agent:

    raw_profit          = participant_payoff(committed periods)
    local_baseline      = the same quantity in the truthful-arm re-clear
    reward              = raw_profit - local_baseline
                          - market_impact_penalty * power   (default 0)

The bounded deviation penalty is **not** in the reward. It is added inside `rl_bidding.update` to the actor loss (present), defined once in `rl_td3.deviation_penalty`, and the env does not reference it (clean). So it cannot be double charged: it shapes the policy gradient and never reduces the logged reward.

## What the training log must keep

`env.step` returns `info` with `reward`, `raw_profit` and `local_baseline_profit` per agent (confirmed). The pilot log writes all three plus the episode sums, so the training signal is never the only record of what happened. The deviation-penalty magnitude is logged from the actor update, apart from the reward.

## Not claimed

A higher raw profit or a higher reward is not social welfare. The clearing objective is not welfare in this project, and reward is not benefit.
