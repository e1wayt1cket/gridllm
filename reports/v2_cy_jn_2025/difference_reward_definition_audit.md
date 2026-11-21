# Difference-reward definition audit (V2-CY-JN-2025)

## Which benchmark the reward uses

**Joint truthful bidding by all four RL storage agents, inside the same window and from the same current state.** It is *not* a unilateral counterfactual in which one agent is truthful and the others keep their learned actions: `rl_env.step` iterates over `self.rl_agents` and sets each one's committed four periods to its own truthful point before the single baseline clear. The four recovery actions come from one clear, not four.

| property | result |
|---|---|
| two clears per block | PASS |
| same wholesale window | PASS |
| same config object | PASS |
| same horizon type (window) | PASS |
| every RL agent truthful on the committed block | PASS |
| non-RL actions unchanged | PASS |
| baseline starts from the current SOC | PASS |

The non-RL agents' actions are unchanged because they have none in this experiment: their declared quotes are their truthful defaults, so holding them fixed is also holding them truthful. That is a property of this pilot, not of the mechanism in general.

## Terminology

Call it a **joint-truthful, state-matched, block-local differential reward**. Canonical terms it is *not*:

- not the classic unilateral difference reward (that changes one agent's action only);
- not the day-level truthful difference (`pi_ai - pi_baseline`), which requires whole-day clears from the same initial SOC;
- not a potential-based shaping term, because it has no potential function and can be non-zero on a closed loop.

The reward is local in two senses: it differences the committed four periods, and the counterfactual re-clears the window with the policy's own later-period actions still in place. Both are deliberate; see `difference_reward_state_matching_audit.md`.

## Per-agent accounting

Each agent's baseline profit is read from the same `base_result`, at that agent's own bus LMP, through `participant_payoff` -- the same ledger used for the real arm. No agent's baseline is taken from another agent's schedule or from the day-level B1 result.
