# Difference-reward state-matching audit (V2-CY-JN-2025)

## Verdict

**The truthful baseline is a state-matched counterfactual clear of the same window**, not the day-level B1 result. It is taken from one re-clear per block with the current SOC, the same wholesale window, the same config, and every non-RL action unchanged; the RL agents' committed four periods are set to their truthful point. The results below come from one instrumented block on a real pair.

| check | result |
|---|---|
| two clears per block | PASS |
| same wholesale | PASS |
| same config object | PASS |
| same horizon type | PASS |
| same agent population | PASS |
| baseline committed block is truthful | PASS |
| baseline later periods keep policy actions | PASS |
| non rl actions unchanged | PASS |
| baseline starts from current soc | PASS |

## Why the baseline is truthful for the committed block only

The reward differences the *decision* over the four committed periods, so the counterfactual restores truthfulness on exactly those periods and leaves the rest of the 16-period window at the policy's own declared actions. Replacing the whole window would confound the committed action with a different future plan, and would no longer answer 'what did this decision add'. The baseline agents still start from the current SOC, so an earlier policy move is inherited by both arms.

## The seven required checks

1. **Full-day B1 used as the baseline?** No. The baseline is a window clear inside `step`; `rl_env` never calls the day-level settlement for reward shaping.
2. **Stale SOC after the policy moved?** No. `_make_window_agents` is given `self.prev_soc`, and the audit reads the baseline agents' `soc0` after a block and finds it equal to `prev_soc`.
3. **One counterfactual clear for every agent?** Yes. A single `base_result` supplies every RL agent's baseline profit.
4. **Same price, load, constraints, cycle cost?** Yes: identical `wholesale` array, identical config object and horizon type; load and network come from the same window agents.
5. **Sign.** `reward = raw_profit - local_baseline_profit`; reported separately in `info`, so the sign can be audited per block.
6. **Penalty placement.** The bounded deviation penalty is added to the actor loss only (`rl_bidding.update`). It is not in the reward. The market-impact penalty is a separate optional reward term, zero in this pilot.
7. **Baseline visible to the actor?** No. The observation builder does not reference `_last_base_result` (not found).

## Relation to B1

B1 is a whole-day truthful clearing and is the *evaluation* benchmark. The training counterfactual is a block-level, state-matched clear. They are consistent in their definition of truthful quotes, but they are different quantities: one is an evaluation outcome, the other a learning signal. The pilot reports them separately and never substitutes one for the other.
