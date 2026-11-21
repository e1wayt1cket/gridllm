# Full-day counterfactual quote interventions (P0-1)

Read-only. Four checkpoints, eight winter states, one changed quote per
rollout. Data: `daily_counterfactual_action_validation.csv/json`.

## Method

- Checkpoints: D+A best `step_00001272`, D+A final `step_00003816`, old R2
  `step_00002544`, old R10 `step_00012720`.
- States: 12-13 b3 ESS14, b12 ESS27; 12-30 b10 ESS14/27, b11 ESS14/27;
  12-31 b17 ESS14/27.
- At each state the joint snapshot is fixed. Only the target ESS's current
  discharge quote changes. The other ESS keep their current-block actions.
- From the next block the frozen deterministic actor continues; the
  counterfactual re-initialises from the state's SOC and does not transplant
  the original future SOC. Same day, B1, physical constraints and terminal
  SOC 0.5.
- Candidates: current actor quote, critic-recommended quote, truthful 0, low
  (25% of high), mid (50%), high (100%). The critic-recommended quote is the
  argmax of `min(Q1,Q2)` among the audited candidates; the fine-grid argmax is
  stored separately.
- `Q` is in reward units; `Q×1000` is the CNY-equivalent. The rollout return
  uses the same block, gamma=0.99 and C2; `b_return_scaled` is the B-only
  component and `training_return_scaled` is B+C2. The economic column is the
  full-day settlement ΔG in CNY.

## Aggregate

| checkpoint | states | Q-top = return-top | mean Spearman(Q, ΔG) | mean best gain vs actor |
|---|---:|---:|---:|---:|
| old R2 | 8 | 1 | −0.064 | +3.5 CNY |
| old R10 | 8 | 0 | −0.645 | +87.1 CNY |
| **D+A best** | **8** | **3** | **+0.288** | **+38.5 CNY** |
| D+A final | 8 | 0 | −0.171 | +65.5 CNY |

The D+A best critic is directionally better than R10 and R2, but it is still
wrong in five of eight states. The final checkpoint loses the direction.

## D+A best, per state

| day | block | unit | actor | critic rec | best return quote | actor ΔG | best ΔG | gain | rho | Q-top correct | SOC start |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 12-13 | 3 | ESS14 | 123.6 | 300.0 | 0.0 | −14.6 | −10.5 | +4.1 | −0.70 | no | 0.500 |
| 12-13 | 12 | ESS27 | 81.2 | 0.0 | 0.0 | −48.1 | −13.8 | +34.2 | +0.83 | yes | 0.673 |
| 12-30 | 10 | ESS14 | 136.8 | 0.0 | 0.0 | −163.1 | −24.2 | +138.9 | +0.66 | yes | 0.900 |
| 12-30 | 10 | ESS27 | 83.9 | 5.0 | 83.9 | +3.7 | +3.7 | 0.0 | +0.09 | no | 0.900 |
| 12-30 | 11 | ESS14 | 136.2 | 300.0 | 0.0 | −163.1 | −142.0 | +21.1 | −0.54 | no | 0.900 |
| 12-30 | 11 | ESS27 | 84.3 | 0.0 | 0.0 | +3.7 | +4.4 | +0.7 | +0.83 | yes | 0.899 |
| 12-31 | 17 | ESS14 | 135.9 | 150.0 | 0.0 | −116.1 | −7.4 | +108.7 | +0.37 | no | 0.440 |
| 12-31 | 17 | ESS27 | 83.5 | 5.0 | 75.0 | −9.6 | −9.6 | 0.0 | +0.77 | no | 0.880 |

## What the counterfactual establishes

1. **Lower quotes do recover material money in several states.** At 12-30 b10
   ESS14 the actor's 136.8 loses 138.9 CNY against q=0; at 12-31 b17 ESS14 it
   loses 108.7; at 12-13 b12 ESS27 it loses 34.2. The critic recommends the
   correct low quote in two of these three (12-30 b10 and 12-13 b12).
2. **The critic is not reliably correct.** It recommends 300 at 12-13 b3 and
   12-30 b11 ESS14, and 150 at 12-31 b17 ESS14, while the best return quote is
   0. Those are material states (21–109 CNY at b11/b17).
3. **The actor is not following the critic even when the critic is right.** At
   12-30 b10 ESS14 the critic's candidate Q-top is 0 and the return gain is
   138.9, but the actor is at 136.8. At 12-31 b17 ESS14 the actor is at 135.9
   while the best is 0. The quote is mid, not 258–300.
4. **Some actor states are already near the best and the surface is flat.**
   At 12-30 b10/b11 ESS27 and 12-31 b17 ESS27 the actor is at the best or the
   gain is <1 CNY; the critic's low recommendation does not change the return.
   A flat local surface is not evidence of a critic defect.
5. **The fine-grid critic argmax and the candidate Q-top can disagree.** The
   Q surface is nearly flat/noisy in some states; the fine-grid argmax is
   stored as `fine_grid_top_quote`, while the reported `critic_recommended`
   is the candidate Q-top. This is a symptom of the small Q separation, not a
   different state.

## Answer to the matched-return question

**Does the critic recommend a quote with a higher remaining full-day return?**
Sometimes. The D+A best critic's Q-top matches the return-top in 3/8 states and
has a positive mean rank correlation, but in five states it recommends a quote
that is not the best; in three of those the gap is material (12-13 b3,
12-30 b11 ESS14, 12-31 b17 ESS14). The critic is better than the old R10
critic (0/8, mean −0.645) but is not a reliable action selector.

## Boundary

The rollouts use the same state, frozen actor, gamma, B/C2, scale and terminal
rule. They are not a four-agent dynamic equilibrium. The all-day fixed-quote
grid in P0-2 is a different policy family and is not correlated with these
single-block Q values.
