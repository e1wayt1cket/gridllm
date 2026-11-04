# Paper terminology and method-definition review (V2-CY-JN-2025)

Accurate vs inaccurate statements, based on the implemented code and the
sixth-phase diagnostics.

## Terms

| avoid | use | why |
|---|---|---|
| "difference reward" (unqualified) | **state-matched, joint-truthful, block-local differential reward** | all four RL units are reset to truthful in the counterfactual; it is not a unilateral difference reward |
| "unilateral difference reward" | — | that term means one agent deviates; here all four do |
| "agents maximize their daily profit" (as the training statement) | "the training signal is a local differential; the paper evaluates the daily settlement" | the critic regresses a discounted, standardized block signal |
| "reward" and "profit" interchangeably | keep **reward** (training) and **settled payoff** (evaluation) separate in every table and sentence | a reward is not a payoff |
| "B1 is the truthful counterfactual of the reward" | "B1 is the **whole-day** joint-truthful benchmark; the training counterfactual is a separate block-level clear" | different states, different horizon |
| "SOC effect from the decomposition" | "numerical accounting decomposition: local + mechanism + state-path" | the state-path term also contains the rolling-vs-full-day mechanism; it is not a pure SOC causal effect |

## Method statements

1. **The baseline.** B1 / JN-Base is a whole-day clearing in which all four
   storage units declare $(0,0)$ and all loads/prosumers declare their own
   truthful point, from $soc_0=0.5$, with terminal SOC pinned; the fleet
   payoff is the sum of each unit's `participant_payoff` at its own bus LMP.
2. **The reward.** Per block: the committed-period settled profit minus the
   same quantity in a re-clear where all four RL units are truthful, the
   current SOC is inherited by both arms, opponents and later-period policy
   actions are unchanged, and one counterfactual clear serves every agent.
   It is discounted at $\gamma=0.99$ per block and standardized by the stored
   rewards' std before the critic target.
3. **The action.** A storage unit declares two reservation prices in CNY/MWh,
   bounded at `[0,150]` and `[0,300]`; the truthful declaration is the lower
   bound. The current actor maps $\tanh(z)$ to the box centre, so truthful is
   at the saturated edge and is not exactly reachable at finite parameters.
4. **The discount.** The training return is discounted per one-hour block and
   standardized; the reported benefit is an undiscounted day settlement. These
   are different quantities and the paper must say so.
5. **The terminal condition.** Training windows do not pin terminal SOC; the
   settlement does. The training return can reward an end-of-day SOC that the
   evaluation neutralizes.
6. **The date sets.** 96 paired days (FULL96), 80 (NEAR); the comparison uses
   `common_split.json` with a common test of 11 days; the 16 FULL96-only days
   are training-only; the 5 national-day days are held out.
7. **The transaction.** Each pair is a 2022 load day with a 2025 Jiangnan
   day-ahead price day. It is a cross-year controlled simulation, not a
   same-period settlement; the price unit is an inference, not an externally
   confirmed data dictionary.
8. **The test status.** The 11-day benchmark has been viewed in the pilot. Any
   later model must either use a fresh holdout or state explicitly that this
   benchmark is no longer blind.

## Claims the current evidence supports

- The environment, clearing and training interface run stably; the reward and
  state-matching are audited.
- MATD3 underperformed B1 on every viewed test day; the extra 16 training days
  did not help at the pilot's budget.
- The local reward is negatively correlated with the day-level difference on
  the sampled training/validation days; the gap is a definition mismatch plus
  a real cross-period opportunity cost, not an implementation bug.

## Claims the evidence does not support

- That MATD3 has an economic advantage, or that a rule policy does.
- That the 16 long pairs are harmful, or that 69 days is a larger effective
  sample than 53 at a fixed budget.
- That the state-path term is a pure SOC causal effect.
- That the trained critic is accurate: it was not saved.
- That the 11-day test is a blind holdout.
- Any welfare interpretation of the clearing objective or of a reward.
