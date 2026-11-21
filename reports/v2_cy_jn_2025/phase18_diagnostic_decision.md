# Phase-18 diagnostic decision

Read-only decision from P0-1 and P0-2. No training, no checkpoint change, no
H2 or legacy test.

## What the ground truth says

1. **The D+A critic is directionally better but not reliable.** Matched
   same-state Q vs B+C2 correlates +0.609 pooled, and the Q-top matches the
   return-top in 3/8 states; it is wrong in five states, three of them
   material. The old R10 is negative on all counts; old R2 is near zero.
2. **The actor does not follow the critic even when the critic is right.**
   At 12-30 b10 ESS14 the critic's Q-top is 0, the actor is at 136.8, and the
   matched full-day gain is 138.9 CNY. At 12-13 b12 ESS27 the corresponding
   gain is 34.2 CNY.
3. **The single-ESS problem is well posed.** With the other three truthful,
   ESS14's all-day fixed-quote optimum is a broad low-discharge basin around
   charge 37.5–50 and discharge 33–47, +9.3 CNY over truthful. High discharge
   ≥166.7 is a loss region. The surface has several local plateaus but a
   clear useful basin.
4. **The actor is far from that basin on ESS14 but near it on ESS27.** The
   D+A best actor's ESS14 mean is charge 20.8 / discharge 110.2; the nearest
   grid point is 10.3 CNY below the conditional best, and its own-policy
   payoff is −95.4 CNY. For ESS27 the same actor's mean (37.9, 77.4) is 1.75
   CNY from the grid best and its own-policy payoff is 110.33. The defect is
   unit-specific.
5. **The gate is not the sole explanation.** The deterministic gate is one
   mode and permanently open, but the critic/actor mismatch is already present
   at quote values 81–137 and in the single-ESS grid, where no gate policy is
   involved.

## Branch

**Do both, in this order: critic calibration first, then actor/credit
assignment.** This is not a "critic only" or "actor only" result.

### P1-A: Critic calibration

Target the three material states where the D+A critic's Q-top is wrong:

- 12-13 b3 ESS14: critic 300, best 0, gain 4.1.
- 12-30 b11 ESS14: critic 300, best 0, gain 21.1.
- 12-31 b17 ESS14: critic 150, best 0, gain 108.7.

Also check the flat states where the critic recommends low but the actor is
already at the best (12-30 b10/b11 ESS27, 12-31 b17 ESS27): those are not
critic failures.

The calibration work should use the P0-1 matched rollouts as the label on
training-only states; the validation states are for reporting, not training
replay. The replay coverage/target questions from phase 16 remain relevant.

### P1-B: Actor gradient chain

On the states where the critic is correct but the actor does not move:

- 12-30 b10 ESS14: critic Q-top 0, actor 136.8, gain 138.9.
- 12-13 b12 ESS27: critic Q-top 0, actor 81.2, gain 34.2.

Check the actor loss gradient through the v3 quote map, the target/critic
gradient, the optimizer step, the policy delay, and the exploration-induced
off-policy mismatch. The actor networks are **per-agent**, not shared, so a
"shared actor parameter conflict" check is not applicable in this code; the
conflict to test is credit assignment among the centralized critics and the
fleet's joint action.

### P1-C: Stochastic/deterministic gate

The deterministic gate is one mode; the stochastic gate visits 243 modes. A
gate ablation (force open/closed, no training) can isolate whether the actor's
quote choice survives without the gate. Do this only after P1-A/P1-B; it is
not the first explanation.

## What not to do

- Do not blame the gate alone; the single-ESS grid shows a quote optimum with
  no gate policy.
- Do not run a larger training budget: the pilot's stop condition 3 was met.
- Do not use the all-day fixed-quote grid as a Q target for single-block
  states; they are different policy families.
- Do not add validation to the replay or read H2/legacy test.

## Answer

The evidence supports a **critic-plus-actor** branch. The simplified one-ESS
problem has a clear, interpretable optimum, so the four-agent failure is not
just an ill-posed objective; but the critic is still wrong in 5/8 matched
states, so an actor-only diagnosis would be premature. ESS27 is near its
conditional optimum while ESS14 is far, so the actor's failure is not a
uniform inability to quote well; it is specific to the ESS14 state/credit
path. The first concrete
checks are the three material critic states and the two "critic right, actor
wrong" states, followed by the gate ablation only if those do not explain the
gap.

## Boundary

No training and no algorithm change in this phase. The P0-1 rollouts are
same-state counterfactuals; the P0-2 grid is a conditional fixed-quote family.
