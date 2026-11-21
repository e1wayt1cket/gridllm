# Reward redesign options (V2-CY-JN-2025)

**Theory only.** Nothing here is implemented, and the current reward, actor,
critic, penalty and action mapping are unchanged. Any change needs a formal
definition and a code audit first.

## The problem the diagnostics found

The current reward is a **joint-truthful, state-matched, block-local**
differential: it differences one committed block against a re-clear in which
every RL storage unit is truthful and the policy's current SOC is inherited by
both arms. That is internally consistent (the state-matching gate passes) but
it is not the quantity the paper reports. The paper reports
`pi_ai - pi_baseline` over a whole day, from the same initial SOC, with
terminal SOC neutralised by the settlement. A policy can therefore earn a
positive cumulative local reward while settling below B1, which the pilot did.

## A. Keep the current block-local differential

- **Consistency with the day objective:** low. It omits the future value of
  the SOC it consumes and the rolling-vs-whole-day mechanism difference.
- **Extra full-day counterfactual:** none.
- **Cost:** one window re-clear per block, already paid.
- **Credit assignment:** immediate; no cross-block credit.
- **Non-stationarity:** the baseline moves with the other agents' concurrent
  actions, so the target is non-stationary by construction.
- **Critic:** learns a bounded local advantage; no scale blow-up.
- **Paper terminology:** must be called a block-local, state-matched,
  joint-truthful differential, never the day-level benefit.

## B. Episode-level truthful differential return

`reward = settled(policy day) - B1(day)`, both from the same initial SOC and
the same exogenous day, with terminal SOC neutralised by the settlement.

- **Consistency with the day objective:** exact — it *is* the reported
  quantity.
- **Extra full-day counterfactual:** one per day; B1 can be precomputed once
  per day and reused, so the marginal cost is one whole-day settlement per
  episode.
- **Cost:** 26× fewer solves than A for the counterfactual (one B1/day vs 24),
  plus one policy settlement/day. Cheaper than A, not more expensive.
- **Credit assignment:** none inside the day — a single scalar. With 24 blocks
  and 4 agents, the critic must attribute a day-level number to 96 decisions.
- **Non-stationarity:** the baseline is fixed per day, so the target is
  stationary across a run.
- **Critic:** reward scale varies by day (0 to ~3400 CNY), so normalization is
  required; the signal is sparse on no-spread days.
- **Paper terminology:** this is the day-level truthful differential; it can be
  reported directly and is the only option whose training target equals the
  paper's effect.

## C. Block rewards plus potential-based shaping on SOC

Add `γΦ(s') − Φ(s)` with a state-only potential `Φ` (e.g. the continuation
value of stored energy), on top of A.

- **Consistency with the day objective:** policy-invariant in the Ng et al.
  sense if and only if `Φ` is a function of state alone and the shaping is
  added to every transition. It then does not change the optimal policy; it
  changes the learned signal's variance and the credit horizon.
- **Extra full-day counterfactual:** none; `Φ` is a learned or analytic value.
- **Cost:** one more value estimate per state, or reuse the critic.
- **Credit assignment:** propagates SOC value backwards across blocks —
  exactly the term A lacks.
- **Non-stationarity:** must not depend on other agents' actions, or
  invariance is lost; use the unit's own SOC (and perhaps time) only.
- **Critic:** risks double counting if `Φ` is also the critic; it should be a
  separate function or a fixed analytic value.
- **Paper terminology:** shaping is a training device; the reported benefit
  must remain the day-level differential, with shaping removed from the metric.

## D. State-action value difference `Q(s, a_policy) - Q(s, a_truthful)`

Use the centralized critic to compare the long-run value of the declared
action with truthful at the same state.

- **Consistency with the day objective:** closer than A (bootstrapped future),
  but only as accurate as the critic. A biased Q produces a biased advantage
  with no clear correction.
- **Extra full-day counterfactual:** none; two critic evaluations.
- **Cost:** cheap, but the target depends on the model being learned.
- **Credit assignment:** bootstrapped across blocks.
- **Non-stationarity:** the centralized critic conditions on other agents'
  states and actions, so the advantage is local to a joint state and inherits
  multi-agent non-stationarity.
- **Critic:** the reward becomes a function of the critic, making the critic's
  own target move with it — the classic moving-target risk. Needs a target
  network and a freeze schedule.
- **Paper terminology:** an advantage estimate over a truthful action, not a
  realized benefit; cannot be reported as `delta_pi` without a separate
  settlement.

## Comparison

| option | matches day payoff | extra counterfactual | cost | credit | non-stationarity | critic coupling | reportable as benefit |
|---|---|---|---|---|---|---|---|
| A current | weakly | none | 1 re-clear/block (paid) | immediate only | high | low | no |
| B episode differential | exactly | one B1/day | one settlement/day | none in-day | low | low | **yes** |
| C A + potential | invariant in principle | none | small | across blocks | low if state-only | medium | no (shaping removed) |
| D Q advantage | as good as Q | none | small | bootstrapped | medium | **high** | no |

## Recommendation (not implemented)

- Keep **B** as the *evaluation and reporting* quantity; it already exists as
  the day-level settlement against B1 and needs no new mechanism.
- For learning, **B alone** is the cleanest target but has no in-day credit;
  **A** has credit but optimizes a different objective. If the learning signal
  is to change, **C with a state-only SOC potential** is the option whose
  invariance is provable, and it can be layered on A without touching the
  clearing or the evaluation.
- **D** should not be used until the critic is accurate and checkpointed; the
  pilot could not even inspect the trained critic because it was not saved.
- Before any change: write the potential or advantage definition, prove the
  invariance (C) or state the bias (D), audit that the reported benefit is
  computed from settlement and not from the reward, and add a unit test that a
  truthful policy earns zero shaping on a closed SOC loop.
