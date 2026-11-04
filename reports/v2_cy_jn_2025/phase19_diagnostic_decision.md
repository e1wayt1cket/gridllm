# Phase 19 diagnostic decision

Read-only diagnosis of the D+A MATD3 pilot. No training, no algorithm change,
no gate ablation, no H2, no legacy test, no checkpoint write. Evidence:
`phase19_critic_calibration.md`, `phase19_actor_gradient_chain.md`,
`phase19_cross_ess_gradient_conflict.md`, and their CSV/JSON files.

## Answers to the four questions

### 1. Does the critic have a reproducible systematic ranking problem?

**Yes, on both validation and training days.**

* Training-day states (24 state-leg groups, three NEAR training days × two
  blocks × two units × two legs): Q-top equals the return-top in **4/24** for
  D+A best and **4/24** for D+A final; mean within-state Spearman is −0.081 and
  −0.038.
* Against the economic quantity (full-day ΔG), the pooled correlation is
  +0.001 (best) and −0.253 (final), i.e. zero or negative.
* The local derivative is not a usable descent direction: `dQ/dq` points
  toward the best-return quote only 58.3% (best) / 48.1% (final) of the time.
* The twin gap is 16–27 CNY on average (p95 57–66 CNY), the same order as or
  larger than the material gains being ranked (11–16 CNY on training days,
  21–139 CNY on the validation cases).
* The failure is not an extrapolation artifact: 87% (best) and 98% (final) of
  the calibrated quote grid is conditionally covered by the replay, and the
  high-quote region is the most frequent ESS14 discharge region in the final
  buffer (median 133.4 CNY/MWh).

Within the phase-19 evidence, the critic's problem is an estimation/ranking
problem on covered states, not a missing action region and not a unit/leg
artifact (the correlation is near zero in every unit, leg and quote band).

### 2. Does the actor have a gradient transfer or update problem?

**The tensor path is sound; the objective path is not.**

* Connected: every price logit has a nonzero gradient; the gate logits receive
  the straight-through surrogate; no NaN/Inf.
* No clipping: total actor gradient norms are 0.006–0.21 against a clip
  threshold of 1.0, so the clip coefficient is 1.0. The optimizer is active and
  carries nonzero Adam moments at both checkpoints. Policy delay and the
  counter are not blocking the replay.
* **Implementation error:** the actor loss is differentiated through
  `truthful_gate_actor_adapter.forward_policy` (v2 sigmoid), while execution
  and the target actor use `da_quote_map.forward_policy_v3`
  (`linear_core_v1`). The critic is trained on the v3 executed quote. The actor
  gradient therefore is not the gradient of the action it executes. The
  direction is close at the two named ESS14 cases (cos 0.986) but can reverse
  at ESS27 states (cos −0.892 best, −0.973 final), and the norm ratio is
  0.29–1.48.
* **Geometry:** at 12-30 b10 ESS14 the critic's local slope at the actor's
  136.8 quote is positive (it points upward) while its own Q-top is 0 and the
  matched gain from 0 is 138.9 CNY. A first-order actor step cannot cross that
  non-concave gap. The virtual one-step update moves the executed discharge
  quote by less than 1 CNY/MWh even on the correct map.

The actor is not detached and not frozen; it is being updated toward a local
direction that does not reach the better quote, and in the current code that
direction is computed on the wrong map.

### 3. Is there a significant cross-ESS gradient conflict?

**No shared-parameter conflict exists in this code.** Actors are per agent;
across the four actors there are zero shared parameter tensors. The within-actor
shared trunk gives charge/discharge cosines of +0.098 (best) and 0.000 (final;
the zero is a closed-gate effect), so there is no stable leg conflict.

The real asymmetry is credit: in D+A final, ESS14's own critic gives only
0.4–2.2% of its squared action sensitivity to ESS14's own action, while ESS27's
critic gives ESS27 44–50%. That is a centralized-critic credit problem, not a
shared-parameter cancellation, and it is consistent with the unit-specific
failure (ESS14 far from optimum, ESS27 near).

### 4. What kind of problem is it?

| component | verdict | category |
|---|---|---|
| actor loss map | v2 map in the update, v3 in execution/target | **code implementation error** |
| actor step | small, local, away from the Q-top at 12-30 b10 ESS14 | consequence of critic geometry |
| critic ranking | Q-top 4/24 on covered training states; local slope inconsistent; twin gap comparable to effect | **critic estimation / training design** |
| action coverage | 87–98% conditional grid coverage | not the cause |
| cross-ESS shared parameters | none exist | not applicable |
| ESS14 credit | own-action share collapses in final | **credit assignment** |
| gate | forward, backward and closed-gate behavior as designed | not implicated |

## What Phase 20 may and may not do

Evidence supports exactly one unambiguous code correction and two open
diagnoses:

1. **Correct the actor-loss map (unambiguous).** The actor update must use the
   same v3 map as execution and the target. This is a one-branch change; it
   must be put in an independent candidate version with a test that asserts the
   update-path quote equals the executed quote for the same logits and gate
   draw.
2. **Critic ranking (open, not yet a code fix).** The evidence locates the
   failure on covered states but does not yet separate "the critic fits noise",
   "the twin-min target is unstable", and "the reward target has high
   variance". A Phase 20 discriminator must measure the per-state Bellman
   residual and target variance before changing the critic. More training data
   is not the first response: coverage is already 87–98% on the evaluated grid.
3. **ESS14 credit (open).** The own-action share is a local sensitivity on four
   states. It must reproduce across more training states before any credit
   design change. No network restructuring is justified by this sample.
4. **Do not touch the gate.** The gate is not the explanation at the named
   states; the phase-18 gate ablation remains out of scope.

If the Phase 20 discriminator finds no reproducible critic defect beyond the
map error, the decision must be "no further correction" rather than a new
architecture.

## Boundary

All five phase-18 cases are NEAR validation days: 12-13 is FULL96 test, 12-30
and 12-31 are FULL96 train but NEAR validation. They remain offline cases and
contribute no training label. New calibration labels came from NEAR training
days only. The critic calibration rollouts are single-block counterfactuals,
not the four-agent dynamic equilibrium. The actor audit is a one-state replay,
not a training episode.
