# D + A minimal repair spec (Phase 16)

Specification only. Nothing in this file is implemented in the current
trainer. The existing B/C2 reward, gamma, fixed scale, terminal SOC rule and
clearing engine stay exactly as they are.

## Scope

The repair has two parts:

- **D** — make the critic able to rank the long-horizon quote value in the
  material states, and make the continuous quote branch usable in the
  high-quote region.
- **A** — keep the gate's ST path alive during the first rounds without
  changing the economic objective.

The smallest version that addresses both confirmed causes:

| id | change | type | touches |
|---|---|---|---|
| D1 | linear-core quote map (candidate C) | action parameterization | actor output semantics, new checkpoint version |
| D2 | stratified critic sampling by price/quote strata | data selection | replay sampler only |
| D3 | controlled discharge-quote exploration in high-price training blocks | exploration | training action only |
| D4 | separate model-generated counterfactual calibration set (conditional) | auxiliary supervision | separate dataset + small critic loss, only if D2/D3 leave the material states flat |
| A1 | gate-only temperature warmup τ=4 → 1 over the first two rounds | exploration | gate probability during training |
| A2 | default gate init at p=0.5; per-leg Bernoulli; ST unchanged | no change | none |

## D1 — quote map

Use candidate C from `continuous_quote_redesign_comparison.md`:

```
g(z) = 0.5 + 0.196 z                               |z| <= 2.5
g(z) = 1 - 0.01 / (1 + 9.8 (z - 2.5))^2             z > 2.5
g(z) = 0.01 / (1 + 9.8 (-z - 2.5))^2                z < -2.5
q = 0 if gate closed, else h * g(z)
```

Properties: monotone, bounded in (0,h), no clipping, gate-closed exact zero,
9.66–58.8 CNY/logit in the 0.86–0.997 region, 2.57 logits from the midpoint to
299. The map changes the price-logit meaning, so:

- the existing R2/R10 actors are **not** valid initialisations;
- the checkpoint schema version must be incremented, with the map name and its
  constants stored in the manifest;
- the new pilot starts from a fresh actor and writes a new run directory.

Do not hot-swap the map into an existing checkpoint or re-label an old
checkpoint as the new branch.

## D2 — stratified critic sampling

Replace the uniform 128-sample batch with:

- 50% uniform over the replay;
- 50% stratified over two strata: block LMP quintile (or a fixed high-price
  threshold) and executed quote deviation quintile. Sample cells uniformly
  within the stratum and then sample a transition uniformly inside the cell.

The critic loss, Bellman target, twins and `min(Q1,Q2)` are unchanged. This
only changes which transitions the critic sees. It must be implemented as a
sampler flag, with the sampler configuration stored in the run manifest.

## D3 — controlled quote exploration

For training days only, with probability `epsilon` on the discharge leg and
only when the block's price is in the upper quintile of the training-day
prices:

- replace the actor's executed discharge quote with a draw from
  `{0, 50, 100, 150, 200, 225, 258, 280, 300}`;
- keep the gate draw as the existing Bernoulli draw (a sampled closed gate
  still executes 0);
- keep the other ESS and the other leg unchanged;
- log `(day, block, ESS, actor quote, explored quote, gate, accepted MWh,
  raw profit)` for every explored transition.

Start with `epsilon = 0.10`. Do not explore on validation, H2 or the legacy
test. This is exploration, not a change to B or C2.

## D4 — conditional auxiliary calibration set

Only if D2/D3 leave the material groups anti-ranked. Generate counterfactual
rollouts from **training states** with the same frozen policy, gamma, B/C2 and
terminal rule. Store them separately with the label
`model_generated_counterfactual` and the fields listed in
`critic_calibration_improvement_design.md`. Use a small auxiliary critic loss
and report the critic's error on the real replay separately from its error on
the generated set. Do not describe the generated set as independent market
data, and do not add it to the real replay.

## A1 — gate temperature warmup

During training, use

```
tau_g(t) = 1 + 3 * max(0, 1 - t / N),  N = 106 episodes (two rounds)
```

for the gate probability only. Evaluation, target actors and the deterministic
policy use `tau_g = 1`. The Bernoulli draw and ST backward are unchanged. This
raises the minimum p(1−p) on the round-10 probabilities from 0.003 to 0.155 at
τ=4 and to 0.050 at τ=2; decay to 1 by the end so the evaluated policy is not
tempered.

If a gate remains open on <1% of decisions over an epoch, allow a temporary
fair-coin draw for that leg and log it. This is a fallback, not the default.

## Invariants that must not change

- B is the whole-day differential, paid at block 23; C2 is the same
  time-decaying SOC potential. No change to either identity.
- `gamma = 0.99`, `fixed_scale = 1000`, terminal SOC = 0.5 on the day-end
  window, the cleared day is the 24-block rolling SOCP.
- The gate is per leg, the estimator is straight-through Bernoulli, truthful is
  exactly 0 when the gate is closed.
- H2 is not read and LEGACY_SEEN_TEST is not run in this phase or in the
  pilot.
- No existing checkpoint is overwritten.

## Acceptance tests

1. **Unit:** the quote map is monotone, finite, in [0,h], has `g(0)=0.5`, no
   clipping, and returns exactly 0 when the gate is closed; the temperature
   schedule starts at 4, ends at 1 and never leaves (1,4].
2. **Sampler:** the stratified sampler returns the requested batch size, never
   samples outside the replay, and the uniform half remains uniform.
3. **One-day integration:** 24/24 blocks clear, zero fallbacks, zero voltage
   and line violations, terminal SOC 0.5, B paid only at block 23, C2
   telescope error < 1e-12.
4. **Economic:** on the winter days, the 15-minute audit must show q=0/100 at
   least matching the same-state truthful payoff and q=258–300 at zero
   acceptance where the quote is the cause; the new actor must not sit at
   q≥0.86h with zero acceptance in a high-price block.
5. **Critic:** the material-group Spearman on training-only held-out days must
   be non-negative and the twin ordering disagreement must fall; the pooled
   correlation is reported but is not the acceptance criterion.
6. **Gate:** p(1−p) minimum and the sampled joint modes on the fixed reference
   states; evaluation at τ=1; no gate permanent by accident.

## Rollback

Each change is a separate commit-sized unit behind its own flag or module. A
new run directory and a new checkpoint version are mandatory. If a change
fails its unit or one-day test, revert it without touching R2/R10. The old
checkpoints and their data are read-only references.

## Answer to the phase question

**Can D + A be implemented without changing the B/C2 economic objective?**
Yes. D1 changes the action map (not the reward), D2 changes sampling (not the
loss), D3 changes exploration (not the realised reward definition), D4 is an
optional separate auxiliary supervision, and A1 changes only the training gate
probability with evaluation at τ=1. None of them changes B, C2, gamma, the
fixed scale, terminal SOC or the clearing engine.
