# A1 gate temperature warmup (P0)

Implemented in `DAExperimentConfig.gate_temperature_at` and
`DAGateTrainer._policy_action`. Data: `da_gate_pilot_summary.json`,
`runs/DEV_DA_GATE_seed11_v2/{episodes.csv,gate_stats.jsonl,eval_history.json}`.

## Schedule

The temperature acts on the **gate logits only**, never on the continuous
price logits:

```
tau_g(t) = 4 + (1 - 4) * min(1, t / 106)
```

| episode | temperature | note |
|---:|---:|---|
| 0 | 4.000 | start |
| 53 | 2.528 | end of round 1 |
| 106 | 1.028 | end of round 2 |
| 159 | 1.000 | end of pilot |

Evaluation, target actors and the deterministic policy always use τ=1. The
estimator remains hard-forward / straight-through Bernoulli; each leg is
independent; no entropy, REINFORCE or Gumbel was added.

## Observed gate behaviour

| round | validation mean ΔG | deterministic modes | sampled modes in training | permanent open / closed |
|---:|---:|---:|---:|---|
| 1 | −32.1 | 1 | — | 7 / 1 |
| 2 | −44.3 | 1 | — | 6 / 2 |
| 3 | −79.3 | 1 | 243 (last episode) | 5 / 3 |

The selected checkpoint is round 1 (best validation); its deterministic
policy uses one joint mode on all 11 validation days. The last training
episode's stochastic gate visited 243 of 256 modes, so the Bernoulli gate is
not frozen; the **deterministic** threshold is.

Per-leg probabilities and p(1−p) at validation:

| round | ESS14 discharge prob | p(1−p) | ESS27 discharge prob | p(1−p) | ESS9 charge prob |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.936 | 0.060 | 0.950 | 0.048 | 0.000 |
| 2 | 0.957 | 0.041 | 0.965 | 0.034 | 0.000 |
| 3 | 0.983 | 0.017 | 0.999 | 0.001 | 0.000 |

The warmup is visible in the trajectory (4 → 2.53 → 1.03 → 1.0), but it did
not keep the probabilities away from 1. By the end of round 1 the discharge
gates are already above 0.93; the deterministic mask is one mode from the first
checkpoint. The permanent gates are mostly open (7 of 8 at round 1), not a
balanced 4/4.

## What the warmup did and did not do

- **Did:** keep the stochastic gate alive. The last training episode sampled
  243 joint modes; the Bernoulli p(1−p) at the saturated discharge gate is
  0.001–0.06 rather than exactly 0, and the ST backward has a non-zero path.
- **Did not:** prevent the deterministic gate from collapsing to one mode or
  prevent permanent open/closed gates. The hard threshold at logit 0 is
  unchanged by the temperature, and the actor still drives the logits far from
  0.

This is the expected distinction from the phase-16 audit: the deterministic
mask is a threshold count, not the exploration count. The stop rule therefore
does not fire on mode count alone; the economic and critic checks decide.

## Assessment

A1 is **operationally implemented and logged**, but it is **not a fix for the
deterministic saturation** observed at R1. The pilot's later rounds still move
the discharge probabilities to 0.96–0.999 and the policy to a high-quote
regime. A stronger intervention would need a gate-logit regulariser or a
different estimator; neither was allowed in this phase, and neither is
recommended as a default without the economic evidence.

## Boundary

No entropy, no REINFORCE, no Gumbel, no change to the gate estimator, no
change to the observation. Evaluation is τ=1. H2 and the legacy test are
untouched.
