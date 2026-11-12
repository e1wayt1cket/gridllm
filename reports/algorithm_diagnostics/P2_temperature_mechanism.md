# P2-TEMP — what the gate temperature changes

Protocol `P2_CONTROLLED_DIAGNOSTICS_V2`, stage 1 (offline, read-only).
Written 2026-11-08. Interpreter
`C:\Users\26036\.conda\envs\energy_env\python.exe` (CPython 3.10.20, conda-forge),
torch `2.11.0+cpu`, numpy `1.24.3`, scipy `1.15.3`, gurobipy `13.0.1`.

Collector: `src/diagnose_p2.py` (new, unpinned), function `temp_study`.
Command:

```
PYTHONPATH=src "C:/Users/26036/.conda/envs/energy_env/python.exe" \
  src/diagnose_p2.py temp
```

Artifact: `reports/algorithm_diagnostics/P2_evidence.json` → `temp`.

The temperature and gate implementation were not modified. Every measurement
calls `da_quote_map.forward_policy_v3` as frozen; only its existing
`gate_temperature` argument varies.

## Method

One charge leg (leg 0, `LEG_HIGHS[0] = 150`) is probed with the other gate
pinned open at logit `+4`. The policy logits are
`[g, 4.0, 1.5, 0.0]`, so the open price quote is
`150 * g(1.5) = 119.1000` CNY/MWh and `g'(1.5) = 0.196` exactly. The gate logit
`g` takes `{-4, -2, -1, 0, 1, 2, 4}`. For each `tau` in `{1.0, 2.5, 4.0}` and
each `g`, 4000 matched torch seeds are drawn from the same RNG stream, so the
seed `k` sees the same uniform at every `tau`. Three quantities are averaged
over the seeds: the forward quote, the gate's hard value, and the backward
gradient reaching the gate logit and the price logit.

## Answers to the six questions

| # | question | measured answer | verdict |
|---|---|---|---|
| 1 | does the gate forward probability depend on `tau` | yes: `p = sigmoid(g / tau)` | **PASS** |
| 2 | does the Bernoulli hard sample depend on `tau` | yes, through `p`; with matched seeds 0–26% of draws flip, and exactly 0 at `g = 0` | **PASS** |
| 3 | does the straight-through backward carry an explicit `1 / tau` | yes: surrogate gradient `p (1-p) / tau * price_quote`, exact to `1.07e-14` | **PASS** |
| 4 | does `p` itself change with `tau` | yes, same map as 1 and 2 (at `g = 0` it is `0.5` for every `tau`) | **PASS** |
| 5 | does the price-branch gradient change with `tau` | yes, but only through the gate: `dquote/dz = gate * high * g'(z)`, no `1 / tau` on the price path | **PASS** |
| 6 | are the target actor and evaluation affected | no: eval, target and critic actions are bitwise identical across `tau` | **PASS** |

### Measured tables

`dQ/dgate` is the gradient reaching the gate logit; `dQ/dprice` the gradient
reaching the price logit. The analytic surrogate for `dQ/dgate` is
`p (1-p) / tau * 119.1`; for `dQ/dprice` it is `gate * 150 * 0.196`.

**tau = 1.0**

| g | p | activation freq | mean quote | E[quote] | dQ/dgate | dQ/dprice (meas / p·150·g') | grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -4 | 0.0180 | 0.0173 | 2.054 | 2.142 | 2.1036 | 0.5071 / 0.5288 | 2.1639 |
| -2 | 0.1192 | 0.1253 | 14.917 | 14.197 | 12.5047 | 3.6824 / 3.5046 | 13.0356 |
| -1 | 0.2689 | 0.2625 | 31.264 | 32.031 | 23.4165 | 7.7175 / 7.9069 | 24.6555 |
| 0 | 0.5000 | 0.4988 | 59.401 | 59.550 | 29.7750 | 14.6632 / 14.7000 | 33.1898 |
| +1 | 0.7311 | 0.7278 | 86.675 | 87.069 | 23.4165 | 21.3958 / 21.4931 | 31.7193 |
| +2 | 0.8808 | 0.8788 | 104.659 | 104.903 | 12.5047 | 25.8352 / 25.8954 | 28.7024 |
| +4 | 0.9820 | 0.9825 | 117.016 | 116.958 | 2.1036 | 28.8855 / 28.8712 | 28.9620 |

**tau = 2.5**

| g | p | activation freq | mean quote | E[quote] | dQ/dgate | dQ/dprice (meas / p·150·g') | grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -4 | 0.1680 | 0.1693 | 20.158 | 20.007 | 6.6583 | 4.9760 / 4.9387 | 8.3123 |
| -2 | 0.3100 | 0.3065 | 36.504 | 36.924 | 10.1907 | 9.0111 / 9.1148 | 13.6033 |
| -1 | 0.4013 | 0.3967 | 47.253 | 47.796 | 11.4460 | 11.6645 / 11.7986 | 16.3423 |
| 0 | 0.5000 | 0.4988 | 59.401 | 59.550 | 11.9100 | 14.6632 / 14.7000 | 18.8907 |
| +1 | 0.5987 | 0.5982 | 71.252 | 71.304 | 11.4460 | 17.5885 / 17.6014 | 20.9850 |
| +2 | 0.6900 | 0.6820 | 81.226 | 82.176 | 10.1907 | 20.0508 / 20.2852 | 22.4919 |
| +4 | 0.8320 | 0.8347 | 99.419 | 99.093 | 6.6583 | 24.5416 / 24.4613 | 25.4288 |

**tau = 4.0**

| g | p | activation freq | mean quote | E[quote] | dQ/dgate | dQ/dprice (meas / p·150·g') | grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -4 | 0.2689 | 0.2625 | 31.264 | 32.031 | 5.8541 | 7.7175 / 7.9069 | 9.6866 |
| -2 | 0.3775 | 0.3748 | 44.633 | 44.965 | 6.9972 | 11.0177 / 11.0997 | 13.0518 |
| -1 | 0.4378 | 0.4353 | 51.838 | 52.145 | 7.3286 | 12.7964 / 12.8720 | 14.7464 |
| 0 | 0.5000 | 0.4988 | 59.401 | 59.550 | 7.4437 | 14.6632 / 14.7000 | 16.4445 |
| +1 | 0.5622 | 0.5560 | 66.220 | 66.955 | 7.3286 | 16.3464 / 16.5280 | 17.9141 |
| +2 | 0.6225 | 0.6200 | 73.842 | 74.135 | 6.9972 | 18.2280 / 18.3003 | 19.5249 |
| +4 | 0.7311 | 0.7278 | 86.675 | 87.069 | 5.8541 | 21.3958 / 21.4931 | 22.1823 |

- `dQ/dgate` equals the analytic surrogate at every one of the 21 points,
  maximum absolute error `1.07e-14`.
- `dQ/dprice` equals `activation frequency * 150 * 0.196`, maximum absolute
  error `1.07e-14`. Its difference from the `p`-based column is Monte-Carlo
  error in the sampled activation frequency (4000 draws), at most `0.234`.

### Forward change at matched seeds

Fraction of the 4000 matched seeds whose quote differs from the `tau = 1.0`
value with the same logits:

| g | tau = 2.5 | tau = 4.0 |
|---:|---:|---:|
| -4 | 0.152 | 0.245 |
| -2 | 0.181 | 0.250 |
| -1 | 0.134 | 0.173 |
| 0 | 0.000 | 0.000 |
| +1 | 0.130 | 0.172 |
| +2 | 0.197 | 0.259 |
| +4 | 0.148 | 0.255 |

At `g = 0` the probability is `0.5` for every `tau`, so the same uniform draw
flips the same way and the forward quote is identical. Everywhere else the
hard sample — and therefore the quote actually sent to the clearing — changes
with `tau`.

### Finite difference

Central difference of the **smooth expected quote** with respect to the gate
logit, `h = 1e-5`, against the analytic surrogate:

| tau | max abs error |
|---:|---:|
| 1.0 | 7.97e-10 |
| 2.5 | 6.49e-10 |
| 4.0 | 6.49e-10 |

The finite difference of the **sampled** forward quote at a fixed seed is
exactly `0` at every point, because the hard Bernoulli draw is piecewise
constant in the logit and a `1e-5` perturbation almost never flips it. The
sampled forward is therefore not differentiable in the usual sense; the
surrogate above is the only gradient that exists, and it is what the actor
loss uses.

### Evaluation, target and critic

For every `tau`, `forward_policy_v3(..., train=False)` ignores the temperature
argument bitwise, and `target_quotes_v3(policy)` and `critic_action_v3(policy)`
both equal that evaluation-mode output bitwise. The evaluation path takes the
hard threshold `gate_logit > 0`, which contains no `tau`.

## What tau does, in one paragraph

`tau` replaces the gate logit with `g / tau` before the sigmoid. Through
`p = sigmoid(g / tau)` it changes three things at once: the probability used by
the hard Bernoulli draw (so the forward quote distribution and the realised
quote change), the gate surrogate gradient `p (1-p) / tau` (which carries an
explicit `1 / tau`), and the expected gate multiplying the price-branch
gradient (so the price-logit gradient changes too, but only through the gate's
mean; the price map itself has no `tau`). It changes nothing on the evaluation
or target path.

The two factors in the gate gradient act in opposite directions off-centre:
at `g = 0`, `p (1-p) = 0.25` for every `tau`, so the gradient scales exactly as
`1 / tau` — `29.7750` at `tau = 1`, `11.9100` at `2.5` (ratio 2.500) and
`7.4437` at `4` (ratio 4.000). At `g = +4` the rising `p (1-p)` partly offsets
it, and `tau = 4` retains `0.359` of the `tau = 1` gradient rather than `0.25`.

## Relation to the two best checkpoints

Seed 11's best checkpoint resumes at episode 53, where the frozen schedule
gives `tau = 2.5`; seed 23's best resumes at episode 212, where it gives
`tau = 1.0`. At a gate logit near zero, seed 11's gate gradient is therefore
`2.5x` smaller than seed 23's at the same logit, and its gradient norm is
`18.89` against `33.19`, while the price-branch gradient at `g = 0` is the same
`14.66` for both. That is the frozen annealing schedule, not a defect: the
P1 observation that the gate-logit gradient is much smaller than the
price-logit gradient is partly this factor, and comparing gradient magnitudes
across the two checkpoints without accounting for `tau` would be wrong.

## Conclusion

**`GATE_FORWARD_AND_BACKWARD`** — the temperature changes the gate's forward
distribution and its realised quote, and it changes both gradient branches:
the gate logit through `p (1-p) / tau`, the price logits through the gate's
mean. Evaluation and the target actor are untouched.

The other three classes are excluded by measurement, not assumption:
`GATE_GRADIENT_ONLY` is contradicted by the activation-frequency and
matched-seed quote differences, `GATE_PROBABILITY_AND_GRADIENT` by the
forward-quote differences, and `OTHER_EFFECT` by the eval/target invariance
combined with the two exact gradient identities above.

## What this establishes and what it does not

Established (**PASS**): the mechanism by which `tau` enters each branch, to
`1e-14` on the gradients, with matched seeds across `tau`.

Not established:

- **NOT_TESTED** — whether the `tau` difference is large enough to matter for
  the observed payoff degradation. That is the control matrix (P2-MATRIX A1),
  which holds everything else fixed and varies `tau` on the seed 11
  checkpoint.
- **NOT_TESTED** — any statement about the trainer's temperature *schedule*
  (`current_gate_temperature` as a function of episode). Only the map's
  response at fixed `tau` was measured.
- **INFERRED** — nothing; every number above is a direct measurement in this
  stage.
