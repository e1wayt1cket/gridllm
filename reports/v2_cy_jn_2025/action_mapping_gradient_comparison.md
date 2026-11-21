# Tanh vs truthful-gate: offline gradient comparison (V2-CY-JN-2025)

No training. Fixed logit grid; analytic derivatives checked against a central
difference. Data: `data/processed/v2_cy_jn_2025/pilot/action_mapping_grid.csv`
and `action_mapping_gradient_check.json`.

## Maps compared

- **Old:** $a=\frac{h}{2}+\tanh(z)\frac{h}{2}$, $da/dz=\frac{h}{2}(1-\tanh^2 z)$.
- **New, gate closed:** $a=0$, no continuous derivative (a discrete mode).
- **New, gate open:** $a=h\,\sigma(z)$, $da/dz=h\,\sigma(1-\sigma)$.

The discharge leg is the charge leg scaled by $h$: every value and derivative
below doubles at $h=300$. The table therefore shows the charge leg
($h=150$) and notes the discharge factor.

## The grid (charge, $h=150$)

| z | old quote | old da/dz | gate closed | closed da/dz | gate open quote | open da/dz |
|---:|---:|---:|---:|---:|---:|---:|
| -50.0 | 0 | 0 | 0 | 0 | 2.89e-20 | 2.89e-20 |
| -10.0 | 3.09e-07 | 6.18e-07 | 0 | 0 | 0.0068 | 0.0068 |
| -5.0 | 0.0068 | 0.0136 | 0 | 0 | 1.0039 | 0.9972 |
| -3.0 | 0.3709 | 0.7400 | 0 | 0 | 7.1139 | 6.7765 |
| -1.0 | 17.8804 | 31.4981 | 0 | 0 | 40.3412 | 29.4918 |
| -0.5 | 40.3412 | 58.9836 | 0 | 0 | 56.6311 | 35.2506 |
| 0.0 | 75.0000 | 75.0000 | 0 | 0 | 75.0000 | 37.5000 |
| 0.5 | 109.6588 | 58.9836 | 0 | 0 | 93.3689 | 35.2506 |
| 1.0 | 132.1196 | 31.4981 | 0 | 0 | 109.6588 | 29.4918 |
| 3.0 | 149.6291 | 0.7400 | 0 | 0 | 142.8861 | 6.7765 |
| 5.0 | 149.9932 | 0.0136 | 0 | 0 | 148.9961 | 0.9972 |
| 10.0 | 150.0000 | 6.18e-07 | 0 | 0 | 149.9932 | 0.0068 |
| 50.0 | 150.0000 | 0 | 0 | 0 | 150.0000 | 0 |

Analytic vs central-difference maximum error: old 6.4e-9, open 9.2e-10.

## What the numbers say

1. **Exact truth.** The gate-closed branch is exactly 0 at every logit,
   including −1e6 and +1e6. The old map reaches 0 only through saturation: at
   $z=-5$ it is 0.0068 CNY/MWh, at −10 it is 3.1e-7; at −50 `tanh` rounds to
   −1.0 in float64 and the printed 0 is a saturation artefact with derivative
   0, not a finite control. **The old map's exactness is the same failure mode
   as its zero gradient.**
2. **Gradient at truth.** Old $da/dz\to0$ as truth is approached (0.0136 at
   −5, 0 at −50). The closed gate has no continuous derivative; the discrete
   mode is where exact truth lives. The open branch also has a vanishing
   derivative at both ends (0.0068 at ±10), but it is not the branch that
   expresses truth.
3. **Mid-box and high end.** At $z=0$ both maps give the midpoint (75); the
   open branch's derivative is 37.5 there, the largest in the grid. Near the
   upper end both vanish; the open branch approaches 150 asymptotically and
   never exceeds it.
4. **No clipping.** Neither map clips; the open branch is bounded by
   construction.

## The discrete gate is not a derivative

The gate has no $da/dz$. A straight-through estimator is a **biased**
surrogate for a deterministic discrete action; a Bernoulli/REINFORCE head is an
unbiased policy-gradient estimator for a stochastic gate, with higher variance;
Gumbel-Softmax is a biased, temperature-dependent surrogate whose executed
action must still be hard. None of these is the analytic derivative of the
continuous branch, and they must not be reported as such. The estimator choice
is compared in `truthful_gate_action_prototype.md`.

## What this does not show

The comparison is local. It shows that the gate can express truth exactly and
that the old map cannot do so without saturation; it does **not** show that a
control policy using the gate will earn more. That is an empirical question
that requires training on the training/validation splits, with a new holdout
for any performance claim. No random-initialised critic gradient is used as
economic evidence here.
