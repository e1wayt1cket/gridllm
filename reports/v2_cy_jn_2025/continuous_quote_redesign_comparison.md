# Continuous quote branch redesign (P0)

Read-only pure-function comparison. Nothing is wired into the actor, the
action spec or the trainer. Data: `continuous_quote_mapping_grid.csv`,
`continuous_quote_mapping_summary.json`.

The gate already makes the closed branch exactly zero. This report only covers
the open branch `q = h * g(z)`, with `h = 150` (charge) and `h = 300`
(discharge). A quote of 258–299 CNY/MWh corresponds to `q/h = 0.86–0.997`.

## Candidates

**A — current logistic**

```
g(z) = sigmoid(z)
```

**B — temperature-scaled logistic**

```
g(z) = sigmoid(z / tau),   tau in {0.5, 1, 2, 4}
```

`tau > 1` flattens the curve and widens the logit range; `tau < 1` sharpens it.
This is a new price-branch temperature; the existing `gate_temperature` applies
only to the gate logits.

**C — linear core with quadratic tails**

```
g(z) = 0.5 + 0.196 z                                   |z| <= 2.5
g(z) = 1 - 0.01 / (1 + 9.8 (z - 2.5))^2                z > 2.5
g(z) = 0.01 / (1 + 9.8 (-z - 2.5))^2                   z < -2.5
```

The core is linear from 0.01 to 0.99 (slope 0.49/2.5 = 0.196 per logit); the
quadratic tails match value and slope at ±2.5 and asymptote to 0/1. The map is
symmetric, `g(-z) = 1 - g(z)`, monotone, bounded, and needs no clipping.

## Summary

| candidate | mid dq/dz | min dq/dz on 0.86–0.997 | high/mid | dq/dz at 299 | dq/dz at 299.9 | logit for 299 | logit span 1–99% |
|---|---:|---:|---:|---:|---:|---:|---:|
| A logistic | 75.0 | 0.897 | 0.012 | 0.997 | 0.100 | 5.70 | 9.19 |
| B logistic τ=0.5 | 150.0 | 1.795 | 0.012 | 1.993 | 0.200 | 2.85 | 4.60 |
| B logistic τ=1 | 75.0 | 0.897 | 0.012 | 0.997 | 0.100 | 5.70 | 9.19 |
| B logistic τ=2 | 37.5 | 0.449 | 0.012 | 0.498 | 0.050 | 11.40 | 18.38 |
| B logistic τ=4 | 18.75 | 0.224 | 0.012 | 0.249 | 0.025 | 22.80 | 36.76 |
| **C linear core** | **58.8** | **9.662** | **0.164** | **11.316** | **0.358** | **2.57** | **5.00** |

`dq/dz` is CNY/MWh per unit price logit. All candidates are monotone, finite,
inside [0,h], need no clipping, and preserve the gate-closed exact zero.

## Gradient inside the problem region

Values are `dq/dz` for the discharge leg at the quote fractions the actor
actually reaches.

| quote (CNY/MWh) | fraction | A logistic | B τ=2 | C linear core |
|---:|---:|---:|---:|---:|
| 258 | 0.860 | 36.12 | 18.06 | **58.80** |
| 285 | 0.950 | 14.25 | 7.13 | **58.80** |
| 297 | 0.990 | 2.97 | 1.49 | **58.80** |
| 299.1 | 0.997 | 0.897 | 0.449 | **9.662** |
| 299.9 | 0.99967 | 0.099 | 0.049 | **0.352** |
| 299.97 | 0.9999 | 0.030 | 0.015 | **0.059** |

The logistic's high/mid ratio is invariant to τ: temperature scaling trades
the middle and the tails against each other but cannot change the shape. C
changes the shape. It gives 10.8× the logistic's minimum absolute gradient over
0.86–0.997 (13.7× on the high/mid ratio) and 3.6× at 299.9, while retaining 78%
of the logistic's midpoint gradient. It also needs only 2.57 logits to move
from the midpoint to 299, against 5.70 for A and 11.40 for B τ=2; with a logit
noise of 0.2, the actor is much closer to the high quote in units of its own
exploration.

## Reachability and coverage

| candidate | logit for 1% | logit for 99% | logit for 258 | logit for 299.9 | output at z=±50 |
|---|---:|---:|---:|---:|---|
| A | −4.60 | +4.60 | 1.82 | 8.01 | 0 / 1 |
| B τ=2 | −9.19 | +9.19 | 3.63 | 16.01 | 0 / 1 |
| C | −2.50 | +2.50 | 1.84 | 2.96 | ~0 / ~1 |

All three reach the legal box only asymptotically, so `h` itself is a limit,
not an exactly attained quote. C reaches 99.9% of the box within ±4.6 logits;
A needs ±9.2. This matters for a bounded actor output and for target smoothing.

## Why not select on the midpoint alone

C has a lower midpoint gradient than A (58.8 vs 75). That is the intended
trade: the actor is already at 258–299, not at the midpoint, so the relevant
gradient is the one at the state it occupies. A's midpoint advantage is 1.28×;
its high-region disadvantage is 10–20× and it needs twice as many logits to
cross the range.

## Recommendation

**Use C as the prototype mapping**, with A as the control and B τ=2 as a
secondary control. C is the only candidate that:

- keeps a usable gradient through 0.86–0.997 (9.66–58.8 CNY/logit);
- reaches 299 with 2.57 logits;
- stays bounded and monotone with no clipping;
- leaves the gate-closed exact zero untouched;
- is numerically stable at |z| ≤ 1000.

Do not hot-swap it into the existing checkpoint. The map changes the meaning of
the price logit, so an existing actor's parameters are not a valid
initialisation and the saved checkpoint schema cannot be reused unchanged. A
fresh single-seed prototype is required. Re-measure the quote distribution,
the 0.86–0.997 saturation share and the target-policy noise in quote space
after the change.

## Boundary

Pure functions only. No actor, critic, action bound, reward, gate estimator or
clearing engine was modified. The self-check in
`audit_continuous_quote_mapping.py` asserts monotonicity, boundedness,
finiteness, `g(0)=0.5`, and that C's high-region minimum gradient exceeds A's.
