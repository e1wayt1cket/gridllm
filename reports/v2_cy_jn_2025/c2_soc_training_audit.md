# C2 and SOC audit on the trained checkpoints

Run: `EXPLORATORY_NEAR_B_C2_GATE_seed11_v2`; audited day: `2022-07-23|2025-07-13`. No test and no H2 holdout.

| check | result |
|---|---|
| max raw vs clipped SOC deviation | 0.000e+00 |
| max raw SOC band overshoot | 0.000e+00 |
| terminal SOC max error | 0.000e+00 |
| voltage / line violation cells | 0 / 0 |
| B vs C2 discounted identity, max error | 2.984e-15 |
| shaping sum vs boundary, max error | 3.553e-15 |
| B paid only at block 23 | True |
| B at block 23 equals delta | True |
| C2 shaping present | True |
| physical checks | True |

The 1e-6 tolerance matches the solver-feasibility scale. The maximum observed raw SOC overshoot is recorded above; a larger excursion would raise, so the tolerance does not admit a real physical violation.
