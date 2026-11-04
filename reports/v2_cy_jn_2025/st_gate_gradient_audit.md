# ST gate-gradient audit (P0)

The actor loss is the real training objective on a 128-transition sample from each checkpoint's replay buffer. No optimizer step is applied and no checkpoint is written. The gate forward is hard; the backward is the straight-through surrogate `p(1-p)`. This is not a finite-difference check and is not an unbiased policy gradient.

| round | ESS | gate grad (ch/ds) | price grad (ch/ds) | surrogate p(1-p) (ch/ds) | gate weight grad | Adam gate update | Adam price update | actor loss | critic loss | target Q | Q gap |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ESS9 | 6.968e-07/8.076e-07 | 7.792e-07/3.168e-06 | 0.000/0.000 | 3.246e-05 | 1.970e-04 | 2.421e-04 | -0.057 | 0.001 | 0.060 | 0.010 |
| 1 | ESS14 | 1.411e-04/8.643e-06 | 4.726e-03/1.907e-05 | 0.019/0.208 | 5.113e-03 | 3.428e-04 | 2.425e-04 | 1.009 | 0.066 | -0.957 | 0.101 |
| 1 | ESS17 | 1.876e-04/6.639e-05 | 2.964e-03/2.637e-04 | 0.043/0.221 | 5.899e-03 | 3.638e-04 | 1.166e-03 | 0.803 | 0.040 | -0.784 | 0.065 |
| 1 | ESS27 | 2.421e-04/4.950e-04 | 2.436e-03/4.237e-03 | 0.041/0.061 | 8.514e-03 | 8.308e-04 | 1.566e-03 | 0.658 | 0.026 | -0.625 | 0.075 |
| 2 | ESS9 | 2.833e-05/2.198e-05 | 6.334e-05/5.658e-04 | 0.025/0.023 | 5.179e-03 | 5.375e-03 | 5.521e-03 | -0.025 | 0.000 | 0.030 | 0.004 |
| 2 | ESS14 | 1.142e-04/7.586e-07 | 3.252e-03/3.922e-06 | 0.019/0.073 | 6.369e-04 | 2.036e-04 | 2.529e-04 | 0.663 | 0.022 | -0.665 | 0.063 |
| 2 | ESS17 | 8.652e-05/1.052e-05 | 1.350e-03/3.379e-05 | 0.052/0.176 | 8.563e-04 | 4.929e-04 | 7.749e-04 | 0.574 | 0.011 | -0.572 | 0.045 |
| 2 | ESS27 | 6.255e-05/1.338e-04 | 1.137e-03/2.201e-03 | 0.042/0.045 | 3.018e-04 | 2.470e-04 | 5.041e-04 | 0.411 | 0.010 | -0.418 | 0.029 |
| 3 | ESS9 | 1.510e-08/2.757e-06 | 0.000e+00/5.444e-06 | 0.005/0.215 | 1.080e-04 | 5.381e-05 | 4.799e-05 | -0.017 | 0.000 | 0.017 | 0.003 |
| 3 | ESS14 | 1.543e-04/8.583e-07 | 3.778e-03/9.898e-07 | 0.022/0.092 | 9.538e-04 | 3.572e-04 | 6.041e-04 | 0.436 | 0.015 | -0.420 | 0.033 |
| 3 | ESS17 | 2.556e-05/4.082e-05 | 7.049e-04/3.651e-03 | 0.017/0.006 | 5.968e-04 | 3.247e-03 | 3.934e-03 | 0.384 | 0.017 | -0.370 | 0.047 |
| 3 | ESS27 | 2.546e-04/2.819e-04 | 2.617e-03/2.365e-03 | 0.057/0.038 | 4.192e-03 | 1.971e-03 | 2.163e-03 | 0.207 | 0.017 | -0.224 | 0.035 |
| 4 | ESS9 | 1.363e-08/3.656e-06 | 0.000e+00/3.494e-05 | 0.007/0.066 | 5.525e-05 | 1.578e-03 | 2.038e-03 | -0.018 | 0.001 | 0.019 | 0.002 |
| 4 | ESS14 | 6.339e-06/2.496e-09 | 5.482e-05/0.000e+00 | 0.001/0.008 | 2.810e-04 | 1.280e-04 | 2.055e-04 | 0.362 | 0.006 | -0.330 | 0.039 |
| 4 | ESS17 | 4.683e-06/2.283e-06 | 1.228e-05/8.075e-05 | 0.242/0.024 | 1.660e-04 | 2.013e-04 | 1.697e-04 | 0.318 | 0.003 | -0.306 | 0.024 |
| 4 | ESS27 | 7.993e-05/8.212e-05 | 1.114e-03/1.124e-03 | 0.035/0.023 | 4.014e-04 | 3.684e-04 | 7.220e-04 | 0.167 | 0.003 | -0.173 | 0.020 |
| 5 | ESS9 | 1.005e-08/1.667e-06 | 1.293e-07/2.589e-05 | 0.011/0.046 | 2.248e-05 | 2.160e-04 | 3.340e-04 | -0.014 | 0.001 | 0.012 | 0.001 |
| 5 | ESS14 | 6.893e-05/2.002e-05 | 1.075e-03/1.369e-03 | 0.036/0.011 | 1.582e-03 | 1.454e-03 | 2.003e-03 | 0.289 | 0.003 | -0.285 | 0.026 |
| 5 | ESS17 | 4.102e-06/3.437e-06 | 9.539e-06/1.124e-04 | 0.235/0.022 | 1.453e-04 | 7.803e-04 | 7.879e-04 | 0.262 | 0.002 | -0.253 | 0.013 |
| 5 | ESS27 | 5.888e-05/5.919e-05 | 8.221e-04/7.929e-04 | 0.037/0.027 | 1.669e-03 | 3.033e-04 | 3.689e-04 | 0.145 | 0.001 | -0.145 | 0.011 |
| 6 | ESS9 | 2.383e-08/1.438e-06 | 3.063e-07/4.061e-05 | 0.020/0.034 | 1.559e-04 | 8.810e-04 | 7.494e-04 | -0.010 | 0.001 | 0.008 | 0.001 |
| 6 | ESS14 | 2.474e-05/8.881e-05 | 5.265e-04/8.179e-04 | 0.026/0.057 | 4.001e-03 | 3.524e-04 | 6.237e-04 | 0.244 | 0.002 | -0.250 | 0.015 |
| 6 | ESS17 | 6.689e-07/1.077e-06 | 2.638e-06/3.035e-05 | 0.097/0.026 | 3.253e-05 | 1.583e-04 | 1.626e-04 | 0.227 | 0.001 | -0.228 | 0.009 |
| 6 | ESS27 | 2.883e-05/3.130e-05 | 2.831e-04/4.439e-04 | 0.041/0.028 | 4.036e-04 | 2.248e-04 | 4.144e-04 | 0.131 | 0.001 | -0.130 | 0.009 |
| 7 | ESS9 | 5.100e-08/8.716e-07 | 1.152e-07/1.097e-05 | 0.023/0.061 | 8.972e-06 | 9.870e-04 | 8.998e-04 | -0.004 | 0.001 | 0.002 | 0.000 |
| 7 | ESS14 | 1.402e-04/1.027e-04 | 4.134e-04/4.954e-04 | 0.211/0.051 | 7.373e-03 | 1.168e-03 | 7.506e-04 | 0.230 | 0.002 | -0.231 | 0.015 |
| 7 | ESS17 | 1.324e-07/4.388e-07 | 6.449e-07/2.421e-06 | 0.068/0.137 | 1.089e-05 | 2.164e-03 | 2.123e-03 | 0.231 | 0.001 | -0.224 | 0.008 |
| 7 | ESS27 | 8.301e-06/1.295e-05 | 6.647e-05/2.910e-04 | 0.009/0.017 | 6.698e-04 | 4.204e-04 | 5.428e-04 | 0.125 | 0.001 | -0.121 | 0.006 |
| 8 | ESS9 | 5.206e-08/6.430e-07 | 1.232e-07/3.167e-06 | 0.033/0.152 | 4.512e-06 | 3.141e-04 | 2.663e-04 | -0.000 | 0.001 | -0.001 | 0.001 |
| 8 | ESS14 | 6.984e-05/2.792e-05 | 1.811e-04/2.809e-04 | 0.131/0.022 | 1.298e-03 | 9.214e-04 | 5.678e-04 | 0.212 | 0.001 | -0.210 | 0.008 |
| 8 | ESS17 | 1.565e-07/7.403e-07 | 4.165e-07/3.583e-06 | 0.077/0.173 | 6.231e-05 | 2.902e-04 | 1.234e-04 | 0.223 | 0.001 | -0.218 | 0.008 |
| 8 | ESS27 | 2.629e-06/6.014e-06 | 1.520e-05/1.766e-04 | 0.005/0.013 | 4.660e-05 | 2.403e-04 | 6.569e-04 | 0.113 | 0.001 | -0.112 | 0.003 |
| 9 | ESS9 | 9.881e-08/2.549e-06 | 1.269e-07/3.769e-05 | 0.037/0.034 | 9.414e-05 | 4.968e-04 | 1.487e-04 | 0.005 | 0.000 | -0.003 | 0.001 |
| 9 | ESS14 | 3.661e-05/1.108e-05 | 1.099e-04/1.737e-04 | 0.106/0.022 | 1.343e-03 | 5.212e-04 | 4.565e-04 | 0.200 | 0.001 | -0.193 | 0.006 |
| 9 | ESS17 | 1.707e-07/1.348e-06 | 5.339e-07/7.365e-06 | 0.082/0.153 | 1.329e-04 | 9.075e-04 | 3.462e-04 | 0.220 | 0.001 | -0.205 | 0.008 |
| 9 | ESS27 | 2.291e-06/5.160e-06 | 1.924e-05/1.464e-04 | 0.005/0.004 | 1.111e-04 | 2.966e-04 | 5.755e-04 | 0.106 | 0.000 | -0.102 | 0.004 |
| 10 | ESS9 | 9.839e-08/4.379e-06 | 2.360e-07/5.093e-05 | 0.047/0.049 | 3.069e-04 | 1.223e-03 | 1.523e-03 | 0.007 | 0.001 | -0.008 | 0.001 |
| 10 | ESS14 | 2.068e-05/4.560e-06 | 1.160e-04/8.246e-05 | 0.076/0.025 | 2.891e-04 | 1.648e-04 | 3.314e-04 | 0.173 | 0.001 | -0.173 | 0.006 |
| 10 | ESS17 | 7.936e-07/3.826e-07 | 1.496e-06/4.097e-06 | 0.104/0.079 | 1.085e-04 | 6.303e-04 | 4.112e-04 | 0.192 | 0.001 | -0.193 | 0.007 |
| 10 | ESS27 | 1.464e-06/3.631e-06 | 4.426e-07/1.685e-04 | 0.076/0.004 | 3.514e-05 | 1.363e-03 | 1.474e-03 | 0.091 | 0.000 | -0.091 | 0.003 |

## Path checks

- The actor output's first two components are gate logits; the last two are price logits. `forward_policy` applies a hard forward gate and the ST surrogate backward; the price branch keeps its exact sigmoid derivative.
- The critic receives the **executed quote** (`replaced`), not the gate logit or the probability. The target actor uses `target_quotes` with a deterministic hard gate.
- The Bernoulli draw and the executed quote come from the same `forward_policy` call; the replay buffer stores the executed quote, gate, probability and logit.
- No `detach` or wrong index cuts the target agent's actor gradient: the gate-logit gradient is nonzero for every ESS and leg. The other agents' actions in the batch are constants, as intended for a centralized critic.

## Findings

- **ST updates the gate logits:** the gate-logit and final-layer gate gradients are nonzero for every ESS and leg; the path is connected. The Adam update on the gate rows is comparable to the price branch (about 0.5–1.5× in rounds 2 and 10), so the gate parameters are not frozen by the optimizer.
- **The surrogate factor is small at saturation:** `p(1-p)` is small when a gate is extreme, so the gate-logit gradient is weak even when the Adam-normalized update is not. The gate can move, but the gradient signal is attenuated by the saturated probability.
- **The gate is not detached or mis-indexed.** The issue is the biased surrogate plus the hard threshold, not a broken tensor path.
- **The target and critic semantics are correct:** the target actor uses a deterministic hard gate; the critic consumes executed quotes; the Bernoulli draw and the executed quote come from the same `forward_policy` call.
- **ST is biased.** The backward is a surrogate, not the true derivative of the hard gate. A finite-difference check would not make it the true hard-policy gradient.

Full per-round/per-agent data: `st_gate_gradient_by_round.json`.
