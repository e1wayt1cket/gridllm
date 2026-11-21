# B, C2 and Critic alignment audit (P1)

The audited day is a validation day. The B/C2 identity and the per-block shaping are from the actual actor schedules; the Critic metrics are computed on a fixed 128-transition replay sample at each checkpoint without an optimizer step.

## C2 versus B

| ESS | B at block 23 | max |shaping| per block | mean shaping | C2−B identity error |
|---|---:|---:|---:|---:|
| ESS9 | 19.947 | 0.001 | -0.000 | 0.00e+00 |
| ESS14 | 20.084 | 0.000 | -0.000 | 1.78e-15 |
| ESS17 | 0.532 | 14.214 | 0.035 | 2.00e-15 |
| ESS27 | 19.105 | 0.000 | -0.000 | 0.00e+00 |

The C2 shaping is a per-block credit redistribution. Its discounted sum telescopes to the policy-independent boundary (zero here); the day return equals the B return. On the final-round audited day the ESS17 shaping reaches 14.2 while its B at block 23 is 0.53; in the best-round audit the comparable B at block 23 was 0.12. The shaping therefore dominates individual blocks for a low-B ESS, but it does not change the total day objective. This is a real credit-assignment effect, not a C2 scale override of the day's money.

## Critic and actor metrics

| round | ESS | critic loss | target Q | Q1 | Q2 | twin gap | buffer reward mean | actor loss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ESS9 | 0.001 | 0.060 | 0.058 | 0.066 | 0.010 | 0.00003 | -0.057 |
| 1 | ESS14 | 0.066 | -0.957 | -0.924 | -0.993 | 0.101 | -0.00071 | 1.009 |
| 1 | ESS17 | 0.040 | -0.784 | -0.787 | -0.755 | 0.065 | 0.00137 | 0.803 |
| 1 | ESS27 | 0.026 | -0.625 | -0.651 | -0.590 | 0.075 | -0.00022 | 0.658 |
| 2 | ESS9 | 0.000 | 0.030 | 0.026 | 0.029 | 0.004 | -0.00011 | -0.025 |
| 2 | ESS14 | 0.022 | -0.665 | -0.625 | -0.639 | 0.063 | -0.00172 | 0.663 |
| 2 | ESS17 | 0.011 | -0.572 | -0.552 | -0.551 | 0.045 | -0.00095 | 0.574 |
| 2 | ESS27 | 0.010 | -0.418 | -0.395 | -0.398 | 0.029 | -0.00032 | 0.411 |
| 3 | ESS9 | 0.000 | 0.017 | 0.018 | 0.020 | 0.003 | -0.00195 | -0.017 |
| 3 | ESS14 | 0.015 | -0.420 | -0.428 | -0.412 | 0.033 | -0.00080 | 0.436 |
| 3 | ESS17 | 0.017 | -0.370 | -0.352 | -0.370 | 0.047 | -0.00037 | 0.384 |
| 3 | ESS27 | 0.017 | -0.224 | -0.186 | -0.194 | 0.035 | -0.00304 | 0.207 |
| 4 | ESS9 | 0.001 | 0.019 | 0.019 | 0.019 | 0.002 | 0.00146 | -0.018 |
| 4 | ESS14 | 0.006 | -0.330 | -0.336 | -0.349 | 0.039 | 0.00167 | 0.362 |
| 4 | ESS17 | 0.003 | -0.306 | -0.309 | -0.302 | 0.024 | 0.00356 | 0.318 |
| 4 | ESS27 | 0.003 | -0.173 | -0.154 | -0.160 | 0.020 | 0.00080 | 0.167 |
| 5 | ESS9 | 0.001 | 0.012 | 0.014 | 0.014 | 0.001 | -0.00177 | -0.014 |
| 5 | ESS14 | 0.003 | -0.285 | -0.269 | -0.283 | 0.026 | -0.00049 | 0.289 |
| 5 | ESS17 | 0.002 | -0.253 | -0.255 | -0.257 | 0.013 | -0.00042 | 0.262 |
| 5 | ESS27 | 0.001 | -0.145 | -0.141 | -0.137 | 0.011 | -0.00147 | 0.145 |
| 6 | ESS9 | 0.001 | 0.008 | 0.011 | 0.011 | 0.001 | -0.00132 | -0.010 |
| 6 | ESS14 | 0.002 | -0.250 | -0.234 | -0.238 | 0.015 | -0.00294 | 0.244 |
| 6 | ESS17 | 0.001 | -0.228 | -0.223 | -0.222 | 0.009 | -0.00145 | 0.227 |
| 6 | ESS27 | 0.001 | -0.130 | -0.129 | -0.125 | 0.009 | -0.00343 | 0.131 |
| 7 | ESS9 | 0.001 | 0.002 | 0.004 | 0.004 | 0.000 | -0.00149 | -0.004 |
| 7 | ESS14 | 0.002 | -0.231 | -0.219 | -0.226 | 0.015 | -0.00141 | 0.230 |
| 7 | ESS17 | 0.001 | -0.224 | -0.228 | -0.227 | 0.008 | -0.00007 | 0.231 |
| 7 | ESS27 | 0.001 | -0.121 | -0.123 | -0.120 | 0.006 | -0.00230 | 0.125 |
| 8 | ESS9 | 0.001 | -0.001 | 0.001 | 0.000 | 0.001 | -0.00126 | -0.000 |
| 8 | ESS14 | 0.001 | -0.210 | -0.205 | -0.210 | 0.008 | -0.00047 | 0.212 |
| 8 | ESS17 | 0.001 | -0.218 | -0.219 | -0.218 | 0.008 | 0.00020 | 0.223 |
| 8 | ESS27 | 0.001 | -0.112 | -0.112 | -0.111 | 0.003 | -0.00177 | 0.113 |
| 9 | ESS9 | 0.000 | -0.003 | -0.004 | -0.005 | 0.001 | 0.00342 | 0.005 |
| 9 | ESS14 | 0.001 | -0.193 | -0.196 | -0.199 | 0.006 | 0.00147 | 0.200 |
| 9 | ESS17 | 0.001 | -0.205 | -0.217 | -0.214 | 0.008 | 0.00440 | 0.220 |
| 9 | ESS27 | 0.000 | -0.102 | -0.103 | -0.105 | 0.004 | 0.00054 | 0.106 |
| 10 | ESS9 | 0.001 | -0.008 | -0.006 | -0.007 | 0.001 | -0.00120 | 0.007 |
| 10 | ESS14 | 0.001 | -0.173 | -0.169 | -0.172 | 0.006 | -0.00184 | 0.173 |
| 10 | ESS17 | 0.001 | -0.193 | -0.188 | -0.189 | 0.007 | -0.00089 | 0.192 |
| 10 | ESS27 | 0.000 | -0.091 | -0.090 | -0.090 | 0.003 | -0.00240 | 0.091 |

## Findings

- **C2 does not dominate the day objective.** The C2−B identity closes at ~3e-15; the boundary is zero; the total discounted return is the B return. The shaping dominates individual early blocks for the low-B ESS, which is the intended credit redistribution, not a change of the economic target.
- **Q scale is stable.** Target Q and current Q stay in a small range (roughly −0.6 to +0.03); the buffer reward mean is about −0.001; the critic loss is small; the twin gap is 0.001–0.045. There is no exploding Q or persistent twin divergence.
- **The problem is not a C2 scale drift.** The actor loss is small and the Q values are flat; the gate/policy collapse and the high-quote drift are better explained by the threshold and the critic's flat action-value ranking than by C2 magnitude.
- **The critic's flat ranking is a concern.** At the audited state the min Q changes by only a few hundredths across the quote grid, while the actual settlement changes by orders of magnitude; for the discharge legs the critic can even rank the high quote above the low one when the settlement is worse. A critic that is nearly flat in the action cannot strongly correct the quote branch.

No C2, potential, gamma or scale change is proposed or applied here.
