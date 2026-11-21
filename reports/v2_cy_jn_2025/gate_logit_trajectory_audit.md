# Gate logit and probability trajectory (P0)

Fixed reference states: the truthful reference trajectory on one training day and one validation day (48 blocks total). Every checkpoint's actors are evaluated on the same observations. `det` is the deterministic hard gate; `sample` is the expected Bernoulli open frequency from the same probabilities; the sampled joint modes use 100 Bernoulli draws per state.

| round | episode | val mean ΔG | ESS9 ch/ds prob | ESS14 ch/ds prob | ESS17 ch/ds prob | ESS27 ch/ds prob | det modes | sample modes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 53 | -35.5 | 1.000/1.000 | 0.978/0.297 | 0.952/0.330 | 0.959/0.934 | 1 | 33 |
| 2 | 106 | -30.7 | 0.969/0.971 | 0.979/0.085 | 0.944/0.240 | 0.958/0.953 | 1 | 45 |
| 3 | 159 | -33.2 | 0.005/0.315 | 0.976/0.108 | 0.982/0.993 | 0.940/0.962 | 1 | 32 |
| 4 | 212 | -78.8 | 0.008/0.917 | 0.999/0.010 | 0.415/0.972 | 0.962/0.975 | 1 | 29 |
| 5 | 265 | -38.4 | 0.012/0.944 | 0.962/0.987 | 0.380/0.977 | 0.964/0.974 | 1 | 29 |
| 6 | 318 | -41.0 | 0.020/0.963 | 0.976/0.927 | 0.110/0.973 | 0.961/0.974 | 1 | 29 |
| 7 | 371 | -64.8 | 0.025/0.930 | 0.732/0.910 | 0.077/0.830 | 0.991/0.983 | 2 | 50 |
| 8 | 424 | -73.1 | 0.034/0.809 | 0.381/0.958 | 0.086/0.773 | 0.995/0.987 | 2 | 44 |
| 9 | 477 | -82.8 | 0.041/0.962 | 0.452/0.952 | 0.095/0.804 | 0.995/0.995 | 2 | 47 |
| 10 | 530 | -77.7 | 0.048/0.947 | 0.457/0.960 | 0.121/0.912 | 0.088/0.996 | 2 | 52 |

## Per-gate logits and deterministic open fraction

| round | gate | logit mean | prob mean | prob min | prob max | det open | sample open | price mean |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ESS9.charge | 8.33 | 1.000 | 0.999 | 1.000 | 1.00 | 1.000 | 149.96 |
| 1 | ESS9.discharge | 7.91 | 1.000 | 0.999 | 1.000 | 1.00 | 1.000 | 299.53 |
| 1 | ESS14.charge | 3.83 | 0.978 | 0.962 | 0.983 | 1.00 | 0.995 | 43.56 |
| 1 | ESS14.discharge | -0.86 | 0.297 | 0.288 | 0.319 | 0.00 | 0.274 | 0.04 |
| 1 | ESS17.charge | 3.01 | 0.952 | 0.923 | 0.959 | 1.00 | 0.958 | 23.07 |
| 1 | ESS17.discharge | -0.71 | 0.330 | 0.316 | 0.353 | 0.00 | 0.353 | 0.88 |
| 1 | ESS27.charge | 3.15 | 0.959 | 0.936 | 0.963 | 1.00 | 0.952 | 38.28 |
| 1 | ESS27.discharge | 2.66 | 0.934 | 0.905 | 0.939 | 1.00 | 0.951 | 37.41 |
| 2 | ESS9.charge | 3.46 | 0.969 | 0.945 | 0.978 | 1.00 | 0.982 | 139.77 |
| 2 | ESS9.discharge | 3.51 | 0.971 | 0.947 | 0.979 | 1.00 | 0.971 | 129.46 |
| 2 | ESS14.charge | 3.85 | 0.979 | 0.963 | 0.984 | 1.00 | 0.980 | 48.23 |
| 2 | ESS14.discharge | -2.38 | 0.085 | 0.075 | 0.117 | 0.00 | 0.092 | 0.02 |
| 2 | ESS17.charge | 2.84 | 0.944 | 0.915 | 0.955 | 1.00 | 0.948 | 13.70 |
| 2 | ESS17.discharge | -1.16 | 0.240 | 0.193 | 0.279 | 0.00 | 0.222 | 0.62 |
| 2 | ESS27.charge | 3.13 | 0.958 | 0.937 | 0.963 | 1.00 | 0.973 | 25.48 |
| 2 | ESS27.discharge | 3.01 | 0.953 | 0.930 | 0.959 | 1.00 | 0.939 | 43.73 |
| 3 | ESS9.charge | -5.27 | 0.005 | 0.004 | 0.013 | 0.00 | 0.002 | 0.20 |
| 3 | ESS9.discharge | -0.78 | 0.315 | 0.305 | 0.341 | 0.00 | 0.286 | 1.22 |
| 3 | ESS14.charge | 3.74 | 0.976 | 0.959 | 0.982 | 1.00 | 0.979 | 47.16 |
| 3 | ESS14.discharge | -2.11 | 0.108 | 0.099 | 0.144 | 0.00 | 0.092 | 0.04 |
| 3 | ESS17.charge | 4.01 | 0.982 | 0.965 | 0.986 | 1.00 | 0.980 | 9.35 |
| 3 | ESS17.discharge | 5.04 | 0.993 | 0.985 | 0.996 | 1.00 | 0.992 | 70.54 |
| 3 | ESS27.charge | 2.77 | 0.940 | 0.913 | 0.951 | 1.00 | 0.970 | 62.04 |
| 3 | ESS27.discharge | 3.25 | 0.962 | 0.941 | 0.969 | 1.00 | 0.990 | 199.18 |
| 4 | ESS9.charge | -4.88 | 0.008 | 0.006 | 0.017 | 0.00 | 0.005 | 0.28 |
| 4 | ESS9.discharge | 2.41 | 0.917 | 0.877 | 0.936 | 1.00 | 0.914 | 12.39 |
| 4 | ESS14.charge | 7.27 | 0.999 | 0.998 | 1.000 | 1.00 | 1.000 | 148.55 |
| 4 | ESS14.discharge | -4.59 | 0.010 | 0.008 | 0.022 | 0.00 | 0.005 | 0.00 |
| 4 | ESS17.charge | -0.35 | 0.415 | 0.402 | 0.432 | 0.00 | 0.385 | 0.49 |
| 4 | ESS17.discharge | 3.57 | 0.972 | 0.950 | 0.979 | 1.00 | 0.977 | 3.09 |
| 4 | ESS27.charge | 3.25 | 0.962 | 0.939 | 0.968 | 1.00 | 0.962 | 69.57 |
| 4 | ESS27.discharge | 3.69 | 0.975 | 0.958 | 0.979 | 1.00 | 0.978 | 195.14 |
| 5 | ESS9.charge | -4.46 | 0.012 | 0.009 | 0.024 | 0.00 | 0.010 | 0.52 |
| 5 | ESS9.discharge | 2.84 | 0.944 | 0.911 | 0.958 | 1.00 | 0.965 | 18.11 |
| 5 | ESS14.charge | 3.26 | 0.962 | 0.932 | 0.977 | 1.00 | 0.984 | 64.50 |
| 5 | ESS14.discharge | 4.37 | 0.987 | 0.974 | 0.995 | 1.00 | 0.985 | 74.70 |
| 5 | ESS17.charge | -0.49 | 0.380 | 0.371 | 0.402 | 0.00 | 0.394 | 0.59 |
| 5 | ESS17.discharge | 3.77 | 0.977 | 0.957 | 0.983 | 1.00 | 0.974 | 6.65 |
| 5 | ESS27.charge | 3.29 | 0.964 | 0.942 | 0.969 | 1.00 | 0.959 | 66.16 |
| 5 | ESS27.discharge | 3.63 | 0.974 | 0.956 | 0.978 | 1.00 | 0.986 | 174.45 |
| 6 | ESS9.charge | -3.88 | 0.020 | 0.016 | 0.039 | 0.00 | 0.014 | 1.00 |
| 6 | ESS9.discharge | 3.27 | 0.963 | 0.936 | 0.972 | 1.00 | 0.974 | 28.41 |
| 6 | ESS14.charge | 3.78 | 0.976 | 0.950 | 0.988 | 1.00 | 0.974 | 67.19 |
| 6 | ESS14.discharge | 2.62 | 0.927 | 0.875 | 0.971 | 1.00 | 0.945 | 105.42 |
| 6 | ESS17.charge | -2.10 | 0.110 | 0.100 | 0.151 | 0.00 | 0.131 | 0.28 |
| 6 | ESS17.discharge | 3.61 | 0.973 | 0.951 | 0.979 | 1.00 | 0.987 | 2.84 |
| 6 | ESS27.charge | 3.22 | 0.961 | 0.938 | 0.966 | 1.00 | 0.990 | 54.38 |
| 6 | ESS27.discharge | 3.63 | 0.974 | 0.956 | 0.978 | 1.00 | 0.995 | 161.90 |
| 7 | ESS9.charge | -3.68 | 0.025 | 0.020 | 0.046 | 0.00 | 0.023 | 1.36 |
| 7 | ESS9.discharge | 2.58 | 0.930 | 0.894 | 0.941 | 1.00 | 0.934 | 23.15 |
| 7 | ESS14.charge | 1.12 | 0.732 | 0.469 | 0.909 | 0.96 | 0.738 | 29.11 |
| 7 | ESS14.discharge | 2.54 | 0.910 | 0.781 | 0.981 | 1.00 | 0.915 | 162.08 |
| 7 | ESS17.charge | -2.49 | 0.077 | 0.069 | 0.114 | 0.00 | 0.083 | 0.12 |
| 7 | ESS17.discharge | 1.59 | 0.830 | 0.785 | 0.843 | 1.00 | 0.839 | 0.43 |
| 7 | ESS27.charge | 4.73 | 0.991 | 0.981 | 0.993 | 1.00 | 0.994 | 135.32 |
| 7 | ESS27.discharge | 4.05 | 0.983 | 0.965 | 0.988 | 1.00 | 0.984 | 182.42 |
| 8 | ESS9.charge | -3.35 | 0.034 | 0.028 | 0.059 | 0.00 | 0.024 | 1.82 |
| 8 | ESS9.discharge | 1.45 | 0.809 | 0.767 | 0.825 | 1.00 | 0.828 | 14.03 |
| 8 | ESS14.charge | -0.78 | 0.381 | 0.024 | 0.860 | 0.40 | 0.394 | 25.08 |
| 8 | ESS14.discharge | 3.79 | 0.958 | 0.838 | 0.998 | 1.00 | 0.977 | 197.75 |
| 8 | ESS17.charge | -2.37 | 0.086 | 0.077 | 0.125 | 0.00 | 0.098 | 0.21 |
| 8 | ESS17.discharge | 1.22 | 0.773 | 0.730 | 0.786 | 1.00 | 0.769 | 0.58 |
| 8 | ESS27.charge | 5.39 | 0.995 | 0.988 | 0.997 | 1.00 | 0.998 | 143.75 |
| 8 | ESS27.discharge | 4.40 | 0.987 | 0.973 | 0.993 | 1.00 | 0.990 | 168.62 |
| 9 | ESS9.charge | -3.17 | 0.041 | 0.035 | 0.068 | 0.00 | 0.045 | 2.69 |
| 9 | ESS9.discharge | 3.24 | 0.962 | 0.936 | 0.969 | 1.00 | 0.968 | 118.50 |
| 9 | ESS14.charge | -0.51 | 0.452 | 0.007 | 0.970 | 0.48 | 0.448 | 32.59 |
| 9 | ESS14.discharge | 4.15 | 0.952 | 0.777 | 0.999 | 1.00 | 0.965 | 172.50 |
| 9 | ESS17.charge | -2.26 | 0.095 | 0.087 | 0.135 | 0.00 | 0.103 | 0.45 |
| 9 | ESS17.discharge | 1.41 | 0.804 | 0.760 | 0.818 | 1.00 | 0.801 | 1.12 |
| 9 | ESS27.charge | 5.30 | 0.995 | 0.987 | 0.997 | 1.00 | 0.998 | 138.92 |
| 9 | ESS27.discharge | 5.48 | 0.995 | 0.988 | 0.999 | 1.00 | 0.994 | 246.00 |
| 10 | ESS9.charge | -3.00 | 0.048 | 0.043 | 0.076 | 0.00 | 0.043 | 3.72 |
| 10 | ESS9.discharge | 2.88 | 0.947 | 0.914 | 0.958 | 1.00 | 0.954 | 97.93 |
| 10 | ESS14.charge | -0.58 | 0.457 | 0.003 | 0.986 | 0.48 | 0.450 | 26.54 |
| 10 | ESS14.discharge | 4.79 | 0.960 | 0.790 | 1.000 | 1.00 | 0.962 | 171.82 |
| 10 | ESS17.charge | -1.98 | 0.121 | 0.110 | 0.165 | 0.00 | 0.118 | 0.83 |
| 10 | ESS17.discharge | 2.35 | 0.912 | 0.874 | 0.923 | 1.00 | 0.932 | 1.00 |
| 10 | ESS27.charge | -2.34 | 0.088 | 0.078 | 0.127 | 0.00 | 0.088 | 2.81 |
| 10 | ESS27.discharge | 5.75 | 0.996 | 0.990 | 0.999 | 1.00 | 0.998 | 218.46 |

## Findings

- **When the collapse starts:** the deterministic joint mode is 1 at round 1 and 2 at round 10. The hard-threshold mask is already collapsed at the first checkpoint.
- **Saturation:** the first round with a gate at mean probability >=0.9 or <=0.1 is round 1; several gates are extreme from round 1, while others remain moderate. The probabilities are not uniformly saturated.
- **Threshold vs sampling:** the fixed-state Bernoulli draws cover 33–52 distinct joint modes (range across rounds 29–52), against 1–2 deterministic modes. The deterministic threshold is the main reason the evaluated policy has so few modes; the stochastic gate is not frozen.
- **Validation timing:** the validation best is round 2, and the mask is already fixed before the later degradation. The degradation is not caused by a late collapse of the gate; the fixed mask and the quote branch are present from the first checkpoint.
- **Permanent gates:** the validation audit shows permanent open/closed gates from round 1 (6 open / 2 closed), changing to 3/3 by round 10. They are not a late-training phenomenon.
- **Exploration:** the Bernoulli samples enter many modes (29–52 of 256); the deterministic evaluation does not reflect that exploration.

The raw logits, probabilities, deterministic gates, sample frequencies and price means are in `gate_trajectory_by_round.json`.
