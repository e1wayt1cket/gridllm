# D1 continuous quote mapping v2 audit (P0)

Implemented in `src/da_quote_map.py`, action schema `storage_quote_v3`
(version 3), new checkpoint schema `matd3-checkpoint-v3`. R2/R10 are not
loaded as initialisations. Data: `da_quote_map_grid.csv`,
`da_gate_offline_validation.json`.

## Exact map

The phase-16 audited candidate C is used verbatim:

```
g(z) = 0.5 + 0.196 z                               |z| <= 2.5
g(z) = 1 - 0.01 / (1 + 9.8 (z - 2.5))^2             z > 2.5
g(z) = 0.01 / (1 + 9.8 (-z - 2.5))^2                z < -2.5
q = 0 if gate closed, else h * g(z)
```

`h = 150` for charge and `h = 300` for discharge. The logit is clamped only at
a ±1e6 numerical guard so the quadratic denominator cannot overflow; that is
not a quote clipping. The gate is the existing hard-forward / straight-through
Bernoulli and is independent per leg.

## Grid and derivatives

| logit | fraction | charge CNY | discharge CNY | dq/dz charge | dq/dz discharge |
|---:|---:|---:|---:|---:|---:|
| −2.5 | 0.010 | 1.5 | 3.0 | 29.4 | 58.8 |
| 0.0 | 0.500 | 75.0 | 150.0 | 29.4 | 58.8 |
| 1.8367 | 0.860 | 129.0 | 258.0 | 29.4 | 58.8 |
| 2.5 | 0.990 | 148.5 | 297.0 | 29.4 | 58.8 |
| 2.5747 | 0.9967 | 149.5 | 299.0 | 5.66 | 11.32 |
| 2.9569 | 0.99967 | 149.95 | 299.9 | 0.179 | 0.358 |
| 5.0 | 0.999985 | 149.998 | 299.996 | 0.0018 | 0.0035 |

The derivative is 0.196 fraction per logit in the linear core, so 29.4
CNY/logit for charge and 58.8 for discharge.

| discharge fraction | quote CNY | C dq/dz | old sigmoid dq/dz | ratio |
|---:|---:|---:|---:|---:|
| 0.860 | 258 | 58.8 | 36.1 | 1.63× |
| 0.950 | 285 | 58.8 | 14.3 | 4.12× |
| 0.990 | 297 | 58.8 | 2.97 | 19.8× |
| 0.997 | 299.1 | 11.32 | 0.897 | 12.6× |
| 0.99967 | 299.9 | 0.358 | 0.100 | 3.58× |

The high-region minimum gradient over 0.86–0.997 is **9.66 CNY/logit** for
discharge, against **0.897** for the old sigmoid, a **10.77×** improvement.
The logit needed to reach 299 is 2.57 (sigmoid 5.70); to 258 it is 1.84
(sigmoid 1.82). The midpoint gradient is 58.8, 78% of the sigmoid's 75; the map
deliberately trades some midpoint slope for the high-quote region.

## Acceptance checks

| check | result |
|---|---|
| monotone on 400,001 logits in [−100,100] | pass |
| bounded in [0,1] (quote in [0,h]) | pass |
| all outputs and derivatives finite | pass |
| no negative quote | pass |
| no post-hoc clipping (only the ±1e6 logit guard) | pass |
| gate closed → exactly 0 | pass |
| gate open → covers the legal interval | pass; reaches 99% at ±2.5, 299 at 2.57 |
| high region gradient > sigmoid | pass; 9.66 vs 0.897 |
| no unreasonable derivative jump at the joint | pass; C1, derivative continuous at ±2.5 |
| numerical boundary test | pass; ±1e6 finite |
| analytic derivative vs central difference | pass; ≤1e-8 off the joint, ≤1e-5 at the joint |

The joint has a continuous first derivative but a jump in the second
derivative; a central difference there is first-order accurate, which is why
the tolerance is 1e-5 at exactly ±2.5 and 1e-8 elsewhere.

## Schema and checkpoint

- `ACTION_STORAGE_QUOTE_V3` (`storage_quote_v3`, version 3) has the same box
  and truthful point as v2, but the price-logit semantics differ.
- The checkpoint stores `specs.action_spec = {name: storage_quote_v3,
  version: 3, quote_map: linear_core_v1}` and schema
  `matd3-checkpoint-v3`.
- `DAGateTrainer.load` refuses a v2 manifest and a non-v3 action schema; the
  offline test confirms the refusal.
- No R2/R10 actor or critic is loaded; those checkpoints remain read-only
  baselines.

## Boundary

The map is implemented and audited; the pilot's result is reported separately.
No economic reward, bound or clearing rule was changed.

