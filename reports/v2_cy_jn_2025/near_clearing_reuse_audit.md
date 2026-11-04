# NEAR clearing-reuse audit (V2-CY-JN-2025)

**Result: 80 of 80 NEAR pairs reuse the FULL96 JN-Base clearing.**

A NEAR row is a copy of a FULL96 row, so the question is whether every link holds: a unique source record, the same two dates, the same load and price digests, the same clearing configuration and support assumption, and a clear that is OPTIMAL and passes the physical checks. Reuse is registered only when all of them pass.

**Scope: JN-Base only.** The 21-day C80/C90/C100/NoESS probe is not a NEAR full-sample sensitivity result, is not audited here, and must not be described as one.

## Per-check pass count

| check | pass |
|---|---:|
| unique full record | 80/80 |
| dates match | 80/80 |
| load hash match | 80/80 |
| price hash match | 80/80 |
| result unique | 80/80 |
| arm is jn base | 80/80 |
| config match | 80/80 |
| optimal | 80/80 |
| solver optimal | 80/80 |
| voltage ok | 80/80 |
| line ok | 80/80 |
| p balance ok | 80/80 |
| q balance ok | 80/80 |
| terminal soc ok | 80/80 |
| exclusive ok | 80/80 |
| lmp ok | 80/80 |
| pcc ok | 80/80 |
| no error | 80/80 |
| reusable | 80/80 |

## Clearing configuration

| check | result |
|---|---|
| support is line50 q2 | PASS |
| impedance scale = 0.5 | PASS |
| reactive fraction is 2 | PASS |
| band = 0.93-1.05 | PASS |
| terminal soc pinned | PASS |
| storage rating matches | PASS |
| manifest pairing hash | PASS |
| manifest bus loads hash | PASS |
| manifest price curves hash | PASS |

Tolerances: active residual <= 1e-05 MW, reactive residual <= 1e-05 MVAr, terminal SOC <= 1e-09, charge/discharge product <= 1e-06, slack-LMP deviation <= 0.01 CNY/MWh. They are declared, not fitted.

## What this does and does not show

- It shows the FULL96 clear for each NEAR day used the same pair, the same curves, the same SOCP configuration, the same support assumption, and returned an OPTIMAL result that passes every physical check.
- It does not re-run the engine. Reuse rests on the inputs and settings being identical, not on a second solve producing the same numbers.
- It does not extend to the cost-sensitivity arms: their 21-day sample is not NEAR.

- Per-pair index: `data/processed/v2_cy_jn_2025/near/clearing_reuse.csv`
