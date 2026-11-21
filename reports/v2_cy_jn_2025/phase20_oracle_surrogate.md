# Phase 20-B: critic gradient versus an offline value oracle

Read-only. No training, no checkpoint write, no replay addition, no H2 and no
legacy test. Data: `phase20_oracle_surrogate.{json,csv}`. Code:
`src/audit_phase20_oracle_surrogate.py`.

## What the oracle is

For each `(state, unit, leg)` the script interpolates the matched full-day
payoff linearly over the observed quote grid. Outside the grid the value is
constant. The critic gradient is the stored `dQ/dquote` at the actor quote
multiplied by the v3 map derivative; the oracle gradient is the local slope of
the interpolated payoff. The virtual update moves only the target leg's price
logit and recomputes the quote through `forward_policy_v3`. No network
parameter is changed; this is a logit-space first-order probe.

The input grids are not uniform:

| source | legs | unique quotes per group |
|---|---|---:|
| `phase19_critic_training_calibration_rows.csv` | charge, discharge | 7–8 |
| `daily_counterfactual_action_validation.csv` | discharge only | 5–6 |

The validation cases therefore do **not** have a seven-point grid. The report
does not present them as such.

## Data sufficiency

| checkpoint | groups | informative (spread ≥ 1 CNY) |
|---|---:|---:|
| D+A best | 32 | 15 |
| D+A final | 32 | 15 |

The threshold is met, so the pre-registered supplement was not run. Seventeen
of the 24 training-state curves in the Phase 19 calibration are economically
flat and are excluded from the ranking evidence.

## Ranking against the matched return

| metric | D+A best | D+A final |
|---|---:|---:|
| informative groups | 15 | 15 |
| critic direction matches oracle direction | 6 / 15 | 6 / 15 |
| Q-top equals return-top | 6 / 15 | **1 / 15** |
| median within-group Spearman(Q, return) | +0.238 | **−0.143** |
| median best-minus-actor return (CNY-equiv.) | 5.5 | 18.0 |

The same check against the full-day ΔG is identical in ordering and gives a
median best-minus-actor gain of 6.0 CNY (best) and 20.3 CNY (final), with the
same direction-match counts.

## Named states

| checkpoint | state | actor quote | grid best | critic slope (CNY-eq./CNY) | oracle slope | direction |
|---|---|---:|---:|---:|---:|---|
| best | 12-30 b10 ESS14 | 136.8 | 0.0 | −0.08 | −0.99 | match |
| best | 12-13 b12 ESS27 | 81.2 | 0.0 | −1.53 | 0.00 | no |
| best | 12-30 b11 ESS14 | 136.2 | 0.0 | +0.10 | −0.01 | **reverse** |
| best | 12-13 b3 ESS14 | 123.6 | 0.0 | −0.03 | 0.00 | no |
| best | 12-31 b17 ESS14 | 135.9 | 0.0 | **+0.88** | −0.84 | **reverse** |
| final | 12-30 b10 ESS14 | 278.7 | 0.0 | −0.19 | 0.00 | no |
| final | 12-13 b12 ESS27 | 300.0 | 0.0 | +497.7 | −0.00 | **reverse** |
| final | 12-31 b17 ESS14 | 277.9 | 0.0 | −0.06 | 0.00 | no |

The critic's local slope often points the wrong way, and at the flat segments
the oracle slope is zero. The first-order step therefore does not move the
quote toward the better basin:

- at 12-30 b10 ESS14 (best), a logit step of 1e-2 moves the discharge quote by
  less than 0.1 CNY/MWh against a 136.8 CNY/MWh gap;
- at 12-30 b10 ESS27 (final), the critic slope is large and points down, but a
  1e-2 logit step moves the quote from 300 to about 267, while the grid best is
  5; the oracle slope is effectively zero, so the oracle step does not move;
- at 12-13 b12 ESS27 (final), the critic slope is positive while the best quote
  is 0.

## Answer

The offline oracle does not support the claim that a first-order update can
reach the better quote region. The critic's local direction is wrong in 9 of 15
informative groups for both checkpoints; the final checkpoint's top-quote
agreement is 1/15. The material gains are non-local relative to the observed
slopes, so a longer run of the same local step is not the repair. This
corroborates the Phase 19 ranking finding; it does not establish a new
deterministic code error.

## Boundary

The oracle is a piecewise-linear proxy, not the environment gradient. The
validation cases are offline diagnosis only. No training, checkpoint write, H2
read or legacy test run occurred.
