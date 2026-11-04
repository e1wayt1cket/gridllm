# MATD3_DA_GATE_v2 single-seed pilot results

Run: `runs/DEV_DA_GATE_seed11_v2` (seed 11, 159 episodes, 3 rounds × 53
training days). Selected model: round 1 (best validation mean). Final model:
round 3. No H2 and no legacy test. Data: `da_gate_pilot_summary.json`,
`da_gate_material_calibration.{csv,json}`, `da_gate_winter_15min.{csv,json}`,
`runs/DEV_DA_GATE_seed11_v2/eval_history.json`.

## Economics

| model | val mean ΔG | val median | positive days | winter 12-13 | winter 12-30 | winter 12-31 |
|---|---:|---:|---:|---:|---:|---:|
| old R2 | −30.7 | −26.6 | 5/11 | −33.5 | −28.0 | −42.3 |
| old R10 | −77.7 | −4.6 | 4/11 | −224.5 | −409.6 | −234.8 |
| **new round 1 (selected)** | **−32.1** | **0.0** | **6/11** | **−53.1** | **−138.5** | **−108.3** |
| new round 2 | −44.3 | −11.2 | 5/11 | −153.1 | −196.7 | −142.2 |
| new round 3 (final) | −79.3 | −4.2 | 5/11 | −219.9 | −316.6 | −266.4 |

The selected model is about level with R2 on the mean, better on the median
and positive-day count, but **worse than R2 on all three winter days** and
better than R10. The later rounds drift back toward the R10 profile. Per-unit
mean ΔG for the selected model: ESS9 +2.4, ESS14 −27.1, ESS17 +3.2,
ESS27 −10.6; old R2 was −21.4, −5.9, +1.0, −4.4. ESS9 improved; ESS14 and
ESS27 are worse on average.

## Gate

| round | deterministic modes | permanent open/closed | ESS14 discharge p | ESS27 discharge p |
|---:|---:|---|---:|---:|
| 1 | 1 | 7 / 1 | 0.936 | 0.950 |
| 2 | 1 | 6 / 2 | 0.957 | 0.965 |
| 3 | 1 | 5 / 3 | 0.983 | 0.999 |

The deterministic policy is one joint mode in every round; the stochastic gate
visited 243 modes in the last training episode. The temperature warmup did not
prevent the deterministic collapse or the permanent gates. This is the same
qualitative failure as R1/R10, although the sampled gate is alive.

## Continuous quotes

| round | ESS14 discharge mean / p95 / max | ESS27 discharge mean / p95 / max |
|---:|---|---|
| 1 | 117.4 / 130.8 / 136.8 | 79.6 / 93.8 / 104.4 |
| 2 | 180.4 / 189.9 / 194.2 | 129.9 / 134.4 / 136.0 |
| 3 | 261.9 / 277.9 / 279.9 | 300.0 / 300.0 / 300.0 |

The selected model's quotes are far below the old R10 means (194/223). The
final model returns to the high end. The D1 map and D3 exploration did not
stop the drift with more training.

The final replay has 705 ESS14 and 252 ESS27 discharge transitions at ≥0.86h
(18.5% / 6.6%), versus 2,307 / 1,276 (18.1% / 10.0%) in the old R10. The
exploration added high quotes at ESS14 but the shorter run and the actor drift
leave ESS27's share lower.

## Winter 15-minute acceptance (selected model)

Worst block per winter day/unit under the new policy; q=0 is the same-state
truthful counterfactual.

| day | unit | block | actor quote | q=0 payoff | B1 block | actor payoff | actor accepted MWh | quote-caused? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 12-13 | ESS14 | 3 | 123.6 | 71.56 | 71.55 | 0.00 | 0.0 | yes |
| 12-13 | ESS27 | 12 | 81.2 | 88.19 | 88.21 | 0.00 | 0.0 | yes |
| 12-30 | ESS14 | 10 | 136.8 | 111.11 | 111.11 | 0.00 | 0.0 | yes |
| 12-30 | ESS27 | 23 | 82.4 | −52.82 | −24.27 | −52.82 | 0.0 | no; path/charge |
| 12-31 | ESS14 | 17 | 135.9 | 98.12 | 98.11 | 0.00 | 0.0 | yes |
| 12-31 | ESS27 | 0 | 94.1 | 53.44 | 53.41 | 0.00 | 0.0 | yes |

In five of six cases the actor's mid quote (81–137) yields zero accepted
discharge and zero payoff, while q=0 reproduces the B1 block. The problem is
not confined to 258–300; any quote above the fleet's truthful offer can be
displaced. The 12-30 b23 ESS27 case is a low-price/path case where the quote
is immaterial.

Some local block losses are partly recovered later in the day: at 12-13 b3
ESS14 the block payoff difference is 71.6, but the full-day ΔG spread over the
quote grid is only 4.1. At 12-30 b10 ESS14 and 12-31 b17 ESS14 the full-day
spreads are 138.9 and 108.7, so those are material both locally and for the
day.

## Critic

The material-state calibration used 13 states (the phase-15 blocks plus the
winter-loss blocks), 52 groups, 15 material:

| statistic | pilot selected |
|---|---:|
| mean Spearman(Q, ΔG) | +0.109 |
| median Spearman | +0.200 |
| material mean Spearman | **+0.128** |
| material groups negative | 5 / 15 |
| twin gap mean / median / p95 | 0.0265 / 0.0206 / 0.0739 |

The ranking is better than the old R10 (mean ≈ 0, 7/10 material negative), and
the twin gap is smaller. But the Q separation is still small in the blocking
states: at 12-30 b10 ESS14 the Q range across the grid is 0.047 while the
full-day ΔG spread is 138.9 CNY. The actor's quote is at the worst end and the
critic barely distinguishes it from q=0. The critic ranking is a partial
improvement, not a value signal strong enough to move the actor.

## Stop-rule assessment

1. Deterministic modes ≤2 and sampled modes not recovered: deterministic = 1,
   but sampled = 243, so the second half is not met.
2. Material mean Spearman negative or at least half anti-ranked: material mean
   +0.128 and 5/15 negative, so this is not met.
3. Winter high-value windows still have quote-caused zero acceptance and a
   same-state alternative with a clearly higher payoff: **met** at 12-30 b10
   ESS14 (full-day ΔG spread 138.9), 12-31 b17 ESS14 (108.7) and 12-31 b0
   ESS27 (69.3); 12-13 b12 ESS27 is material at 34.2. The 12-13 b3 block is a
   local 71.6 loss but only a 4.1 full-day spread, so it is not counted as
   material for the day.
4. Physical/SOC/clearing failure: none (24/24 OPTIMAL, terminal SOC 0.5, zero
   voltage/line cells).
5. NaN/Inf or checkpoint restore failure: none.

**Stop condition 3 is met.** Do not expand the training budget. The next step
is a diagnosis of why the actor remains at a mid quote when the same-state
truthful quote is materially better, not a larger run.

## Answer

The 159-episode pilot is **not** a green light to scale. It shows:

- D1/D2/D3/A1 are mechanically integrated and the critic ranking improved on
  the evaluated material set;
- the selected round-1 policy is roughly R2-level on the mean and worse on the
  winter days, and the later rounds drift back to the high-quote regime;
- the actor still chooses mid quotes that lose high-value blocks, while the
  critic is too flat to separate them strongly.

The next work should target the actor/gate decision: the gate is permanently
open and the actor has no path to the truthful/near-zero quote that the critic
weakly prefers. That is a different problem from the one D+A attempted.

## Boundary

H2 unread; LEGACY_SEEN_TEST not run; R2/R10 and the original 10-round run are
preserved; no old checkpoint was overwritten. The pilot is exploratory and
does not support a generalisation claim.
