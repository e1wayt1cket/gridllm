# Phase 21 decision: candidate 001 accepted, no further candidate opened

Read-only over two finished smoke runs. No test, holiday or H2 evaluation. The
smoke is a controlled diagnostic, not a generalisation result. Evidence:
`phase21_smoke_control_vs_candidate_001.json`, `phase21_smoke_protocol.json`,
`p21_candidates/P21_candidate_001/`.

## Pre-registered smoke

Common split, five fixed training days:
2022-08-14, 2022-10-13, 2022-11-01, 2022-12-17, 2022-12-27; seed 11; four
rounds = 20 episodes = 480 block interactions; four validation passes. The
control is the same code and protocol before the actor-loss map fix.

## Comparison

| run | best round | validation mean fleet ΔG | positive days |
|---|---:|---:|---:|
| control (pre-fix) | 2 | −25.0 | 6 / 11 |
| candidate 001 | 4 | **−18.5** | **7 / 11** |

Paired over the same 11 validation days at each run's best round: mean
difference +6.5 CNY/day, day-level SD 34.0. The pre-registered non-inferiority
threshold (control mean − 1 SD) is satisfied. Per-unit change versus control:
ESS9 +13.9, ESS14 +3.2, ESS17 −7.9, ESS27 −2.7 CNY; no unit is worse by more
than 10 CNY. Positive days are not lower by more than one.

## Hard gates

Both runs: exactly 20 episodes, five days × four exposures, 1,920 block rows,
four validation passes with 11 days each, common split, no test/H2, terminal
SOC 0.5, 0 voltage cells, 0 line cells, 0 LMP fallbacks, settlement identity
closed. The actor-routing audit in the candidate code reports
`v2_calls_in_update = 0`, `update == execution` and a finite actor loss.

## Action behaviour

Quote medians are comparable to the control. High-quote share (≥ 95% of the
box) is 0.6–1.9% in the candidate and 0.6–1.7% in the control; the zero-quote
share is 41–53%. There is no new corner collapse. ESS17's discharge median
moves from 1.5 to 18.3 CNY/MWh, a small shift within the low band; ESS14's
charge median moves from 17.5 to 23.2.

## Why no candidate 002 or 003

Phase 20 found no evidence-supported second change:

- target smoothing changes the target by 4.2 CNY (best) / 0.6 CNY (final) at
  the median, far below the twin gap; candidate 002 is not opened;
- the ESS14 own-action collapse does not reproduce in eval mode on nine states;
  candidate 003 is not opened;
- no network-structure change is supported.

## Result

`P21_candidate_001` is accepted as the code version for formal-training
readiness. The candidate directory records the patch, code/config hashes, test
results, smoke result and checkpoint pointers. No earlier candidate exists.

## Boundary

The smoke is not a formal result and its validation numbers are not a
generalisation claim. No historical checkpoint, report or data file was
modified. The critic-ranking residual risk recorded in Phase 20 remains.
