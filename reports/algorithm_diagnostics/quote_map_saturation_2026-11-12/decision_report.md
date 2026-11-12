# Quote-map saturation audit and offline candidate comparison

label: NON_FORMAL_DIAGNOSTIC; map under audit: `linear_core_v1`; no version registered

admission status: **ADMITTED_Z10.0**

## Admission criteria

| Z | math | coverage | per ESS-leg | median dq/dz | p10 dq/dz | edge -pp | closed gate | physical | all pass |
|---|---|---|---|---|---|---|---|---|---|
| 2.5 | PASS | FAIL | FAIL | NOT_TESTED | NOT_TESTED | NOT_TESTED | PASS | PASS | **False** |
| 6.0 | PASS | FAIL | FAIL | PASS | PASS | PASS | PASS | PASS | **False** |
| 8.0 | PASS | PASS | FAIL | PASS | PASS | PASS | PASS | PASS | **False** |
| 10.0 | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | **True** |

## Readings of the coverage criterion

The pooled column is the admission reading. The rollout reads one checkpoint, so it is cross-checked against the buffer slice that checkpoint wrote (`best` = round 1, `final` = round 10).

| Z | checkpoint | buffer, matched round | buffer, pooled | deployment rollout |
|---|---|---|---|---|
| 2.5 | 11|best | 0.658 | 0.351 | 0.714 |
| 2.5 | 11|final | 0.214 | 0.351 | 0.203 |
| 2.5 | 23|best | 0.780 | 0.269 | 0.794 |
| 2.5 | 23|final | 0.039 | 0.269 | 0.039 |
| 6.0 | 11|best | 0.784 | 0.863 | 0.827 |
| 6.0 | 11|final | 0.778 | 0.863 | 0.744 |
| 6.0 | 23|best | 1.000 | 0.717 | 1.000 |
| 6.0 | 23|final | 0.547 | 0.717 | 0.563 |
| 8.0 | 11|best | 0.877 | 0.964 | 0.858 |
| 8.0 | 11|final | 0.958 | 0.964 | 0.954 |
| 8.0 | 23|best | 1.000 | 0.977 | 1.000 |
| 8.0 | 23|final | 0.873 | 0.977 | 0.875 |
| 10.0 | 11|best | 0.998 | 0.999 | 0.987 |
| 10.0 | 11|final | 1.000 | 0.999 | 0.999 |
| 10.0 | 23|best | 1.000 | 1.000 | 1.000 |
| 10.0 | 23|final | 1.000 | 1.000 | 0.999 |

## Criterion values

| Z | seed | min ESS-leg cov | med dq/dz ratio | p10 dq/dz ratio | edge -pp charge | edge -pp discharge |
|---|---|---|---|---|---|---|
| 2.5 | 11 | 0.001 | 1.0x | 1.0x | 0.000 | 0.000 |
| 2.5 | 23 | 0.000 | 1.0x | 1.0x | 0.000 | 0.000 |
| 6.0 | 11 | 0.199 | 804.9x | 236.0x | 56.433 | 45.317 |
| 6.0 | 23 | 0.169 | 2297.9x | 242.9x | 51.252 | 29.694 |
| 8.0 | 11 | 0.662 | 603.7x | 22281.6x | 63.606 | 59.493 |
| 8.0 | 23 | 0.884 | 1723.4x | 18026.8x | 63.245 | 63.814 |
| 10.0 | 11 | 0.989 | 482.9x | 17825.3x | 67.091 | 66.982 |
| 10.0 | 23 | 1.000 | 1378.7x | 14421.5x | 63.340 | 81.033 |

## Verdict vocabulary

- `PASS` — the named criterion's own evidence clears its threshold.
- `FAIL` — the evidence contradicts the threshold.
- `PARTIAL` — the buffer reading and the deployment rollout disagree on the same criterion; it is never reported as `PASS`.
- `NOT_TESTED` — the baseline row (v1 against itself) and the historical raw-logit layer, which was never recorded.
- `INCONCLUSIVE` — no candidate clears every criterion; no new candidate is added and no training is started.

## Not tested or blocked

- Historical per-row raw logits: `NOT_TESTED`. `block_decisions.jsonl` carries the post-noise price logits and the executed quotes, not the actor output; the buffer forward is a different reading and is labelled as such.
- Buffer samples cannot be clustered by day or episode: `NOT_TESTED` for that stratification. `ess_leg_summary.csv` clusters the historical noisy layer by training pair instead.
- Payoff, clearing outcome and settlement under any candidate: `NOT_TESTED`. Nothing here runs the market.
- The base B+C2 stack (`rl_b_c2_gate`, v2 sigmoid map): `NOT_TESTED` by scope, not by failure.

## What the evidence supports

- The mapping layer is reproduced exactly by the candidate module at `Z=2.5`; the remap error over every recorded decision row and every buffer sample is in `phase1_observations.json`.
- Per-seed core coverage, gradient retention and edge-quote fractions for each candidate on identical logits, in `phase3_candidates.json` and `candidate_ess_leg.csv`.
- Per-seed, per-ESS-leg coverage on the primary reading, in `candidate_comparison.json`.
- Admission: **ADMITTED_Z10.0**.

## What this does not show

- Offline quote statistics are not payoffs. A wider core lowers every core quote (`g(2.5)` falls from 0.99 to 0.65 at Z=8 and 0.62 at Z=10), so the candidates are a quote-level intervention, not a tail-only repair; the clearing and the settlement are not simulated here.
- The historical actor-loss inputs were never recorded and cannot be recovered: historical rows are `NOT_TESTED` for raw logits, and the buffer forward is the checkpoint policy on stored states, not the policy that filled the buffer.
- `best` is round 1 (episode 53), `final` is round 10 (episode 530). They are reported separately and never averaged.
- The base B+C2 stack still uses the v2 sigmoid map and is outside this audit.

## Files

`input_hashes.json`, `phase1_observations.json`, `execution_mapping_stats.csv`, `seed_summary.csv`, `ess_leg_summary.csv`, `mapping_candidates.json`, `mapping_unit_test_results.json`, `phase3_candidates.json`, `actor_update_gradient_stats.csv`, `candidate_comparison.json`, `probe_trace.csv`, `buffer_raw_logits_seed*.npz`, `output_hashes.json`.
