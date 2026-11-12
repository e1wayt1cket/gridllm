# Phase 25 — protocol v3 formal training results

Run set: `FORMAL_1b47e0ed_seed{11,23,37}`, config `1b47e0ed`,
`protocol_version: 3`. Written 2026-11-08. Seed 11's launch and resume
narrative is in `phase25_seed11.md`; this report does not repeat it.

## Run inventory

| seed | run dir | terminal status | rounds | episodes | block steps | updates | best | final |
|---|---|---|---|---|---|---|---|---|
| 11 | `runs/FORMAL_1b47e0ed_seed11` | `completed_early_stop`, round 10 | 10 | 530 | 12,720 | 12,593 | round 1, -35.288 | `checkpoints/step_00012720` |
| 23 | `runs/FORMAL_1b47e0ed_seed23` | `completed_early_stop`, round 10 | 10 | 530 | 12,720 | 12,593 | round 4, -24.479 | `checkpoints/step_00012720` |
| 37 | `runs/FORMAL_1b47e0ed_seed37` | none (process ended mid-round) | 3 full + 11 episodes of round 4 | 170 | 4,080 | 3,953 | — | — |

All three manifests record the same `market_config_hash`, `b1_mechanism_hash`,
`action_schema: storage_quote_v3`, `quote_map: linear_core_v1`,
`code_revision`, `data_hashes` and `source_sha256`; `da_config` differs only in
`experiment_id`. Seed 11 and seed 23 each expose every training pair exactly 10
times (`exposure.json`); seed 37 three times. `run_manifest.exposures_per_day`
records the protocol cap (15); the realized exposure is 10/day for the two
completed runs, which is the floor, not the cap.

Seed 37 has no `finished_at`, so `scripts/finalize_run.ps1` refuses it
(`run_manifest.json` exists but the run has not terminated). Its artifacts are
evidence of a stopped run, not of a completed one; no `best` and no `final`
checkpoint are recorded. Its last `train.log` line is
`ep 170/795 r4 2022-11-03|2025-11-03`, and `checkpoints/` holds rounds 1–3.

## Protocol audit — seed 23

`scripts/finalize_run.ps1 -RunDir runs/FORMAL_1b47e0ed_seed23` wrote
`reports/cc01/audit/FORMAL_1b47e0ed_seed23.json`; overall verdict
`PASS_WITH_NOT_TESTED`. An independent read-only audit re-derived every item
below from the run's artifacts and returned `PASS_WITH_INFERRED`, the
`INFERRED` being the same TD-01 log-hash caveat.

| check | verdict | value |
|---|---|---|
| run terminated cleanly | PASS | `completed_early_stop` |
| min_rounds recorded / rounds | PASS | 10 recorded, 10 completed, 15 cap |
| early stop not before the floor | PASS | stopped at round 10, the floor |
| episodes / block steps | PASS | 530 of 530–795; 12,720 of 12,720–19,080 |
| optimizer updates | PASS | 12,593 |
| best and final checkpoints | PASS | distinct: `seed23_best/step_00005088` vs `seed23/step_00012720` |
| validation matches reported best | PASS | round 4, -24.479 |
| artifact hashes | PASS | `artifacts_sha256` mismatches empty |
| `train.log` hash | INFERRED | TD-01: hashed before the completion line (predates this protocol) |
| test not evaluated / legacy test not run | PASS | `test_evaluation=None`, `legacy_seen_test_run=False` |
| H2 not read | PASS | `h2_holdout_read=False` |
| voltage / line violations | PASS | 0 / 0 |
| LMP fallbacks | PASS | 0 |
| terminal SOC error | PASS | 0.0 |

Seed 11 carries the same verdict from `phase25_seed11.md`. Seed 37 cannot be
audited by the same path because it has no terminal status.

The live gate still matches at report time:

```
PYTHONPATH=src python src/run_formal_training.py --check
→ all_pass: true, budget_contract_ok: true, status: GO_FORMAL_TRAINING,
  config_sha256: 1b47e0ed0e5e469a510a77c6afb9b058aba3e41b90bd98734f40593645ffd2ff
```

The run manifests record the same `source_sha256` and `code_revision`, and all
12 source entries recorded by seed 23 match the live code manifest. The gate
also self-reports `code_sources_match: true`, so the sources on disk now are the
ones the manifest pins.

## Validation result

The metric is the mean over the 11 validation days of the fleet's daily payoff
difference against the truthful B1 baseline, in CNY/day; higher is better.
Negative means the trained fleet earns less than the truthful anchor.

| round | seed 11 | seed 23 | seed 37 |
|---|---|---|---|
| 1 | -35.3 | -46.2 | -67.8 |
| 2 | -61.4 | -34.4 | -96.9 |
| 3 | -77.0 | -48.4 | -113.3 |
| 4 | -93.7 | **-24.5** (best) | — (killed mid-round) |
| 5 | -80.9 | -68.1 | — |
| 6 | -70.4 | -86.1 | — |
| 7 | -66.9 | -101.3 | — |
| 8 | -101.9 | -86.2 | — |
| 9 | -93.7 | -84.9 | — |
| 10 | -95.0 | -98.9 | — |

No round of any seed has a positive fleet mean. The selected best round is the
earliest round for seed 11 (round 1) and the fourth for seed 23; both are far
above their final round. The same pattern appears in training itself, so it is
not a validation-only gap:

| | round 1 | round 10 |
|---|---|---|
| seed 11, training days (53/round) | -41.6 | -139.8 |
| seed 23, training days (53/round) | -44.2 | -131.0 |

Episode 1 of all three runs has a fleet delta indistinguishable from zero
(1.9e-4, 3.0e-4 and 5.0e-4 CNY): at initialisation the policy reproduces the
truthful baseline. The deviation, and the loss, appear only after optimizer
updates begin.

### Seed 23, best round (4), day by day

| validation pair | fleet delta (CNY) | note |
|---|---|---|
| 2022-07-22 \| 2025-07-15 | +144.2 | B1 = -164.1; the policy loses less |
| 2022-08-24 \| 2025-08-25 | +0.0004 | policy charge/discharge = 0 |
| 2022-08-25 \| 2025-08-26 | +0.0002 | policy charge/discharge = 0 |
| 2022-10-14 \| 2025-10-14 | +0.0004 | policy charge/discharge = 0 |
| 2022-10-16 \| 2025-10-11 | +0.0006 | policy charge/discharge = 0 |
| 2022-11-05 \| 2025-11-02 | -45.1 | |
| 2022-11-14 \| 2025-11-14 | -25.4 | |
| 2022-11-19 \| 2025-11-15 | -36.4 | |
| 2022-12-23 \| 2025-12-24 | -239.5 | |
| 2022-12-28 \| 2025-12-29 | +73.6 | |
| 2022-12-31 \| 2025-12-28 | -140.8 | |

On the four pairs whose |delta| is below 1e-3 CNY, the truthful baseline's own
daily fleet payoff is also below 1e-3 CNY (`G_b1_fleet` in `eval_history.json`)
and the policy's own `charge_mwh` / `discharge_mwh` are 0 on every unit: the
run's artifacts show no storage throughput on the policy side, so those days
carry no information about policy quality. `positive_days: 6` counts four of
them. The seven informative days sum to -269.3 CNY, an active-day mean of
-38.5 CNY/day, with two days better than B1 and five worse.

By unit at round 4: ESS9 -21.7, ESS14 -5.4, ESS17 -0.0, ESS27 +2.6 mean daily
delta. Only ESS27 beats its own baseline on average.

## Gate and quote behaviour

The evaluation path (`train=False`) is deterministic. In all ten seed 23
validation rounds `distinct_modes == 1` across 264 block decisions, and the same
is true for seed 11. Training samples the gate, so `gate_stats.jsonl` holds
235–236 distinct modes at episodes 300–530. The difference is explained by the
evaluation path taking the argmax: it never samples, so a gate vector that stays
on one side of 0.5 yields one fleet mode all day. It does not show that the
policy would reproduce the training-time mode diversity when deployed.

Two charge gates are closed on every validation day, not just on average:

- `ESS27.charge`: closed in all ten rounds (mean probability 0.006–0.455, never
  crossing 0.5);
- `ESS14.charge`: closed in nine of ten rounds, open only in round 4
  (probability 0.700).

A closed gate sets that leg's declared price to exactly 0
(`da_quote_map.forward_policy_v3`: `quotes = gate * price_quotes`); it does not
remove the leg from the clearing. The distinction is load-bearing: `ESS27`'s
charge leg is gated shut in all ten rounds, yet at round 4 the cleared schedule
still charges it 0.9–1.6 MWh on the seven active validation days (0 on the four
dust days). The gate is a price at zero, not a physical prohibition.

Every other leg is open in every round; round 4 also opens `ESS14.charge`, so
only seed 23's round 4 has seven open legs and the other nine rounds have six.

Quotes migrate to the box edges during training. Leg bounds are
`(charge 150, discharge 300)`. At round 10 seed 23's open quotes have
`ESS9.discharge` mean 299.1 (max 300.0) and charge quotes at 0.0;
`saturation_lower` counts 520 leg-decisions at the lower edge. Seed 11 round 10
is the same shape (`ESS27.discharge` mean 299.4, max 300.0, `ESS9.charge` mean
7.1, `saturation_lower` 778 of its 2,112 leg-decisions). This is the corner-collapse
diagnostic the smoke tooling reports, now visible in a formal run.

## What is and is not established

Established (`PASS`):

- both completed runs satisfy the frozen v3 budget and stop rule, with intact
  best/final checkpoints and matching artifact hashes;
- no voltage, line, LMP-fallback or terminal-SOC violation appears in any
  evaluation of either completed run;
- no test or holiday pair id appears in `episodes.csv`, `eval_history.json`,
  `gate_stats.jsonl` or `block_decisions.jsonl` of seed 23; both runs record
  `test_evaluation: null` and `legacy_seen_test_run: false`;
- on seeds 11 and 23, and on seed 37's three rounds, the trained fleet's daily
  payoff is below the truthful baseline's on every validation mean.

`NOT_TESTED`:

- test-block performance; no test evaluation exists and `pi_ai`,
  `pi_baseline`, `delta_pi`, `delta_pi_percent` and `p_delta_pi_positive` are
  not measurable from these runs;
- the cause of the degradation — critic drift, reward/payoff mismatch and the
  gate collapse are all consistent with the measurements and none is isolated
  by them;
- which interpreter ran seeds 23 and 37 (TD-10: the manifest records neither the
  interpreter nor package versions, and no launch log was kept);
- seed 37 beyond round 3.

`INFERRED`:

- the seed 23 and seed 11 `train.log` hashes, for TD-01;
- the H2 seal, which rests on each manifest's own `h2_holdout_read: false`; no
  artifact would record a read, so this cannot be demonstrated from the run
  directory alone.

`BLOCKED`: nothing in this report.

## For the next phase

1. Do not read "best = round 1 or 4" as early stopping working. The floor
   forced ten rounds; the best value was already below the floor's earliest
   measurement, and the proposal to add seeds should wait until the training
   objective's descent is explained.
2. Validation-day reporting should separate the four zero-B1 days from the
   seven informative ones, or at least not treat `positive_days` as a count of
   improvements. Changing the metric is a protocol change, not an edit.
3. The quote-edge migration and the two permanently closed charge gates are
   testable in an offline replay of the saved policies; the smoke checker
   already supports that path, so no new training run is needed to confirm it.
