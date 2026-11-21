# Local day-ahead v7 smoke: three training days

Evidence for the Task 8 smoke. Days are the first three dates of
`local_da_days.train_pairs(...)` in manifest order: `2021-06-04`,
`2021-06-13`, `2021-06-17`. Every day was walked with the truthful storage
quote `(0.0, 0.4, 0.0, 0.4)`; the shaded arm is a single bounded addition on
`2021-06-04`.

The recorded probe is the environment's own block-11 probe, taken during the
walk. A second `probe_block` call is timed for the latency line and its own
result is not evidence.

## Acceptance

| Item | 2021-06-04 | 2021-06-13 | 2021-06-17 | Verdict |
| --- | --- | --- | --- | --- |
| `rewards_finite` | true | true | true | PASS |
| `no_none_clears` | 24 clears, 0 missing | 24 clears, 0 missing | 24 clears, 0 missing | PASS |
| `soc_in_band` | 0.100000-0.596831 | 0.100000-0.899999 | 0.100000-0.639436 | PASS |
| `one_inventory_term` | 1 | 1 | 1 | PASS |
| `declared_point_deviation` | 0.0 | 0.0 | 0.0 | PASS |

Band is `0.1 <= SOC <= 0.9`, the storage spec `rl_da_env.storage_agents`
declares. `summarize(record)` is the source of each row; the values are read
from `smoke.json`. The `soc_min`/`soc_max` values in `smoke.json` are the
post-clamp carried states (`env.prev_soc`) as measured at run time; the harness
now records `info["soc_raw"]`, the raw energy-derived SOC before the clamp, so
`soc_in_band` can fail on a solve outside the band. That the audited values
above are unchanged by the switch is INFERRED from `smoke.json`, not
re-measured: `smoke.json` was produced before the switch and was not
regenerated, no truthful episode's recorded extreme sits exactly on a band
edge, so no clamp fired on those walks. The shaded arm did clamp at 0.1 and a
rerun would record a raw value below the band for it; that arm is not
summarized for acceptance, so no flag above depends on the inference.

## Latency

Mean over the three truthful days.

| Measure | Seconds |
| --- | --- |
| Exact solve, 20-period window (`exact`) | 0.822 |
| Conditional price solve (`price`) | 0.091 |
| Baseline re-clear, exact + price (`baseline`) | 1.128 |
| Probe (`probe_day`) | 2.182 |
| Episode, 24 blocks including the block-11 probe (`episode`) | 47.778 |
| Block step, mean over 72 blocks (`block_seconds`) | 1.991 |

Per-day episode seconds: 49.524, 48.559, 45.251.

## Per-seed projection at both windows

Frozen block-step bounds per seed under v7: 21,600 (min) and 32,400 (max),
i.e. 900-1350 episodes of 24 blocks over the 90-day training block. (An
earlier version of this table carried v6's 12,720 / 19,080 bounds, which are
the 53-day block's counts and understate a v7 seed.)

| Window | Seconds per block | Projected hours per seed |
| --- | --- | --- |
| `ROLL_HORIZON=20` | 1.991 | 11.95 - 17.92 |
| 12-period fallback (8 lookahead) | 1.328 | 7.97 - 11.95 |

The projection is the typical-block rate. A block that reaches the 300 s
solver limit spends all of it, so a day whose blocks all bind would cost two
hours and no average bounds it.

No wall-clock gate exists in `training_contract.py`. Its budget keys are
rounds, exposures, episodes and block steps (`minimum_rounds`,
`exposures_per_day_min/max`, `episodes_per_seed_min/max`,
`block_steps_per_seed_min/max`, `seeds`); there is no latency criterion for the
brief's fallback to trigger on. Both projections are recorded; choosing
between the two windows is the reader's decision.

## Probe outcomes

| Day | Block | `price_qualification` | `qualified_for_settlement` | `failure` |
| --- | --- | --- | --- | --- |
| 2021-06-04 | 11 | `NONSMOOTH_OR_FAILED` | false | `PROBE_FAILED` |
| 2021-06-13 | 11 | `NONSMOOTH_OR_FAILED` | false | `PROBE_FAILED` |
| 2021-06-17 | 11 | `SMOOTH_PASS` | true | null |

The failed probes are recorded as evidence. They do not abort a day and are not
retried; the acceptance flags above are about the clearing walk, not about
settlement qualification.

## Shaded arm

One day, one deterministic near-truthful quote with both prices shaded above
the truthful point and full quantity available: `(10.0, 0.4, 20.0, 0.4)`,
box-normalized squared deviation 0.008889.

| Day | Blocks cleared | Aborts | Episode seconds |
| --- | --- | --- | --- |
| 2021-06-04 | 24 of 24 | 0 | 49.117 |

This arm is near-truthful -- both prices sit at 10/150 and 20/300 of their box
with quantities at full -- so its zero abort count does not bound an untrained
policy's abort rate; an aggressive quote can still hit the 120 s limit.

Before the SOC carry was clamped, this same quote aborted `2021-06-04` at block
22: `ValueError: Invalid independent storage device`, carried SOC
`0.0999999996268` on ESS14. The clamp in `rl_da_env.carried_soc` now stores the
band edge, and the arm completes with no abort. The truthful arm's abort count
is zero on all three days.

Provenance:
- The pre-fix block-22 abort is quoted from the fixing report's traceback; it
  is not an artifact in this evidence directory.
- The 2021-06-04 whole-day-clear corroboration is a coordinator ruling, not a
  measurement stored in this evidence directory.

## Commands run

```bash
# Red: ModuleNotFoundError: No module named 'smoke_local_da_v7'
PYTHONPATH=src <py> -m pytest tests/test_smoke_local_da_v7.py -v

# Green
PYTHONPATH=src <py> -m pytest tests/test_smoke_local_da_v7.py -v   # 2 passed
PYTHONPATH=src <py> -m pytest tests/test_local_da_env_v7.py -q     # 9 passed

# Per-day evidence (day 1 also runs the shaded and fallback arms)
PYTHONPATH=src <py> -B src/smoke_local_da_v7.py --days 2021-06-04 --out tmp/smoke_da_v7/2021-06-04
PYTHONPATH=src <py> -B src/smoke_local_da_v7.py --days 2021-06-13 --no-side-arms --out tmp/smoke_da_v7/2021-06-13
PYTHONPATH=src <py> -B src/smoke_local_da_v7.py --days 2021-06-17 --no-side-arms --out tmp/smoke_da_v7/2021-06-17

# Merge into the named evidence file
PYTHONPATH=src <py> -B src/smoke_local_da_v7.py --merge \
  tmp/smoke_da_v7/2021-06-04/smoke.json \
  tmp/smoke_da_v7/2021-06-13/smoke.json \
  tmp/smoke_da_v7/2021-06-17/smoke.json \
  --out reports/local_da_v7_2026-11-18
```

`<py>` is `/c/Users/26036/.conda/envs/energy_env/python.exe`.

## The deviation weight is re-derived for this box

The v5 quote is four entries wide per unit, so the parent's 0.10 was selected
for a different regularizer. `calibrate_dev_penalty_v7.calibrate(seed=123)`
applies the parent's own selection rule -- the same `LAMBDA_GRID`, the same
`PROBE_STEPS`, `calibrate_dev_penalty.select` -- to the v7 trainer's anchor on
one fixed day with no learning. It selects **0.05**, at which the anchor's
gradient norm is 0.043 of the critic's.

| `lambda` | anchor / critic gradient | corner fraction |
| --- | --- | --- |
| 0.0 | 0.0000 | 0.812 |
| 0.01 | 0.0086 | 0.875 |
| 0.05 | 0.0429 | 0.875 |
| 0.1 | 0.0858 | 0.812 |
| 0.25 | 0.2145 | 0.812 |
| 0.5 | 0.4290 | 0.938 |
| 1.0 | 0.8580 | 1.000 |

`DAV7ExperimentConfig.anchor_weight` now defaults to 0.05, so a launch that
omits the field still runs the selected value. The smoke evidence above was
produced at that weight: the three days were re-walked and re-merged, and the
acceptance flags and SOC extremes are unchanged by it. The corner fraction is
high at every weight, including zero, so it is a property of the anchor's
measurement point rather than of the regularizer; the selection rule reads
deviation and spread, not the fraction.

## Acceptance 4 to 8

| Item | Verdict | Value |
| --- | --- | --- |
| 4 training-vs-whole-day prices | PASS | sign agreement 1.000 on all three days, median relative gap 0.028-0.048, criterion `>= 0.95` and `<= 0.10` |
| 5 block-vs-day gap | recorded, no criterion | median 0.163-0.208 per bus; p95 0.303-9.017 |
| 6 deviation weight | PASS | `gradient_share` 0.0429 at weight 0.05, criterion `<= 0.50` |
| 7 smoke flags | PASS | three days, every flag true |
| 8 feasibility selection | recorded | train 90 selected 90, validation 18, test 18, holiday 10, no day dropped |

Day construction, recorded in `summary.json`:

- whole day, 96 periods, `terminal_soc_equal` true, degradation 80 CNY/MWh per
  unit;
- inverter support from `local_da_operating.support_day` with the 1.10
  nameplate multiplier (`Q_ESS9/14/17/27` at 0.44 MVAr, PV at 0.33);
- `EXTERNAL33` at bus 32 with 2.0 MVAr.

This is not the training day. The training window carries the declared
condensers `SC17/31/32` and neither the `Q_` inverter units nor an external
device, so the audit path has strictly more reactive room than the path it
audits. The gap above is measured across that difference, not within one
construction.

| Day | Cells above the floor | Sign agreement | Median relative gap |
| --- | --- | --- | --- |
| 2021-06-04 | 384 | 1.000 | 0.0278 |
| 2021-06-13 | 384 | 1.000 | 0.0476 |
| 2021-06-17 | 384 | 1.000 | 0.0381 |

Acceptance 5, block price against the day's own price for the committed
period (`median / p95`):

| Day | Buses |
| --- | --- |
| 2021-06-04 | 14: 0.165 / 0.306 | 17: 0.169 / 0.303 | 27: 0.175 / 0.335 | 9: 0.163 / 0.310 |
| 2021-06-13 | 14: 0.186 / 8.086 | 17: 0.187 / 7.943 | 27: 0.163 / 9.017 | 9: 0.171 / 8.504 |
| 2021-06-17 | 14: 0.202 / 0.613 | 17: 0.208 / 0.596 | 27: 0.178 / 0.722 | 9: 0.196 / 0.657 |

The 8.6-9.0 p95 values on `2021-06-13` are a ratio against a near-zero price,
not a large absolute gap; the same day's median is 0.16-0.19. `rolling_gap`
applies no price floor and is reported as measured.

Acceptance 8 reads `local_da_audit.SCREEN`, which now points at
`qualification/screen/`, the per-day records that
`qualification/build_screen.py` rewrote from the full-library measurement.
The screen `train_pairs` used before it predates the declared reactive asset
and dropped `2021-12-24`, `2022-10-11` and `2022-10-14` as exact-infeasible;
all three clear on the current default, so the v7 train block keeps its full
90 days.

```bash
# Per-day screen, from the full-library measurement
python -B reports/local_da_v7_2026-11-18/qualification/build_screen.py   # 136 records

# Price stack, anchor weight and smoke flags
PYTHONPATH=src python -B tools/audit_local_da_v7.py \
  --smoke reports/local_da_v7_2026-11-18/smoke.json \
  --out reports/local_da_v7_2026-11-18/audit          # acceptance_4/6/7 PASS
PYTHONPATH=src python -m pytest tests/test_local_da_audit_v7.py -q   # 5 passed
PYTHONPATH=src python -m pytest tests/test_local_da_v7_days.py -q    # 5 passed
```

## Training-window gap tolerance

The exact model demands a 1e-6 relative and 0.001 CNY absolute gap. Some
shaded-quote windows in the training loop cannot certify that inside the 120 s
limit while holding a feasible incumbent, and the launcher read those as
infeasible and aborted the day. The reproduction was `2021-06-13` block 0 at
the probe policy's quote: `solver_status 9`, three solutions, `MIPGap`
5.47e-3 against an objective bound of -230.44 CNY; the same window cleared
optimally at `seconds=600`, so the failure was a time limit and not a
physically impossible window.

`rl_da_env.GAP_TOLERANCE` (`0.01`) accepts such a dispatch when its own
reported relative gap is inside the bound. The accepted dispatch is cleared
and priced as usual, the conditional objective is compared to the exact one
on the same bound, and the block is counted in `env.gap_solved_blocks`, which
every episode row carries as `gap_solved_blocks`. A solve with no incumbent
stays `CLEAR_INFEASIBLE`; a solve with an incumbent outside the bound stays
`CLEAR_UNPROVEN`. `local_da_clearing.clear_day`, the settlement ledger and
every audit pass `0.0` and keep demanding the certified optimum, so no
settlement payment and no audit price is read off an uncertified incumbent.

The tolerance is declared in `formal_training.training_gap_tolerance` of the
v7 config and bounded in `claims.gap_accepted_payoffs`. The acceptance is
tested on a stub solve (`test_local_da_env_v7.py`) and fired on the
reproduction above.

### The window limit is raised to 300 s

The tolerance alone did not carry a training episode. On the exploration
policy two episodes of the launcher smoke (`runs/SMOKE_local_da_v7_launcher3`,
120 s windows) aborted:

| Episode | Day | Block | Failure | `gap_solved_blocks` |
| --- | --- | --- | --- | --- |
| 1 | `2021-06-13` | 5 | `CLEAR_UNPROVEN` (incumbent held, gap above 1 % at 120 s) | 1 |
| 2 | `2021-06-04` | 1 | `CLEAR_INFEASIBLE` (no incumbent found in 120 s) | 1 |

`2021-06-04` clears in seconds at the truthful quote, so the second failure is
the exploration quote meeting the limit, not a physically impossible day. The
frozen config now declares `training_window_seconds: 300`, `train_local_da_v7`
defaults to it, and `run_formal_training.command` passes it, so a launch
cannot silently use another value.

The limit is a search budget, not a criterion: a gap-solved block still has to
report its own gap inside 1 %, and a block with no incumbent is still
infeasible. The cost is stated rather than hidden: a block that reaches the
limit spends all of it, so a day whose 24 blocks all bind costs 2 hours while
the declared 12-18 h per-seed projection assumes the typical ~2 s block. No
end-to-end episode has yet completed under the 300 s limit; the launcher smoke
was killed for low system memory after those two aborts and has not been
re-run.
