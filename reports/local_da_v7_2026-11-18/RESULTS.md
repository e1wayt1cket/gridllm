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
| Exact solve, 20-period window (`exact`) | 0.795 |
| Conditional price solve (`price`) | 0.088 |
| Baseline re-clear, exact + price (`baseline`) | 1.101 |
| Probe (`probe_day`) | 2.079 |
| Episode, 24 blocks including the block-11 probe (`episode`) | 46.872 |
| Block step, mean over 72 blocks (`block_seconds`) | 1.953 |

Per-day episode seconds: 50.705, 46.347, 43.564.

## Per-seed projection at both windows

Frozen block-step bounds per seed: 12,720 (min) and 19,080 (max).

| Window | Seconds per block | Projected hours per seed |
| --- | --- | --- |
| `ROLL_HORIZON=20` | 1.953 | 6.90 - 10.35 |
| 12-period fallback (8 lookahead) | 1.327 | 4.69 - 7.04 |

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
| 2021-06-04 | 24 of 24 | 0 | 47.785 |

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
