# Price qualification re-measurement

The old screen reported `qualified_for_settlement` 0 of 136 days, in two
different classes: 108 days whose direction-conditional price solve returned
suboptimal, and 24 days whose price solve was optimal but whose probes all
failed. Both counts predate the market's declared 2.5 MVAr reactive asset
(`docs/project_context/10_DECISION_LOG.md` D-011), so this sample re-measures
them on the current default.

## What was measured

- Sample: 16 days, declared in `rule.json` before the run — six of each of the
  two old failure classes, taken at even spacing over the date-sorted class,
  plus all four days whose exact dispatch was suboptimal.
- `measure_qualification.py` calls `local_da_clearing.clear_day(day,
  price_probes=True)` on `run_local_da.build_day`'s day and records the dispatch
  and pricing solver status, the conditional objective difference, every probe,
  and the qualification label.
- No criterion was widened. The three-state label is the one
  `da_market_pricing.qualify_prices` has always emitted; the old copy in
  `local_da_clearing` could not produce its middle state.

## Result

| Label | Before the three-state rule | After |
| --- | --- | --- |
| `SMOOTH_PASS` | 6 | 6 |
| `SUBGRADIENT_PASS` | 0 (rule absent) | 7 |
| `NONSMOOTH_OR_FAILED` | 10 | 3 |
| `qualified_for_settlement` | 5 | 12 |

Evidence: `class_a.json`, `class_b.json`, `class_c.json` (before) and
`after.json` (after), each `sample[i]` carrying its own fields.

### The suboptimal price solves are gone on the sample

`pricing_status` is 2 on all 16 days in every run, including all six class-A
days, which the old screen recorded as suboptimal. The exact dispatch is
optimal on all four class-C days, which were unservable before the reactive
asset was declared. Measured, not inferred: `after.json` field
`pricing_status` and `dispatch_status`.

### The three-state rule recovers seven days

`probe_prices` had always computed `subgradient_bracket`; `clear_day` wrote
only `SMOOTH_PASS` or `NONSMOOTH_OR_FAILED`, so a bracketed corner could not be
labelled as one. Seven of the sixteen days are bracketed on every probe and not
smooth on at least one. They are now `SUBGRADIENT_PASS`, and
`qualified_for_settlement` accepts that label where it already accepted it in
`da_market_pricing.py`. `QualificationLabelTests` in
`tests/test_local_da_market.py` pins the three labels and the settle decision.

## What still fails, and what it is

### Three days: the probe cannot resolve its own perturbation

| Day | Probe that fails | Dual | Left | Right | Spread |
| --- | --- | --- | --- | --- | --- |
| 2021-09-02 | bus 0, period 61, epsilon 1e-5 | 420.0 | 429.82 | 415.94 | 13.9 |
| 2022-08-05 | bus 0, period 56, epsilon 1e-5 | 130.0 | 142.26 | 117.75 | 24.5 |
| 2022-08-14 | bus 0, period 53, epsilon 1e-5 | 80.0 | 81.08 | 78.92 | 2.2 |

All three fail exactly one probe, always the slack bus at the smaller
perturbation, and the same probe passes at 1e-4. A 1e-5 MW demand step at a
420 CNY/MWh dual moves the objective by 420 * 1e-5 * 0.25 = 1.05e-3 CNY, while
the pricing model's `BarQCPConvTol` is 1e-8 on an objective of about 1.8e4 CNY,
i.e. a 1.8e-4 CNY quantization. The two are the same order, so the difference
quotient at 1e-5 divides solver rounding by 2.5e-6.

This is `NOT_TESTED`: whether the market is nonsmooth at those points cannot be
answered at 1e-5 with this solver setting, and the sample does not answer it.

### One day: the pricing cone is not tight enough for its own audit

`2021-12-24` passes every probe but `physical_audit` fails on
`cone_tightness_max_mw2 = 4.63e-6` against a 1e-6 threshold, on a pricing model
whose exact dispatch is tight to 2.7e-7.

Tightening `BarQCPConvTol` to 1e-9 does bring that day's cone to 5.36e-7 with
status 2 and an objective difference of 1.15e-5 CNY, but across the sample it
makes the probes worse: eleven of the thirteen days that are smooth or
bracketed at 1e-8 lose that property at 1e-9, and `2021-10-23` and `2022-10-11`
return status 13. The comment above `ClearingModel.BarQCPConvTol` already records that
1e-10 returns suboptimal prices. Recorded in `tolerance_probe.json`; the
tolerance is not changed on this evidence.

## The first library batch, and the batch that did not finish

`library_a.json` covers the 28 earliest library days: `SMOOTH_PASS` 8,
`SUBGRADIENT_PASS` 7, `NONSMOOTH_OR_FAILED` 13, `qualified_for_settlement` 14.
Every pricing solve is status 2. `2021-09-11` carries a passing label and a
failing settlement, and this record kept no audit breakdown for it, so which
audit refused it is `NOT_TESTED`.

The label is not reproducible day for day. `2021-08-14` is
`NONSMOOTH_OR_FAILED` in `after.json` and `SMOOTH_PASS` in `library_a.json`, on
the same day, the same default and the same code. A single run's label is
therefore not evidence that a day is qualified; only the probe records are.

`BLOCKED`: batches two to five, 108 days. The first attempt raised `KeyError:
'clearing_objective'` on a day whose exact dispatch did not solve (the recorder
now tolerates that and records the failure string); the second was stopped by
the system for low memory. Neither is restarted here, so the 136-day counts are
`NOT_TESTED`.

## Not covered

- The 136-day counts are `NOT_TESTED` for the reason above.
- The sample is 16 of 136 days and is not a rate estimate for the library.
- `SMOOTH_PASS` and `SUBGRADIENT_PASS` counts are reported separately and are
  not added into a single "qualified" figure.

## Commands

```bash
PYTHONPATH=src python -B reports/local_da_v7_2026-11-18/qualification/measure_qualification.py \
  --dates 2021-06-04,2021-09-02,2021-10-23,2021-12-22,2022-08-05,2022-11-04 \
  --out reports/local_da_v7_2026-11-18/qualification/class_a.json
PYTHONPATH=src python -m pytest tests/test_local_da_market.py -q   # 26 passed
```
