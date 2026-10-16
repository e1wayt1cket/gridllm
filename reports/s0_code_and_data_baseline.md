# S0: code and data baseline

What the repository held before any change in this round. Every figure here is
either read from a file or produced by a command whose output is quoted; nothing
is inferred from naming and nothing is copied from an earlier report.

## Git

| | |
|---|---|
| `git rev-parse HEAD` | `9f61c9aa02f73398ff3495e23f465e6de4298db3` |
| branch | `master` |
| remotes | `gridllm`, `origin` — **nothing pushed this round** |

`git status --short` before any change:

```
 M docs/2026-10-14-data-and-experiment-decisions.md
 M src/industry_load.py
 M tests/test_industry_load.py
?? data/processed/industry_meter_mapping_manifest.json
?? reports/industry_load_unit_gate.md
?? reports/industry_meter_mapping_s43.md
?? src/map_industry_meters.py
?? tests/test_map_industry_meters.py
```

These are the uncommitted S4.3 working-tree changes. They are preserved, not
overwritten, and nothing is committed or pushed without explicit permission.

`git log -5 --oneline`:

```
9f61c9a Record the screening manifest beside the processed load artifacts
bbc6d67 Freeze reading B for the absolute load, gate the module on it, and record its units
09f6781 Report the metered block screening and the unit the qualified count belongs to
42e1bbb Keep the paper working copies and temporary notes out of version control
00eab21 Record the load, price, network and accounting rulings, and the qualified-block criterion
```

## The pre-S1 screening baseline (recorded so S1 can prove supersession)

| artefact | field | value |
|---|---|---|
| `data/processed/industry_block_screening_manifest.json` | `screening_rule_version` | `f647fa60d88c3b29` |
| same | `dataset_version` | `industry-blocks-B/310fbfb3ccde` |
| `data/processed/industry_meter_mapping_manifest.json` | (no rule-version field) | `frozen_case.rule = R3_meter_spread` |

The old frozen numbers now treated as superseded: **87 / 113** gated meters,
**39,365** qualified blocks, **84** contributing meters, **36** users, **72**
mapped meters, **29** buses in use, **312** fully covered days.

## Configuration and data paths

| | |
|---|---|
| config | `config/defaults.yaml`, `config/scenarios.yaml` |
| raw load | `docs/16类行业用户.xlsx` — sha256 `63b826d9206f239f31f977bde281253ee5d61e94b02f3d8c23897478b52aebbb`, 67,277,368 bytes |
| raw load copy | `docs/16类行业用户 - 副本.xlsx` — **identical sha256**, the same bytes |
| raw price | `docs/日前价格261009.xlsx` |
| processed load | `data/processed/` — 15 files, only `industry_typical_profiles.csv` and `industry_block_screening_manifest.json` are tracked |
| wholesale | `data/wholesale/` — `synthetic_da_v1.csv`, `synthetic_da_v1_manifest.csv`, `price_ref_calibration.json` |

## Load data as the file states it

Read from the workbook, not from a summary:

- one sheet, `Sheet3`, **117,271 data rows x 108 columns**;
- columns 1-12 are metadata: `序号 总户号 地点 用户名称 行业 合同容量 运行容量 户号 电表局编号 综合倍率 功率相序 数据日期`;
- columns 13-108 are **times of day** (`00:00:00` ... `23:45:00`), i.e. 96 slots at
  15 minutes;
- dates present: **730 distinct non-blank dates**, 2021-01-01 to 2022-12-31, plus
  **81 rows whose date is blank** (all `empty_identity`).

**No unit is stated anywhere in the workbook.** The header names no unit, the
slot columns are labelled only by time of day, and the package metadata carries
no title, subject or description (`docProps/core.xml` has creator `40285`,
last modified by `yuhuo bing`; `docProps/custom.xml` holds only a WPS build id).
The one column that describes the readings at all is `功率相序 = 有功功率`.

## Price data as the file states it

Four sheets, each 96 points, units **CNY/MWh**, `spatial_statistic:
spatial_mean` for all four:

| sheet | `temporal_statistic` | window | date |
|---|---|---|---|
| `261009` | `single_day` | 1 | 2026-10-09 (from the sheet name, by project ruling; `externally_confirmed: False`) |
| `7days` | `window_mean` | 7 | `window_start`/`window_end` both `None` |
| `30days` | `window_mean` | 30 | both `None` |
| `halfyear` | `window_mean` | 180 | both `None` |

## Real load dates and real price dates overlap: **none**

The load covers 2021-01-01 to 2022-12-31. The only price curve carrying a
calendar date is `261009` = 2026-10-09, entirely outside that range. The three
window means carry no dates at all in the workbook. **There is no date on which a
real price curve and a real load day are both in hand.**

Per the S4.3 ruling this is handled by using `261009` as a **declared price
scenario**, labelled as such in code and in every report; it is never described
as the load day's real price and supports no cross-day generalisation.

## Existing absolute-load gate status (pre-S1)

- `industry_load.READING_UNIT` is `None` and `READING_UNIT_SOURCE` is `None`.
- `require_reading_unit()` raises while that is so, but it is called **only by
  `tests/test_industry_load.py`** — no producing path calls it, and
  `absolute_mw()` does not either. The gate is therefore currently inert.
- The capacity check applies **x4**: both `_capacity_contradicted` and
  `verify_ratios` compute `max|v| * 综合倍率 * (60/SLOT_MINUTES)`. This is the
  assumption the S1 ruling replaces.

## Existing mapping and scenario policy (pre-S1)

- `src/map_industry_meters.py`: `FROZEN_RULE = R3_meter_spread`,
  `UNSUPPORTED_BUSES = (22, 29, 32)`, `FABRICATED_DATES = ("2022-05-26",)`.
- `config/scenarios.yaml` holds 12 scenarios. `load_shape: industry` is used by
  exactly one, `typical_day_industry`, whose `load_scale_mode` is still
  `native_peak`. Eight scenarios are tagged `scenario_scale_source:
  legacy_explicit` at 10 MW.
- The industry path contributes **intraday shape only**: `grid.py` sets
  `base_load = p_mw` (the IEEE33 rated bus demand) for both `native` and
  `industry`, then retargets the coincident peak to `native_peak` = 3.715 MW.
  Absolute MW from the workbook reaches the network **nowhere** today.

## Rebuild entry points and test set

| entry | command |
|---|---|
| load shapes + quality report | `PYTHONPATH=src python src/industry_load.py` |
| block screening | `PYTHONPATH=src python src/screen_industry_blocks.py --out data/processed` |
| candidate mapping + coverage | `PYTHONPATH=src python src/map_industry_meters.py --out data/processed` |
| evaluation | `PYTHONPATH=src python src/eval_agents.py` |

`python -m pytest --collect-only -q` -> **942/994 tests collected, 52 deselected**.
The 52 are marked `slow` (`pytest.ini`: `addopts = -m "not slow"`).

## Global grep results required by the task book

Counts exclude `__pycache__`. Files are the live sources only.

| pattern | live files | note |
|---|---|---|
| `READING_UNIT\|READING_UNIT_SOURCE\|require_reading_unit` | `industry_load.py`, `map_industry_meters.py`, `tests/test_industry_load.py` | gate defined; called by tests only |
| `industry_meter_bus_map\|UNSUPPORTED_BUSES\|frozen_case` | `map_industry_meters.py`, `tests/test_map_industry_meters.py` | present |
| `tradeable_box_share\|TRADEABLE_BOX_SHARE` | `action_box_sensitivity.py`, `compare_price_sources.py`, `diagnose_typical_day.py`, `ess_siting.py`, `market_opportunity.py`, 2 tests | used as an eligibility test |
| `formal_ensemble\|RL_SCENARIOS\|EXPORT_SCENARIOS\|DEFAULT_TRAIN_SCENARIO` | `run.py`, `train_rl.py`, `ui/sim_page.py`, `tests/test_rl_defaults.py` | **`formal_ensemble` does not exist**; three modules and `train_rl` each keep their own list |
| `day_rotation\|cycle_days\|synthetic_v1\|LIBRARY_DAYS\|ENSEMBLE_SIZE` | `day_rotation.py`, `train_rl.py`, `market_opportunity.py`, `wholesale_price.py`, `synthetic_scenarios.py`, `grid.py`, `counterfactual_eval.py`, `compare_price_sources.py`, `action_box_sensitivity.py`, `audit_protocol_smoke.py`, `config/*.yaml`, 4 tests | the 40-day system |
| `CALIBRATED_DEV_PENALTY\|calibrate_dev_penalty` | `calibrate_dev_penalty.py`, `rl_td3.py`, `smoke_matd3.py`, `train_rl.py`, 2 tests | present |
| `pins_terminal_soc\|horizon_type` | `dispatch_core.py`, `dispatch_ldf.py`, `dispatch_socp.py`, `market.py`, `models.py`, `rl_env.py`, `counterfactual_eval.py`, `market_opportunity.py`, `diagnose_typical_day.py`, `compare_price_sources.py`, 2 audits, 3 tests | present; some callers omit `horizon_type` |
| `lmp_fallbacks\|grid_carbon_tco2\|_PROVIDER_CACHE\|reset_cache` | `dispatch_socp.py`, `dispatch_ldf.py`, `dispatch_core.py`, `market_opportunity.py`, `wholesale_price.py`, `phase7_*`, `phase9_*`, several audits, 9 tests | present |
| `hash(a.name)\|_compute_opponent_features` | `dispatch_core.py`, `rl_env.py`, `strategies/random_bidding.py`, `strategies/rl_bidding.py`, `tests/test_rl_spec.py` | present |

**Two patterns from the task book do not exist in the repository:**
`formal_ensemble` and the defect labels `A2–A6 / B1–B6 / C1–C5`. The labels are
from the task book; the mapping from each label to repository symbols was
established by inspection and is recorded in the implementation plan.

## Not confirmed by any source, and therefore recorded as assumption

1. **The unit of a slot reading.** The workbook states none; the S1 ruling
   declares kW. This is a **declared assumption**, not a data-dictionary fact.
2. **The unit of `运行容量` and `合同容量`.** Also unstated. The capacity check
   compares a peak against `运行容量`, so it is only meaningful if both are the
   same power unit — the same assumption, applied twice.
3. **The dates behind the three window-mean price curves.** `window_start` and
   `window_end` are `None` in the workbook; the module records them as unknown
   rather than inventing them.
4. **Line thermal ratings, the PCC rating and the reverse-flow limit.** Their
   basis — IEEE33 original vs equipment data vs synthetic assumption — is not
   established by this audit and must be labelled in the S8 validation report.
5. **Whether a user's several meters are additive circuits.** The absolute path
   sums them; the workbook cannot confirm that they measure disjoint load.

## Working-tree artefacts this audit did not touch

`reports/` holds 18 earlier reports, the newest being
`industry_meter_mapping_s43.md`, `industry_load_unit_gate.md`,
`industry_load_screening.md`. Several predate this round and quote the pre-S1
numbers; S1 marks the affected ones superseded rather than editing them in place.
